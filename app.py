import os
import logging
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from langchain_groq import ChatGroq
from langchain.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain,create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HF_TOKEN")
file_path = os.getenv("PDF_FILE_PATH", "C:/Users/goyal/Desktop/project/sewak_ai/data/data.pdf")

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize models globally
try:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")  # Ensure model name is correct
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
except Exception as e:
    logging.error(f"Failed to initialize models: {e}")
    raise RuntimeError("Model initialization failed.")

# Flask App Setup
app = Flask(__name__)

# Load and process PDF
def load_and_process_pdf(file_path):
    """Loads a PDF file and processes it into text chunks."""
    try:
        loader = PyPDFLoader(file_path)
        data = loader.load()
        logging.info("Successfully loaded and processed PDF.")
        return data
    except Exception as e:
        logging.error(f"Error loading PDF: {e}")
        raise RuntimeError("Failed to load the PDF. Check the file path and format.") from e

data = load_and_process_pdf(file_path)

# Create vectorstore with persistence
vectorstore_path = "db"
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=80)

if not os.path.exists(vectorstore_path):  # Avoid recomputing embeddings if available
    splits = text_splitter.split_documents(data)
    vectorstore = Chroma.from_documents(splits, embedding_model, persist_directory=vectorstore_path)
    vectorstore.persist()
else:
    vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embedding_model)

retriever = vectorstore.as_retriever()

# Define system prompt
rag_prompt = (
    "You are Sewak AI, an AI assistant for the Constitution of India. You will provide answers in a simple, easy-to-understand format. "
    "For every question, follow these instructions:"
    "Start with the Article number and title in bold"
    "Use bullet points new line point."
    "Bold important words"
    "give the complete information for every question"
    "Avoid long paragraphs. Each explanation should be concise and separated into clear points"
    "Do not add unnecessary information or opinions"
    "\n{context}"
)

# Create RAG prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", rag_prompt),
    ("human", "{input}"),
])

# Create question-answer chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Contextualizing system prompt
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question, "
    "which might reference context in the chat history, "
    "formulate a standalone question that can be understood "
    "without the chat history. Do not answer the question, "
    "just reformulate it if needed, otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Create history-aware retriever
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# Define QA prompt with chat history
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", rag_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Create final RAG chain with memory
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Initialize chat memory
memory = ConversationBufferMemory(return_messages=True)

# Flask Routes

@app.route("/")
def index():
    return render_template("index.html")  # Simple home page with input/output

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.form.get("user_input")
    
    if user_input:
         try:
              response = rag_chain.invoke({
                   "input": user_input, 
                   "chat_history": memory.chat_memory.messages
                   })
              answer = response.get("answer", "Sorry, I couldn't process that request.")
              
              memory.chat_memory.add_user_message(user_input)
              memory.chat_memory.add_ai_message(answer)
              return jsonify({"answer": answer})
         except Exception as e:
            return jsonify({"error": f"An error occurred: {e}"})
    else:
        return jsonify({"error": "No input provided."})

if __name__ == "__main__":
    app.run(debug=True)