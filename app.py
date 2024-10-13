from flask import Flask, render_template, jsonify, request
from src.logger import logging
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.prompt import (
    system_prompt, 
    contextualize_q_system_prompt, 
    contextualize_q_system_prompt)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.messages import HumanMessage, SystemMessage

# Initialize Flask
app = Flask(__name__)

# Load environment variables from .env
load_dotenv()

# Define the index name for the medical chatbot embeddings
index_name = "medicalchatbot"
# Get embedding model - Hugging Face
embeddings = download_hugging_face_embeddings()
# Load Existing index from Pinecone (vector store) index 
doc_DB = PineconeVectorStore.from_existing_index(
    index_name = index_name,
    embedding = embeddings
)

# Define the user's question
query = "What is Acne?"

# Retrieve relevant documents based on the query
retriever = doc_DB.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)
relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
logging.info("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    logging.info(f"Document {i}:\n{doc.page_content}\n")
    logging.info(f"Source: {doc.metadata['source']}\n")

# Create a ChatOpenAI model
llm = OpenAI(temperature=0.4, max_tokens=500)

# Prompt with System and Human Messages (Using Tuples)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create a chain to combine documents for question answering
# `create_stuff_documents_chain` feeds all retrieved context into the LLM
question_answer_chain = create_stuff_documents_chain(llm, prompt)
# Create a retrieval chain that combines the history-aware retriever and the question answering chain
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug=True)