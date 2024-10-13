from flask import Flask, render_template, jsonify, request
from src.logger import logging
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.prompt import system_prompt, contextualize_q_system_prompt, qa_system_prompt
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

# # Create a ChatOpenAI model
# llm = OpenAI(temperature=0.4, max_tokens=500)

# # Prompt with System and Human Messages (Using Tuples)
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ]
# )

# # Create a chain to combine documents for question answering
# # `create_stuff_documents_chain` feeds all retrieved context into the LLM
# question_answer_chain = create_stuff_documents_chain(llm, prompt)
# # Create a retrieval chain that combines the history-aware retriever and the question answering chain
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)



# Create a ChatOpenAI model
llm = ChatOpenAI(model="gpt-4o-mini")

# Contextualize question prompt
# This system prompt helps the AI understand that it should reformulate the question
# based on the chat history to make it a standalone question
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

# Create a prompt template for contextualizing questions
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


# Create a history-aware retriever
# This uses the LLM to help reformulate the question based on chat history
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


# Answer question prompt
# This system prompt helps the AI understand that it should provide concise answers
# based on the retrieved context and indicates what to do if the answer is unknown
qa_system_prompt  = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

# Create a prompt template for contextualizing questions
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a chain to combine documents for question answering
# `create_stuff_documents_chain` feeds all retrieved context into the LLM
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create a retrieval chain that combines the history-aware retriever and the question answering chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


# chat_history = []  # Collect chat history here (a sequence of messages)
# while True:
#     query = input("You: ")
#     if query.lower() == "exit":
#         break
#     # Process the user's query through the retrieval chain
#     result = rag_chain.invoke({"input": query, "chat_history": chat_history})
#     # Display the AI's response
#     print(f"AI: {result['answer']}")
#     # Update the chat history
#     chat_history.append(HumanMessage(content=query))
#     chat_history.append(SystemMessage(content=result["answer"]))


@app.route("/")
def index():
    return render_template('chat.html')

# @app.route("/get", methods=["GET", "POST"])
# def chat():
#     msg = request.form["msg"]
#     input = msg
#     print(input)
#     response = rag_chain.invoke({"input": msg})
#     print("Response : ", response["answer"])
#     return str(response["answer"])


@app.route("/get", methods=["GET", "POST"])
def chat():
    logging.info(f"Start chatting with the AI! Type 'exit' to end the conversation.")
    chat_history = []  # Collect chat history here (a sequence of messages)
    while True:
        msg = request.form["msg"]
        if msg.lower() == "exit":
            break
        input = msg
        print(input)
        # Process the user's query through the retrieval chain
        response = rag_chain.invoke({"input": msg, "chat_history": chat_history})
        print(f"AI Response : ", response["answer"])
        # Update the chat history
        chat_history.append(HumanMessage(content=msg))
        chat_history.append(SystemMessage(content=response["answer"]))
        return str(response["answer"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug=True)