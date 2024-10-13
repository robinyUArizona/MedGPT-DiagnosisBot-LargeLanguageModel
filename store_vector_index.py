from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
from src.logger import logging

# Load environment variables from .env
load_dotenv()

# Load Data From the PDF File
extracted_data = load_pdf_file(data="data/")
logging.info(f"==== PDF file loading completed ====")

# Split the Data into Text Chunks
text_chunks = text_split(extracted_data)
logging.info(f"Number of document chunks: {len(text_chunks)}")
logging.info(f"Sample chunk:\n{text_chunks[0].page_content}\n")
logging.info(f"====  PDF file content splitted into chunk completed ====")

# Download the Embedding model from Hugging Face
embeddings = download_hugging_face_embeddings()
logging.info(f"==== Embedding models downloaded ====")

# Initialize Pinecone gRPC client instance
pc = Pinecone()
# Define the index name for the medical chatbot embeddings
index_name = "medicalchatbot"
pc.create_index(
    name = index_name,            # Name of the index to be created
    dimension = 384,              # Dimension size of the embeddings (replace with your model's embedding size)
    metric = "cosine",            # Distance metric for nearest neighbor search (e.g., cosine similarity)
    spec = ServerlessSpec(        # Serverless configuration
        cloud="aws",            # Cloud provider where the index is hosted
        region="us-east-1"      # Specific region to deploy the index
    )
)
logging.info(f"Index '{index_name}' has been created.")

# Embed each chunk and insert (upsert) the embeddings into your Pinecone index.
doc_DB = PineconeVectorStore.from_documents(
    documents = text_chunks,
    embedding = embeddings, 
    index_name = index_name  
)



