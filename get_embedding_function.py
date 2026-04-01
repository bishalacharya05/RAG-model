from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()  # Loads variables from .env into environment

def get_embedding_function():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )
    return embeddings