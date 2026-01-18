from langchain_google_genai import GoogleGenerativeAIEmbeddings


import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyDOMLFBsXHzz1CX-0ZbADatE74hedHDvjw"

def get_embedding_function():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"   # New 2025 embedding model
    )
    return embeddings

