import argparse
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from get_embedding_function import get_embedding_function

load_dotenv()  # loads variables from .env

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma")
DATA_PATH = os.path.join(BASE_DIR, "data")


PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # Load embedding function (you can replace with Gemini embeddings)
    embedding_function = get_embedding_function()

    # Load vector DB
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search vectors
    results = db.similarity_search_with_score(query_text, k=5)

    # Format context
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Build prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # ---- GEMINI LLM HERE ----
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",   # or gemini-1.5-pro
        temperature=0.2
    )
    response_text = model.invoke(prompt)
    # ---------------------------

    # Extract sources
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"

    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()
