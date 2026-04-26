from query_data import query_rag
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))

# Read Gemini API key from environment (set in .env)
if not os.getenv("GOOGLE_API_KEY"):
    raise EnvironmentError("Missing GOOGLE_API_KEY. Add it to your .env file.")

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""

# ------------------------
# Django-Related Tests
# ------------------------

def test_django_framework_definition():
    return query_and_validate(
        question="What is Django? (Answer in one sentence)",
        expected_response="Django is a high-level Python web framework that encourages rapid development and clean design.",
    )

def test_mvt_architecture():
    return query_and_validate(
        question="What does MVT stand for in Django? (Answer with the full form only)",
        expected_response="Model View Template",
    )

def test_django_default_database():
    return query_and_validate(
        question="What is the default database used by Django? (Answer one word only)",
        expected_response="SQLite",
    )

def test_django_manage_py():
    return query_and_validate(
        question="What file is used to run Django commands and manage the project? (Answer with the filename only)",
        expected_response="manage.py",
    )

def test_django_template_language():
    return query_and_validate(
        question="What is the name of the templating system used in Django? (Answer one phrase)",
        expected_response="Django Template Language",
    )

def test_django_settings_file():
    return query_and_validate(
        question="Which file contains the configuration settings for a Django project? (Answer filename only)",
        expected_response="settings.py",
    )


# ------------------------
# Evaluation Logic
# ------------------------

def query_and_validate(question: str, expected_response: str):
    response_text = query_rag(question)
    
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response,
        actual_response=response_text
    )

    # ---- GEMINI LLM ----
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  # or "gemini-1.5-pro"
        temperature=0.2
    )
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(prompt)

    if "true" in evaluation_results_str_cleaned:
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError("Invalid evaluation result. Cannot determine if 'true' or 'false'.")
