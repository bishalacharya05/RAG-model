from django.shortcuts import render
import os, sys

# Add the parent folder (rag-tutorial-v2) to Python path so we can import query_data
import os
import sys
from django.shortcuts import render

# Add the path to your rag-tutorial-v2 folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from query_data import query_rag  # Now Python can find it
 # your existing RAG function

def home(request):
    answer = ""
    question = ""

    if request.method == "POST":
        question = request.POST.get("question")
        if question:
            raw_answer = query_rag(question)

            # Extract clean text instead of printing the whole object
            if hasattr(raw_answer, "content"):
                answer = raw_answer.content
            else:
                answer = raw_answer

    return render(request, "UI/home.html", {"question": question, "answer": answer})
