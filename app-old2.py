#!/usr/bin/env python3
# Dependencies: fastapi, uvicorn, openai, pandas, zipfile36

from fastapi import FastAPI, File, UploadFile, Form
import openai
import os
import zipfile
import requests
import subprocess
import importlib.util
from datetime import date, timedelta

# Function to install missing dependencies
def install_dependency(package):
    if importlib.util.find_spec(package) is None:
        print(f"Installing missing dependency: {package}")
        subprocess.run(["pip", "install", package])

# Install required dependencies
dependencies = ["fastapi", "uvicorn", "openai", "pandas", "zipfile36"]
for dep in dependencies:
    install_dependency(dep)

import pandas as pd  # Import after ensuring installation

app = FastAPI()

# Set API key
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AIPROXY_TOKEN")
if not api_key or api_key.strip() == "":
    raise ValueError("API key is not set!")

print("AIPROXY_TOKEN:", api_key[:5] + "..." + api_key[-5:])  # Partial print for security

# OpenAI API endpoints
CHAT_PROXY_URL = "https://aiproxy.sanand.workers.dev/openai/v1/"
CHAT_COMPLETIONS_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
EMBEDDINGS_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# Cache for consistent responses
cache = {}

def get_cached_answer(question):
    return cache.get(question)

def store_answer(question, answer):
    cache[question] = answer

def count_wednesdays(start_date, end_date):
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    current = start
    count = 0
    while current <= end:
        if current.weekday() == 2:  # Wednesday
            count += 1
        current += timedelta(days=1)
    return count

@app.post("/api/")
async def solve_assignment(question: str = Form(...), file: UploadFile = None):
    # Handle file if provided
    if file:
        file_path = f"uploads/{file.filename}"
        os.makedirs("uploads", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(file.file.read())

        # If the file is a ZIP, extract it
        if file.filename.endswith(".zip"):
            os.makedirs("extracted_files", exist_ok=True)
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall("extracted_files")
            extracted_csv = [f for f in os.listdir("extracted_files") if f.endswith(".csv")]
            
            if extracted_csv:
                df = pd.read_csv(f"extracted_files/{extracted_csv[0]}")
                if "answer" in df.columns:
                    return {"answer": str(df["answer"].iloc[0])}

    # Check if the question is about counting Wednesdays
    if "Wednesdays" in question and "date range" in question:
        try:
            parts = question.split(" ")
            start_date = parts[-3]
            end_date = parts[-1].strip("?")
            answer = count_wednesdays(start_date, end_date)
            return {"answer": str(answer)}
        except Exception:
            pass

    # Check cache
    cached_answer = get_cached_answer(question)
    if cached_answer:
        return {"answer": cached_answer}

    # Process question with OpenAI API
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    question_prompt = f"""
    You are a highly accurate AI specializing in calculations. 
    Please answer this question with a numerical answer only:
    {question}
    Provide only the final number without explanation.
    """
    payload = {
        "model": CHAT_MODEL,
        "messages": [{"role": "user", "content": question_prompt}],
        "temperature": 0.0,
        "top_p": 1.0
    }
    response = requests.post(CHAT_COMPLETIONS_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        final_answer = response.json()["choices"][0]["message"]["content"]
        store_answer(question, final_answer)
        return {"answer": final_answer}
    else:
        return {"error": response.text}
# command to run code : 
#pkill -f uvicorn  # Kill existing FastAPI processes
#uvicorn app:app --reload --host 0.0.0.0 --port 8000
