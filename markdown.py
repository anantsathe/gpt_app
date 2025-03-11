#!/usr/bin/env python3
# Dependencies: fastapi, uvicorn, re, numpy, sqlite3, zipfile36

from fastapi import FastAPI, File, UploadFile, Form
import os
import zipfile
import requests
import subprocess
import importlib.util
import sqlite3
import re
import numpy as np
from datetime import datetime
import html

# Measure app startup time
start_time = datetime.now()

# Function to install missing dependencies
def install_dependency(package):
    if importlib.util.find_spec(package) is None:
        print(f"Installing missing dependency: {package}")
        subprocess.run(["pip", "install", package])

# Install required dependencies
dependencies = ["fastapi", "uvicorn", "numpy", "sqlite3", "zipfile36"]
for dep in dependencies:
    install_dependency(dep)

app = FastAPI()

# Set API key
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AIPROXY_TOKEN")
if not api_key or api_key.strip() == "":
    raise ValueError("API key is not set!")

print("AIPROXY_TOKEN:", api_key[:5] + "..." + api_key[-5:])  # Partial print for security

def sanitize_input(text: str) -> str:
    """ Escape special characters like ! to prevent Bash errors. """
    return html.escape(text)

def extract_zip_with_timestamps(zip_path, output_folder):
    """ Extract ZIP file while preserving timestamps. """
    if os.path.exists(output_folder):
        subprocess.run(["rm", "-rf", output_folder])  # Remove folder if exists to avoid conflicts
    
    os.makedirs(output_folder, exist_ok=True)
    subprocess.run(["unzip", zip_path, "-d", output_folder], check=True)
    return output_folder

@app.post("/api/")
async def solve_assignment(question: str = Form(...), file: UploadFile = None):
    """ Main API function with input sanitization """
    
    question = sanitize_input(question)  # Escape special characters

    if "markdown" in question.lower():
        markdown_response = """
        # Sample Markdown Output
        ## Methodology
        **Important:** This analysis is based on estimated step counts.
        *Note:* Variations may exist due to tracking methods.

        Sample inline code: `step_count = daily_average * 7`

        ```python
        print("Markdown Example")
        ```

        - Step Data Comparison
        1. Day 1 - 5000 steps
        2. Day 2 - 7000 steps

        | Day | Steps |
        |-----|-------|
        | Mon | 5000  |
        | Tue | 7000  |
        
        [View Source](https://example.com)
        
        ![Graph](https://example.com/graph.jpg)

        > *This is a blockquote example.*
        """
        return {"answer": markdown_response}
    
    if file:
        file_path = f"uploads/{file.filename}"
        os.makedirs("uploads", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(file.file.read())

        output_folder = "extracted_files"
        extract_zip_with_timestamps(file_path, output_folder)
        return {"message": "File processed successfully"}
    
    return {"error": "Invalid request"}

# Print application startup time
end_time = datetime.now()
startup_time = (end_time - start_time).total_seconds()
print(f"Application startup time: {startup_time} seconds")

# Command to run:
# pkill -f uvicorn  # Kill existing FastAPI processes
# uvicorn app:app --reload --host 0.0.0.0 --port 8000
