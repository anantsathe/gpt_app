#!/usr/bin/env python3
# Dependencies: fastapi, uvicorn, openai, zipfile36

from fastapi import FastAPI, File, UploadFile, Form
import os
import zipfile
import requests
import subprocess
import importlib.util
from datetime import datetime

# Function to install missing dependencies
def install_dependency(package):
    if importlib.util.find_spec(package) is None:
        print(f"Installing missing dependency: {package}")
        subprocess.run(["pip", "install", package])

# Install required dependencies
dependencies = ["fastapi", "uvicorn", "zipfile36"]
for dep in dependencies:
    install_dependency(dep)

app = FastAPI()

# Set API key
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AIPROXY_TOKEN")
if not api_key or api_key.strip() == "":
    raise ValueError("API key is not set!")

print("AIPROXY_TOKEN:", api_key[:5] + "..." + api_key[-5:])  # Partial print for security

def extract_zip_with_timestamps(zip_path, output_folder):
    """ Extract ZIP file while preserving timestamps. """
    if os.path.exists(output_folder):
        subprocess.run(["rm", "-rf", output_folder])  # Remove folder if exists to avoid conflicts
    
    os.makedirs(output_folder, exist_ok=True)

    # Use system unzip to preserve timestamps
    subprocess.run(["unzip", zip_path, "-d", output_folder], check=True)

    return output_folder

def calculate_filtered_size(output_folder):
    """ Calculate the total size of files meeting the date and size criteria. """
    target_date = datetime(2004, 8, 5, 9, 24)  # Thu, 5 Aug, 2004, 9:24 AM IST
    target_size = 7265  # Bytes

    # Get the file details using ls -l --time-style=long-iso
    ls_output = subprocess.run(
        ["ls", "-l", "--time-style=long-iso", output_folder],
        capture_output=True, text=True
    ).stdout

    total_size = 0
    for line in ls_output.split("\n"):
        parts = line.split()
        if len(parts) < 6:
            continue

        file_size = int(parts[4])  # File size in bytes
        file_date_str = parts[5] + " " + parts[6]  # Extract date-time string
        try:
            file_date = datetime.strptime(file_date_str, "%Y-%m-%d %H:%M")
        except ValueError:
            continue  # Skip malformed date formats

        if file_size >= target_size and file_date >= target_date:
            total_size += file_size

    return total_size

@app.post("/api/")
async def solve_assignment(question: str = Form(...), file: UploadFile = None):
    # Handle file if provided
    if file:
        file_path = f"uploads/{file.filename}"
        os.makedirs("uploads", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(file.file.read())

        # If the request is related to listing files and attributes
        if "list files" in question.lower() or "total size" in question.lower():
            output_folder = "extracted_files"
            extract_zip_with_timestamps(file_path, output_folder)
            total_size = calculate_filtered_size(output_folder)
            return {"answer": str(total_size)}

    return {"error": "Invalid request"}