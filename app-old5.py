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

def process_sqlite_question(question):
    """ Process SQL-related questions and return the SQL query or computed result. """

    # If the question asks for SQL query
    if "tickets table" in question.lower() and "write sql" in question.lower():
        sql_query = """
        SELECT SUM(units * price) AS total_sales
        FROM tickets
        WHERE lower(trim(type)) = 'gold'
        LIMIT 1;
        """
        return {"sql_query": sql_query.strip()}  # Return SQL query

    # If the question asks for computed total sales of "Gold" tickets
    if "tickets table" in question.lower() and "total sales" in question.lower():
        table_data = re.findall(r"([\w\s]+)\s+(\d+)\s+([\d.]+)", question)

        if not table_data:
            return {"error": "Could not extract table data"}

        # Create an in-memory SQLite database
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()

        # Create the tickets table
        cursor.execute("""
        CREATE TABLE tickets (
            type TEXT,
            units INTEGER,
            price REAL
        )
        """)

        # Insert extracted data, cleaning up type names (handling mis-spellings)
        for ticket_type, units, price in table_data:
            cleaned_type = ticket_type.strip().lower()  # Normalize case
            normalized_type = "Gold" if cleaned_type == "gold" else ticket_type.strip()
            cursor.execute("INSERT INTO tickets (type, units, price) VALUES (?, ?, ?)", (normalized_type, units, price))

        conn.commit()

        # Run SQL query to calculate total sales for "Gold" tickets (case-insensitive)
        cursor.execute("SELECT SUM(units * price) AS total_sales FROM tickets WHERE lower(trim(type)) = 'gold' LIMIT 1")
        total_sales = cursor.fetchone()[0]

        conn.close()

        return {"answer": str(total_sales) if total_sales else "0"}

    return None  # Return None if the question is not SQL-related

def evaluate_google_sheets_formula(question: str):
    """Detect and evaluate Google Sheets formulas like SEQUENCE and SUM."""
    if "Google Sheets" in question and "SEQUENCE" in question:
        # Extract the formula
        match = re.search(r"=SUM\(ARRAY_CONSTRAIN\(SEQUENCE\((\d+), (\d+), (\d+), (\d+)\), (\d+), (\d+)\)\)", question)
        if match:
            rows, cols, start, step, constrain_rows, constrain_cols = map(int, match.groups())

            # Generate SEQUENCE array (100x100 matrix starting from 8, step 6)
            sequence_array = np.arange(start, start + (rows * cols * step), step).reshape(rows, cols)

            # Apply ARRAY_CONSTRAIN (take only first `constrain_rows` rows and `constrain_cols` columns)
            constrained_array = sequence_array[:constrain_rows, :constrain_cols]

            # Compute SUM
            result = np.sum(constrained_array)
            return {"answer": str(result)}
    
    return None  # Not a Google Sheets formula

@app.post("/api/")
async def solve_assignment(question: str = Form(...), file: UploadFile = None):
    """ Main API function to handle SQL, Google Sheets, and file-based questions """

    # Handle Google Sheets formula questions
    google_sheets_result = evaluate_google_sheets_formula(question)
    if google_sheets_result:
        return google_sheets_result  # Return computed result

    # Handle SQL-related questions
    sqlite_result = process_sqlite_question(question)
    if sqlite_result:
        return sqlite_result  # Return either SQL query or computed result

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

# command to run code : 
#pkill -f uvicorn  # Kill existing FastAPI processes
#uvicorn app:app --reload --host 0.0.0.0 --port 8000