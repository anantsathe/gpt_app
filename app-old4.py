#!/usr/bin/env python3
# Dependencies: fastapi, uvicorn, openai, zipfile36, sqlite3

from fastapi import FastAPI, File, UploadFile, Form
import os
import zipfile
import requests
import subprocess
import importlib.util
import sqlite3
import re
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

def run_prettier_and_hash(file_path):
    """Run Prettier on the file and return the SHA256 checksum of the output."""
    try:
        # Run Prettier and pipe output to sha256sum
        result = subprocess.run(
            ["npx", "-y", "prettier@3.4.2", file_path],
            capture_output=True, text=True, check=True
        )
        formatted_output = result.stdout

        # Compute SHA256 hash of the formatted output
        sha256_result = subprocess.run(
            ["sha256sum"],
            input=formatted_output, capture_output=True, text=True, check=True
        )

        return sha256_result.stdout.strip().split()[0]  # Extract hash value

    except subprocess.CalledProcessError as e:
        return f"Error running prettier: {e}"

@app.post("/api/")
async def solve_assignment(question: str = Form(...), file: UploadFile = None):
    """ Main API function to handle SQL, file-based, and shell command questions """

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

        # ✅ Handle Prettier formatting request
        if "prettier" in question.lower() and "sha256sum" in question.lower():
            hash_value = run_prettier_and_hash(file_path)
            return {"answer": hash_value}

        # If the request is related to listing files and attributes
        if "list files" in question.lower() or "total size" in question.lower():
            output_folder = "extracted_files"
            extract_zip_with_timestamps(file_path, output_folder)
            total_size = calculate_filtered_size(output_folder)
            return {"answer": str(total_size)}

    return {"error": "Invalid request"}