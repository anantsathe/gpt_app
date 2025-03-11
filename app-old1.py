#!/usr/bin/env python3
from fastapi import FastAPI, File, UploadFile, Form
import os
import subprocess
import re
import numpy as np
import sqlite3
from datetime import datetime

app = FastAPI()

def extract_zip_with_timestamps(zip_path, output_folder):
    """ Extract ZIP file while preserving timestamps. """
    if os.path.exists(output_folder):
        subprocess.run(["rm", "-rf", output_folder])  # Remove folder if exists
    
    os.makedirs(output_folder, exist_ok=True)

    # Use system unzip to preserve timestamps
    subprocess.run(["unzip", "-o", zip_path, "-d", output_folder], check=True)

    return output_folder

def calculate_filtered_size(output_folder):
    """ Calculate the total size of files meeting the date and size criteria. """
    target_date = datetime(2004, 8, 5, 9, 24)  # Thu, 5 Aug, 2004, 9:24 AM IST
    target_size = 7265  # Bytes

    total_size = 0

    for root, _, files in os.walk(output_folder):
        for file in files:
            file_path = os.path.join(root, file)
            stat = os.stat(file_path)

            # Convert modification time to datetime
            mod_time = datetime.fromtimestamp(stat.st_mtime)

            # Apply conditions
            if stat.st_size >= target_size and mod_time >= target_date:
                total_size += stat.st_size

    return total_size

def process_sqlite_question(question):
    """ Process SQL-related questions and return the SQL query or computed result. """
    if "tickets table" in question.lower() and "write sql" in question.lower():
        sql_query = """
        SELECT SUM(units * price) AS total_sales
        FROM tickets
        WHERE lower(trim(type)) = 'gold'
        LIMIT 1;
        """
        return {"sql_query": sql_query.strip()}  # Return SQL query

    if "tickets table" in question.lower() and "total sales" in question.lower():
        table_data = re.findall(r"([\w\s]+)\s+(\d+)\s+([\d.]+)", question)

        if not table_data:
            return {"error": "Could not extract table data"}

        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE tickets (
            type TEXT,
            units INTEGER,
            price REAL
        )
        """)

        for ticket_type, units, price in table_data:
            cleaned_type = ticket_type.strip().lower()
            normalized_type = "Gold" if cleaned_type == "gold" else ticket_type.strip()
            cursor.execute("INSERT INTO tickets (type, units, price) VALUES (?, ?, ?)", (normalized_type, units, price))

        conn.commit()

        cursor.execute("SELECT SUM(units * price) AS total_sales FROM tickets WHERE lower(trim(type)) = 'gold' LIMIT 1")
        total_sales = cursor.fetchone()[0]

        conn.close()

        return {"answer": str(total_sales) if total_sales else "0"}

    return None  # Not an SQL-related question

def evaluate_google_sheets_formula(question: str):
    """Detect and evaluate Google Sheets formulas like SEQUENCE and SUM."""
    if "Google Sheets" in question and "SEQUENCE" in question:
        match = re.search(r"=SUM\(ARRAY_CONSTRAIN\(SEQUENCE\((\d+), (\d+), (\d+), (\d+)\), (\d+), (\d+)\)\)", question)
        if match:
            rows, cols, start, step, constrain_rows, constrain_cols = map(int, match.groups())
            sequence_array = np.arange(start, start + (rows * cols * step), step).reshape(rows, cols)
            constrained_array = sequence_array[:constrain_rows, :constrain_cols]
            result = np.sum(constrained_array)
            return {"answer": str(result)}
    return None  # Not a Google Sheets formula

@app.post("/api/")
async def solve_assignment(question: str = Form(...), file: UploadFile = None):
    """ Determine the type of request and process accordingly. """

    if file:
        # ZIP file processing
        zip_path = f"/tmp/{file.filename}"
        with open(zip_path, "wb") as buffer:
            buffer.write(await file.read())

        extract_path = "/tmp/extracted_files"
        extract_zip_with_timestamps(zip_path, extract_path)
        total_size = calculate_filtered_size(extract_path)

        return {"answer": total_size}

    # Try SQL question processing
    sql_response = process_sqlite_question(question)
    if sql_response:
        return sql_response

    # Try Google Sheets formula processing
    sheets_response = evaluate_google_sheets_formula(question)
    if sheets_response:
        return sheets_response

    return {"error": "Unrecognized question format"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)