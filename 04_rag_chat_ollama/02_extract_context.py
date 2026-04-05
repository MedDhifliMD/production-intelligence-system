#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import pdfplumber
from sqlalchemy import create_engine, text

# -------------------------
# CONFIG
# -------------------------
DB_USER = "postgres"
DB_PASS = "*****"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "******"
TABLE_NAME = "******"  # your main table
PDF_PATH = r"C:\Users\Issa\source\repos\mygit\Glory_AI\dataset\Predictive-AI\ai_genarate_query\postgresql-16-A4.pdf"
CONTEXT_FILE = "rag_context.pkl"

# -------------------------
# 1️⃣ GET SCHEMA FROM POSTGRES
# -------------------------
print(" Connecting to PostgreSQL...")
engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

with engine.connect() as conn:
    result = conn.execute(text(f"""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = '{TABLE_NAME}';
    """))
    columns = result.fetchall()

schema_text = f"### TABLE SCHEMA - VERY IMPORTANT\nTable: {TABLE_NAME}\nColumns:\n"
for col_name, data_type in columns:
    schema_text += f"- {col_name}: {data_type}\n"

print(" Table schema extracted from PostgreSQL.")

# -------------------------
# 2️⃣ EXTRACT PDF TEXT
# -------------------------
print("📄 Extracting PDF text...")
pdf_text = ""
with pdfplumber.open(PDF_PATH) as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            pdf_text += text + "\n"

# -------------------------
# 3️⃣ COMBINE CONTEXT
# -------------------------
full_text = schema_text + "\n\n" + pdf_text

# -------------------------
# 4️⃣ SAVE CONTEXT
# -------------------------
with open(CONTEXT_FILE, "wb") as f:
    pickle.dump(full_text, f)

print(f" Context saved to '{CONTEXT_FILE}'")