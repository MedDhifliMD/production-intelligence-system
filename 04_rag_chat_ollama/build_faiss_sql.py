#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_faiss_sql.py
──────────────────
Builds a FAISS vectorstore that contains:
  1. Table schema (from PostgreSQL live query)
  2. Table explanation + sample rows (hardcoded domain knowledge)
  3. Full PostgreSQL 16 PDF documentation
  4. Hand-crafted SQL Q&A pairs for this specific table

Run once. Saves index to:  faiss_sql/
"""

import os
from sqlalchemy import create_engine, text
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ──────────────────────────────────────────────
# CONFIG  ← change these if needed
# ──────────────────────────────────────────────
DB_USER    = "postgres"
DB_PASS    = "123"
DB_HOST    = "localhost"
DB_PORT    = "5432"
DB_NAME    = "ai_db"
TABLE_NAME = "ai_data"

PDF_PATH   = r"C:\Users\Issa\source\repos\mygit\Glory_AI\dataset\Predictive-AI\ai_genarate_query\postgresql-16-A4-1-568.pdf"
OUTPUT_DIR = "faiss_sql"

EMBED_MODEL = "all-MiniLM-L6-v2"

# ──────────────────────────────────────────────
# 1. LIVE SCHEMA FROM POSTGRESQL
# ──────────────────────────────────────────────
print("🔌 Connecting to PostgreSQL and extracting schema...")
engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

with engine.connect() as conn:
    cols = conn.execute(text(f"""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = '{TABLE_NAME}'
        ORDER BY ordinal_position;
    """)).fetchall()

schema_lines = [
    f"### POSTGRESQL TABLE SCHEMA",
    f"Table name: {TABLE_NAME}",
    f"Database: {DB_NAME}",
    "",
    "Columns (name | SQL type | nullable):",
]
for col_name, data_type, nullable in cols:
    schema_lines.append(f"  - \"{col_name}\" : {data_type}  (nullable={nullable})")

schema_lines += [
    "",
    "IMPORTANT RULES for querying this table:",
    "- Always wrap column names in double quotes: \"ColumnName\"",
    "- Always wrap string values in single quotes: 'value'",
    "- Table name never needs quoting: ai_data",
    "- \"NPM_Date\" and \"Verif_Date\" are stored as TEXT in ISO-8601 format",
    "  e.g. '2024-10-21T13:06:32+01:00'",
    "  To filter by date use:  \"NPM_Date\" LIKE '2024-10-21%'",
    "  Or cast:  DATE(\"NPM_Date\"::timestamptz) = '2024-10-21'",
    "  Or range: \"NPM_Date\" >= '2025-10-01' AND \"NPM_Date\" < '2025-10-11'",
    "- \"Barcode\" is bigint — do NOT quote its values",
    "- \"Componenet_Result\" values are exactly: 'Pass' or 'Fail'",
    "- \"Has_Verification\" and \"Has_Component_Verification\" are boolean",
]

schema_text = "\n".join(schema_lines)
print(schema_text)

# ──────────────────────────────────────────────
# 2. DOMAIN KNOWLEDGE + TABLE EXPLANATION
# ──────────────────────────────────────────────
domain_knowledge = """
### DOMAIN KNOWLEDGE — NPMVF MACHINE TABLE

This table stores data produced by a NPMVF (pick-and-place) machine in a PCB assembly line.

KEY CONCEPTS:
- "Barcode": unique ID of a PCB card produced. One Barcode can span many rows.
- "NPM_Date": timestamp when the Barcode (PCB card) was produced by the NPMVF machine. Stored as TEXT ISO-8601.
- "Verif_Date": timestamp when the card was tested at the verification station. Stored as TEXT ISO-8601.
- "Pattern_Barcode": the name/ID of the assembly pattern used for the card. One Barcode can have 0 to N Pattern_Barcodes.
- "Pattern_Index": numeric index of the pattern (1, 2, 3...).
- "Designator": a component placed on the card (e.g. C120, L850, R001). Each row = one Designator on one card.
  So one Barcode has as many rows as it has Designators × Pattern_Barcodes.
- "Nozel_Name": the nozzle on the NPMVF machine that placed the Designator (e.g. NZ002, NZ003).
- "Feede_ID": the feeder on the NPMVF machine responsible for supplying the Designator component.
- "Componenet_Result": placement result for that Designator — either 'Pass' or 'Fail'.
- "Coordinate_X", "Coordinate_Y": physical coordinates (mm) where the component was placed on the card.
- "Rotation": rotation angle of the component in degrees.
- "Has_Verification": boolean — whether the card was sent to verification station at all.
- "Has_Component_Verification": boolean — whether this specific Designator was tested at verification.
- "DefectCode": reason for defect if Fail (e.g. 'Pseudofehler'). NULL if Pass.

IMPORTANT COUNTING RULES:
- To count UNIQUE cards produced: COUNT(DISTINCT "Barcode")
- To count total component placements: COUNT(*)
- To count unique cards per day: GROUP BY DATE("NPM_Date"::timestamptz)
- One card (Barcode) appears in many rows because it has many Designators

### SAMPLE DATA (real values from the table):

Barcode             | NPM_Date                      | Pattern_Barcode               | Idx | Designator | Result | Feede_ID         | Nozel | X       | Y       | Rot | HasVerif | HasCompVerif | DefectCode
50122816709014171   | 2024-10-21T13:06:32+01:00     | CE65AK9P5A08002428428195      | 2   | L850       | Pass   | FA0299503000001  | NZ002 | 23.696  | 42.056  | -90 | true     | false        | NULL
50122816709014171   | 2024-10-21T13:06:32+01:00     | CE65AK9P5A08002428428195      | 2   | C121       | Pass   | FE0071AA1080085  | NZ002 | 29.932  | 120.178 | 0   | true     | false        | NULL
50122816709014171   | 2024-10-21T13:06:32+01:00     | CE65AK9P5A08002428428195      | 2   | C120       | Pass   | FE0071AA1080085  | NZ002 | 72.712  | 121.958 | 0   | true     | false        | NULL
50122816709014171   | 2024-10-21T13:06:32+01:00     | CE65AK9P5A08002428428194      | 1   | L850       | Pass   | FA0299503000001  | NZ002 | 206.374 | 140.254 | 90  | true     | false        | NULL
50122817232816924   | 2024-10-21T13:48:00+01:00     | CE65AK9P5A08002428833701      | 2   | C120       | Pass   | FE0071AA1080085  | NZ003 | 72.712  | 121.958 | 0   | true     | true         | Pseudofehler
50122817232816605   | 2024-10-21T14:13:32+01:00     | CE65AK9P5A08002428833063      | 2   | L850       | Pass   | FA0299503000001  | NZ003 | 23.696  | 42.056  | -90 | true     | true         | Pseudofehler
"""

# ──────────────────────────────────────────────
# 3. HAND-CRAFTED SQL Q&A PAIRS (few-shot)
# ──────────────────────────────────────────────
sql_examples = """
### SQL EXAMPLES FOR ai_data TABLE

Q: How many unique barcodes were produced each day between 2025-10-01 and 2025-10-10?
SQL:
SELECT DATE("NPM_Date"::timestamptz) AS day, COUNT(DISTINCT "Barcode") AS unique_barcodes
FROM ai_data
WHERE "NPM_Date" >= '2025-10-01' AND "NPM_Date" < '2025-10-11'
GROUP BY DATE("NPM_Date"::timestamptz)
ORDER BY day;

Q: Count total Pass and Fail in Componenet_Result
SQL:
SELECT "Componenet_Result", COUNT(*) AS total
FROM ai_data
WHERE "Componenet_Result" IN ('Pass', 'Fail')
GROUP BY "Componenet_Result";

Q: Which 3 Feede_ID have the most Pass results?
SQL:
SELECT "Feede_ID", COUNT(*) AS pass_count
FROM ai_data
WHERE "Componenet_Result" = 'Pass'
GROUP BY "Feede_ID"
ORDER BY pass_count DESC
LIMIT 3;

Q: How many unique barcodes produced per day in October 2024?
SQL:
SELECT DATE("NPM_Date"::timestamptz) AS day, COUNT(DISTINCT "Barcode") AS unique_barcodes
FROM ai_data
WHERE "NPM_Date" >= '2024-10-01' AND "NPM_Date" < '2024-11-01'
GROUP BY DATE("NPM_Date"::timestamptz)
ORDER BY day;

Q: What is the fail rate per Nozel_Name?
SQL:
SELECT "Nozel_Name",
       COUNT(*) AS total,
       SUM(CASE WHEN "Componenet_Result" = 'Fail' THEN 1 ELSE 0 END) AS fails,
       ROUND(100.0 * SUM(CASE WHEN "Componenet_Result" = 'Fail' THEN 1 ELSE 0 END) / COUNT(*), 2) AS fail_rate_pct
FROM ai_data
GROUP BY "Nozel_Name"
ORDER BY fail_rate_pct DESC;

Q: How many cards were verified (Has_Verification = true) per day?
SQL:
SELECT DATE("NPM_Date"::timestamptz) AS day, COUNT(DISTINCT "Barcode") AS verified_cards
FROM ai_data
WHERE "Has_Verification" = true
GROUP BY DATE("NPM_Date"::timestamptz)
ORDER BY day;

Q: List all distinct DefectCodes and how many times each appears
SQL:
SELECT "DefectCode", COUNT(*) AS occurrences
FROM ai_data
WHERE "DefectCode" IS NOT NULL AND "DefectCode" <> ''
GROUP BY "DefectCode"
ORDER BY occurrences DESC;

Q: How many unique barcodes were produced in total?
SQL:
SELECT COUNT(DISTINCT "Barcode") AS total_unique_barcodes
FROM ai_data;

Q: What are the top 5 Designators with the most Fail results?
SQL:
SELECT "Designator", COUNT(*) AS fail_count
FROM ai_data
WHERE "Componenet_Result" = 'Fail'
GROUP BY "Designator"
ORDER BY fail_count DESC
LIMIT 5;

Q: How many cards were produced per month in 2024?
SQL:
SELECT TO_CHAR("NPM_Date"::timestamptz, 'YYYY-MM') AS month, COUNT(DISTINCT "Barcode") AS unique_barcodes
FROM ai_data
WHERE "NPM_Date" >= '2024-01-01' AND "NPM_Date" < '2025-01-01'
GROUP BY TO_CHAR("NPM_Date"::timestamptz, 'YYYY-MM')
ORDER BY month;

Q: Which Pattern_Barcode has the highest number of Fail results?
SQL:
SELECT "Pattern_Barcode", COUNT(*) AS fail_count
FROM ai_data
WHERE "Componenet_Result" = 'Fail'
GROUP BY "Pattern_Barcode"
ORDER BY fail_count DESC
LIMIT 1;

Q: What is the average number of Designators per Barcode?
SQL:
SELECT ROUND(AVG(designator_count), 2) AS avg_designators_per_barcode
FROM (
    SELECT "Barcode", COUNT(*) AS designator_count
    FROM ai_data
    GROUP BY "Barcode"
) sub;
"""

# ──────────────────────────────────────────────
# 4. LOAD POSTGRESQL PDF
# ──────────────────────────────────────────────
print(f"\n📄 Loading PostgreSQL PDF: {PDF_PATH}")
pg_loader = PyPDFLoader(PDF_PATH)
pg_pages  = pg_loader.load()
print(f"   → {len(pg_pages)} pages loaded")

# ──────────────────────────────────────────────
# 5. BUILD DOCUMENTS LIST
# ──────────────────────────────────────────────
documents = []

# Schema doc — small chunk, keep whole
documents.append(Document(
    page_content=schema_text,
    metadata={"source": "live_schema", "priority": "critical"}
))

# Domain knowledge doc — keep whole
documents.append(Document(
    page_content=domain_knowledge,
    metadata={"source": "domain_knowledge", "priority": "critical"}
))

# SQL examples — keep whole
documents.append(Document(
    page_content=sql_examples,
    metadata={"source": "sql_examples", "priority": "critical"}
))

# PDF pages — split into chunks
pdf_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " "]
)
pdf_chunks = pdf_splitter.split_documents(pg_pages)
documents.extend(pdf_chunks)

print(f"\n📦 Total documents to embed:")
print(f"   - Schema:         1 doc")
print(f"   - Domain knowl.:  1 doc")
print(f"   - SQL examples:   1 doc")
print(f"   - PDF chunks:     {len(pdf_chunks)} chunks")
print(f"   TOTAL:            {len(documents)} documents")

# ──────────────────────────────────────────────
# 6. EMBED AND SAVE FAISS
# ──────────────────────────────────────────────
print(f"\n⏳ Generating embeddings with '{EMBED_MODEL}'...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local(OUTPUT_DIR)

print(f"\n✅ SQL FAISS index saved to: '{OUTPUT_DIR}/'")
print(f"   Files: {os.listdir(OUTPUT_DIR)}")