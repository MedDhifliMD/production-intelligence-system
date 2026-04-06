#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
chatbot.py
──────────
Main RAG chatbot. Loads both FAISS indexes built by:
  - build_faiss_sql.py     → faiss_sql/
  - build_faiss_chartjs.py → faiss_chartjs/

Usage:
  (Your question here)                    → SQL query only
  (Your question here) [chart type]       → SQL + Chart.js HTML
"""

import re
import os
import json
import webbrowser
import string

import psycopg2
from psycopg2.extras import RealDictCursor
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.completion import WordCompleter
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DB_USER    = "postgres"
DB_PASS    = "123"
DB_HOST    = "localhost"
DB_PORT    = "5432"
DB_NAME    = "ai_db"
TABLE_NAME = "ai_data"

FAISS_SQL_DIR = "faiss_sql"
EMBED_MODEL   = "all-MiniLM-L6-v2"
HTML_FOLDER      = "html"

COLUMNS_RAW = [
    "Barcode", "NPM_Date", "Verif_Date", "Pattern_Barcode",
    "Pattern_Index", "Designator", "Componenet_Result", "Feede_ID",
    "Nozel_Name", "Coordinate_X", "Coordinate_Y", "Rotation",
    "Has_Verification", "Has_Component_Verification", "DefectCode"
]
COLUMNS_QUOTED = [f'"{col}"' for col in COLUMNS_RAW]
columns_str    = ", ".join(COLUMNS_QUOTED)

os.makedirs(HTML_FOLDER, exist_ok=True)

# ──────────────────────────────────────────────
# LLM — qwen3:14b with thinking disabled
# ──────────────────────────────────────────────
llm = ChatOllama(
    model="gemma2:2b",
    temperature=0.1,
    num_predict=1024,
    extra_body={"think": False}
)

# ──────────────────────────────────────────────
# EMBEDDINGS + FAISS LOADERS
# ──────────────────────────────────────────────
print("⏳ Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

print(f"⏳ Loading SQL FAISS index from '{FAISS_SQL_DIR}'...")
sql_vectorstore = FAISS.load_local(
    FAISS_SQL_DIR, embeddings, allow_dangerous_deserialization=True
)
sql_retriever = sql_vectorstore.as_retriever(search_kwargs={"k": 5})

print("✅ SQL FAISS index loaded.\n")

# ──────────────────────────────────────────────
# PROMPT 1 — SQL generation
# ──────────────────────────────────────────────
SQL_PROMPT = """/no_think
You are a PostgreSQL 16 expert. Your job is to write a single correct SQL SELECT query.

### RELEVANT DOCUMENTATION (from PostgreSQL 16 docs + table schema):
{sql_context}

### ABSOLUTE RULES:
1. Table name: {table}
2. Wrap ALL column names in double quotes: "ColumnName"
3. Wrap ALL string values in single quotes: 'value'
4. "NPM_Date" and "Verif_Date" are TEXT stored as ISO-8601 timestamps.
   - Filter by date range: "NPM_Date" >= '2025-10-01' AND "NPM_Date" < '2025-10-11'
   - Group by day: DATE("NPM_Date"::timestamptz)
5. To count unique barcodes: COUNT(DISTINCT "Barcode")
6. Never use UNION or UNION ALL — use WHERE "Col" IN ('A','B') GROUP BY "Col"
7. If you SELECT a non-aggregate column with COUNT(*), add GROUP BY for it
8. Only use these columns: {columns}
9. Return ONLY the raw SQL query. No explanations. No markdown. No extra text.

### QUESTION:
{question}

### SQL:
"""

sql_prompt = PromptTemplate(
    template=SQL_PROMPT,
    input_variables=["sql_context", "table", "columns", "question"]
)

# ──────────────────────────────────────────────
# PROMPT 2 — Chart.js HTML generation
# ──────────────────────────────────────────────
CHART_PROMPT = """/no_think
You are a Chart.js expert. Generate a complete, self-contained HTML page with a Chart.js chart.

### RULES:
1. CDN: https://cdn.jsdelivr.net/npm/chart.js
2. Must work when opened directly in browser (no server needed)
3. Use the EXACT data values from the JSON below — do NOT change them
4. Apply the chart type and colors the user requested
5. Output ONLY raw HTML starting with <!DOCTYPE html>
6. No markdown, no code fences, no explanations outside the HTML
7. Clean professional style: white card background, padding, shadow, readable font
8. Add a descriptive <h2> title above the chart
9. For horizontal bar: use type:'bar' with options.indexAxis:'y'
10. For doughnut: use type:'doughnut'
11. Labels on X axis must show each individual value (one bar per row)

### SQL RESULT DATA (JSON):
{sql_result}

### USER CHART REQUEST:
{chart_request}

### HTML:
"""

chart_prompt = PromptTemplate(
    template=CHART_PROMPT,
    input_variables=["sql_result", "chart_request"]
)

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def execute_sql(query):
    try:
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
            user=DB_USER, password=DB_PASS
        )
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(query)
        result = cur.fetchall()
        cur.close()
        conn.close()
        return [dict(row) for row in result]
    except Exception as e:
        return f"SQL Error: {e}"


def extract_sql(llm_output):
    """Strip markdown fences, bracket noise, keep only SELECT statement."""
    text = llm_output.strip()
    text = re.sub(r"```sql", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    # Cut off anything starting with "["
    bracket = text.find("[")
    if bracket != -1:
        text = text[:bracket]
    # Extract SELECT ... block
    match = re.search(r"(SELECT\s.+?)(?:;|\Z)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def call_llm(prompt_value):
    response = llm.invoke(prompt_value)
    if hasattr(response, "content"):
        return response.content.strip()
    elif hasattr(response, "generations"):
        return response.generations[0].text.strip()
    elif isinstance(response, str):
        return response.strip()
    raise ValueError("Cannot extract text from LLM response")


def sanitize_filename(name, max_len=60):
    valid = f"-_.() {string.ascii_letters}{string.digits}"
    return "".join(c if c in valid else "_" for c in name)[:max_len]


def retrieve_context(retriever, query, label=""):
    docs = retriever.invoke(query)
    context = "\n\n---\n\n".join([d.page_content for d in docs])
    if label:
        print(f"   📚 Retrieved {len(docs)} {label} docs")
    return context


# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────
column_completer = WordCompleter(COLUMNS_QUOTED, ignore_case=True)

print("💬 RAG SQL + Chart chatbot ready! Type 'exit' to quit.\n")
print("📋 Available columns:")
print("   " + columns_str)
print("\n📌 Format:")
print("  (question)                                     — SQL query only")
print("  (question) [vertical bar chart]               — SQL + chart")
print("  (question) [donut chart Pass blue Fail red]   — with colors")
print("  (question) [horizontal bar chart]\n")

while True:
    try:
        user_input = pt_prompt("You: ", completer=column_completer)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        break

    if user_input.lower() in ["exit", "quit"]:
        break
    if not user_input.strip():
        continue
    if not (user_input.startswith("(") and ")" in user_input):
        print("⚠️  Wrap your question in parentheses: (Your question here)")
        continue

    try:
        # Parse (question) and optional [chart hint]
        q_match = re.search(r'\(([^)]+)\)', user_input)
        c_match = re.search(r'\[([^\]]+)\]', user_input)

        question   = q_match.group(1) if q_match else user_input
        chart_hint = c_match.group(1) if c_match else None

        # ── STEP 1: Retrieve SQL context from FAISS ────────────────
        print("\n🔍 Retrieving SQL context...")
        sql_context = retrieve_context(sql_retriever, question, label="SQL")

        # ── STEP 2: LLM generates SQL ──────────────────────────────
        print("⏳ Generating SQL...")
        raw_sql = call_llm(sql_prompt.format_prompt(
            sql_context=sql_context,
            table=TABLE_NAME,
            columns=columns_str,
            question=question
        ))
        print(f"\n🤖 LLM SQL output:\n{raw_sql}")

        final_sql = extract_sql(raw_sql)
        print(f"\n📝 Cleaned SQL:\n{final_sql}")

        # ── STEP 3: Execute SQL ────────────────────────────────────
        db_result = execute_sql(final_sql)

        if isinstance(db_result, str):
            # SQL error
            print(f"\n❌ {db_result}")
            continue

        print(f"\n📊 Result ({len(db_result)} rows):")
        for row in db_result[:10]:   # preview first 10 rows
            print(f"   {dict(row)}")
        if len(db_result) > 10:
            print(f"   ... ({len(db_result) - 10} more rows)")

        # ── STEP 4: LLM generates Chart HTML ──────────────────────
        if chart_hint and db_result:
            print(f"⏳ Generating chart for: [{chart_hint}]...")
            sql_result_json = json.dumps(db_result, indent=2, default=str)

            raw_html = call_llm(chart_prompt.format_prompt(
                sql_result=sql_result_json,
                chart_request=chart_hint
            ))

            # Strip accidental markdown fences
            raw_html = re.sub(r"```html", "", raw_html, flags=re.IGNORECASE)
            raw_html = re.sub(r"```", "", raw_html).strip()

            # Save and open
            file_name = sanitize_filename(question) + ".html"
            file_path = os.path.join(HTML_FOLDER, file_name)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(raw_html)

            webbrowser.open(f"file://{os.path.abspath(file_path)}")
            print(f"\n✅ Chart saved & opened: {file_path}\n")

        elif chart_hint and not db_result:
            print("⚠️  No data returned — chart skipped.")

    except Exception as e:
        print(f"❌ Error: {e}")

print("\nGoodbye! 👋")