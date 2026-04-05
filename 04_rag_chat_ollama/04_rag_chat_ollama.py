#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import psycopg2
import re
from psycopg2.extras import RealDictCursor
from langchain_ollama import ChatOllama
from langchain_classic.chains import RetrievalQA 
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.completion import WordCompleter

# -------------------------
# CONFIG
# -------------------------
VECTORSTORE_FILE = "rag_vectorstore.faiss"

DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "*****"
DB_USER = "postgres"
DB_PASS = "*****"
TABLE_NAME = "*****" 

COLUMNS_RAW = [
    "Barcode", "NPM_Date", "Verif_Date", "Pattern_Barcode",
    "Pattern_Index", "Designator", "Componenet_Result", "Feede_ID",
    "Nozel_Name", "Coordinate_X", "Coordinate_Y", "Rotation",
    "Has_Verification", "Has_Component_Verification", "DefectCode"
]

COLUMNS_QUOTED = [f'"{col}"' for col in COLUMNS_RAW]
column_completer = WordCompleter(COLUMNS_QUOTED, ignore_case=True)

# -------------------------
# 1️⃣ Load vectorstore
# -------------------------
print(" Loading vectorstore...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    VECTORSTORE_FILE, embeddings, allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
print(" Vectorstore loaded.")

# -------------------------
# 2️⃣ Load LLM
# -------------------------
llm = ChatOllama(model="gemma2:2b", temperature=0.1)
#llm = ChatOllama(model="qwen3:14b", temperature=0.05)

print(" LLM ready.")

# -------------------------
# 3️⃣ Prompt template (Hardened for PostgreSQL)
# -------------------------
columns_str = ", ".join(COLUMNS_QUOTED)

prompt_template = f"""
You are a PostgreSQL expert. Return ONLY valid SELECT SQL.

### MANDATORY RULES:
1. Table name is strictly "{TABLE_NAME}".
2. Wrap ALL column names in double quotes: "Barcode", "Componenet_Result".
3. Use LIMIT only if needed. Let the SQL reflect the user's intent.
4. If you SELECT a column AND use COUNT(*), you MUST include: GROUP BY "ColumnName".
5. For "Fail" status, always use: WHERE "Componenet_Result" = 'Fail'.

Available columns: {columns_str}

Context:
{{context}}

Question:
{{question}}

Answer:
"""

sql_prompt_template = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# -------------------------
# 4️⃣ Create RetrievalQA chain
# -------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": sql_prompt_template}
)

# -------------------------
# 5️⃣ THE SAFETY NET: Auto-Quote, Table Fixer, Syntax & Group-By Fixer
# -------------------------
def clean_and_fix_sql(sql, columns_raw, real_table):
    """
    Fix SQL hallucinations while keeping LLM LIMIT intact.
    Removes explanations and text that is not SQL.
    """
    # 1️⃣ Strip markdown code fences
    fixed_sql = sql.strip().replace("```sql", "").replace("```", "")

    # 2️⃣ Keep only the part starting from SELECT
    select_match = re.search(r"(SELECT\s.*?)(;|\Z)", fixed_sql, flags=re.IGNORECASE | re.DOTALL)
    if select_match:
        fixed_sql = select_match.group(1).strip()
    else:
        # fallback if no SELECT found, leave original
        fixed_sql = fixed_sql.splitlines()[0]

    # 3️⃣ Fix table name hallucinations
    hallucinations = ["your_table", "table_name", "ai_table"]
    for hallu in hallucinations:
        fixed_sql = re.sub(rf'\b{hallu}\b', real_table, fixed_sql, flags=re.IGNORECASE)

    # 4️⃣ Force double quotes on column names
    for col in columns_raw:
        pattern = rf'(?<!"){re.escape(col)}(?!")'
        fixed_sql = re.sub(pattern, f'"{col}"', fixed_sql)

    # 5️⃣ Multi-Column GROUP BY Fixer
    agg_funcs = ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN("]
    if any(f in fixed_sql.upper() for f in agg_funcs) and "GROUP BY" not in fixed_sql.upper():
        select_cols = re.search(r'SELECT\s+(.*?)\s+FROM', fixed_sql, re.IGNORECASE | re.DOTALL)
        if select_cols:
            cols = re.findall(r'("(?:[^"]+)")', select_cols.group(1))
            if cols:
                group_clause = f"GROUP BY {', '.join(cols)}"
                # Remove any existing ORDER BY/LIMIT temporarily
                order_limit_match = re.search(r'(ORDER BY.*|LIMIT.*)', fixed_sql, re.IGNORECASE)
                tail = ""
                if order_limit_match:
                    tail = " " + order_limit_match.group(1)
                    fixed_sql = fixed_sql[:order_limit_match.start()]
                fixed_sql = fixed_sql.strip() + " " + group_clause + tail

    # 6️⃣ Auto-append LIMIT 1 for top/min queries if missing
    if re.search(r'ORDER BY COUNT\(\*\)|ORDER BY COUNT\("Componenet_Result"\)', fixed_sql, re.IGNORECASE):
        if not re.search(r'LIMIT \d+', fixed_sql, re.IGNORECASE):
            fixed_sql += " LIMIT 1"

    return fixed_sql

# -------------------------
# 6️⃣ PostgreSQL helper
# -------------------------
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
        return result
    except Exception as e:
        return f" SQL Error: {e}"

# -------------------------
# 7️⃣ Chat loop
# -------------------------
print("\n💬 RAG SQL chatbot ready! Type 'exit' to quit.\n")

while True:
    try:
        user_input = pt_prompt("You: ", completer=column_completer)
    except KeyboardInterrupt:
        break

    if user_input.lower() in ["exit", "quit"]:
        break
    if not user_input.strip():
        continue

    try:
        # Run the chain
        response = qa_chain.invoke(user_input)
        raw_sql = response["result"]
        
        #  APPLY THE COMPLETE SAFETY NET (without overriding LLM LIMIT)
        final_sql = clean_and_fix_sql(raw_sql, COLUMNS_RAW, TABLE_NAME)
        
        print(f"\nGenerated SQL (Processed):\n{final_sql}")
        
        # Run fixed query
        db_result = execute_sql(final_sql)
        print(f"\nResult:\n{db_result}\n")

    except Exception as e:
        print(f" Error: {e}")

print(" Goodbye!")