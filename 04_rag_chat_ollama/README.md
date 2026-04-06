# 🏭 Production Intelligence System — RAG SQL Chatbot

A local AI chatbot that understands your NPMVF machine data, converts natural language questions into PostgreSQL queries, executes them, and optionally generates Chart.js visualizations — all running **100% offline** using Ollama.

---

## 📁 Project Structure

```
04_rag_chat_ollama/
│
├── load_csv_to_postgres.py       # Step 1 — Load CSV data into PostgreSQL
├── 02_extract_context.py         # Step 2 — Extract PostgreSQL docs + schema → rag_context.pkl
├── 03_generate_embeddings.py     # Step 3 — Build FAISS index from rag_context.pkl
├── build_faiss_sql.py            # Step 3b — Improved FAISS builder (use this instead)
│
├── 04_rag_chat_ollama.py         # Original chatbot (basic version)
├── chatbotV2.py                  # ✅ Improved chatbot with chart support (use this)
│
├── faiss_sql/                    # Generated FAISS index (created by build_faiss_sql.py)
├── rag_vectorstore.faiss/        # Generated FAISS index (created by 03_generate_embeddings.py)
├── rag_context.pkl               # Extracted context (created by 02_extract_context.py)
│
├── postgresql-16-A4.pdf          # PostgreSQL 16 documentation (full)
├── postgresql-16-A4-1-568.pdf    # PostgreSQL 16 documentation (pages 1-568)
└── requirements.txt              # Python dependencies
```

---

## ⚙️ Prerequisites

### 1. PostgreSQL 16

You need a running PostgreSQL instance to store and query your data.

- Download: https://www.postgresql.org/download/
- Default connection used in scripts:

```python
DB_HOST    = "localhost"
DB_PORT    = 5432
DB_NAME    = "*****"       # ← your database name
DB_USER    = "postgres"
DB_PASS    = "*****"       # ← your password
TABLE_NAME = "*****"       # ← your table name
```

> ⚠️ Update these values in every script before running.

---

### 2. Ollama (Local LLM)

Ollama runs LLM models locally on your machine. No internet needed after installation.

- Download: https://ollama.com/download

After installing, verify it works:
```cmd
ollama list
```

Pull a model (choose one):
```cmd
# Lightweight, fast (recommended for testing)
ollama pull gemma2:2b

# More accurate, slower
ollama pull qwen3:14b

# Best balance
ollama pull qwen3:8b
```

> The scripts default to `gemma2:2b`. To use another model, change this line in `chatbotV2.py`:
> ```python
> llm = ChatOllama(model="gemma2:2b", ...)
> ```

---

### 3. Python Dependencies

```cmd
pip install -r requirements.txt
```

Or manually:
```cmd
pip install psycopg2 sqlalchemy pdfplumber faiss-cpu langchain langchain-community
pip install langchain-core langchain-text-splitters langchain-huggingface langchain-ollama
pip install sentence-transformers prompt_toolkit
```

---

## 🚀 Setup & Run Order

Run the scripts **in this exact order** the first time:

### Step 1 — Load your CSV data into PostgreSQL

```cmd
python load_csv_to_postgres.py
```

This reads your CSV file and creates the table in PostgreSQL.

---

### Step 2 — Extract context (optional — for basic chatbot only)

```cmd
python 02_extract_context.py
```

Extracts text from `postgresql-16-A4.pdf` + table schema and saves it to `rag_context.pkl`.

---

### Step 3 — Generate embeddings

**Option A — Basic (uses rag_context.pkl):**
```cmd
python 03_generate_embeddings.py
```
Generates `rag_vectorstore.faiss/`

**Option B — Improved ✅ (recommended):**
```cmd
python build_faiss_sql.py
```
Generates `faiss_sql/` with richer context including:
- Live schema from your PostgreSQL table
- Table domain knowledge and column explanations
- 12 hand-crafted SQL Q&A examples
- Full PostgreSQL 16 PDF documentation

> After running once, the FAISS index is cached. No need to rebuild unless your schema changes.

---

### Step 4 — Start the chatbot

**Basic version:**
```cmd
python 04_rag_chat_ollama.py
```

**Improved version with chart support ✅ (recommended):**
```cmd
python chatbotV2.py
```

---

## 💬 How to Use the Chatbot

### Input Format

```
(your question here)
```
For SQL query only.

```
(your question here) [chart type]
```
For SQL query + auto-generated Chart.js visualization.

> ⚠️ The question **must** be inside `( )`. Chart request goes inside `[ ]`.

---

### Autocomplete Tips

- Press `Tab` after typing `"` to see available column names
- Use **double quotes** for column names: `"Barcode"`, `"Componenet_Result"`
- Use **single quotes** for values: `'Pass'`, `'Fail'`

---

### Available Columns

| Column | Type | Description |
|---|---|---|
| `"Barcode"` | bigint | Unique ID of the PCB card |
| `"NPM_Date"` | text (ISO-8601) | Date/time card was produced by NPMVF |
| `"Verif_Date"` | text (ISO-8601) | Date/time card was tested at verification |
| `"Pattern_Barcode"` | text | Assembly pattern name used for the card |
| `"Pattern_Index"` | bigint | Pattern index number |
| `"Designator"` | text | Component placed on the card (e.g. C120, L850) |
| `"Componenet_Result"` | text | Placement result: `'Pass'` or `'Fail'` |
| `"Feede_ID"` | text | Feeder ID that supplied the component |
| `"Nozel_Name"` | text | Nozzle that placed the component |
| `"Coordinate_X"` | float | X position on card (mm) |
| `"Coordinate_Y"` | float | Y position on card (mm) |
| `"Rotation"` | bigint | Component rotation angle |
| `"Has_Verification"` | boolean | Was the card sent to verification? |
| `"Has_Component_Verification"` | boolean | Was this component verified? |
| `"DefectCode"` | text | Defect reason if Fail (e.g. `Pseudofehler`) |

---

## 📊 Example Questions

### SQL only
```
(How many unique "Barcode" were produced in total?)
```
```
(What is the average number of "Designator" per "Barcode"?)
```
```
(List all "DefectCode" and how many times each appears?)
```

### SQL + Chart
```
(Count total 'Pass' and 'Fail' in "Componenet_Result") [donut chart Pass in blue Fail in red]
```
```
(How many unique "Barcode" were produced each day between '2024-10-01' and '2024-10-31'?) [vertical bar chart showing each day on X axis]
```
```
(Which top 5 "Feede_ID" have the most 'Pass' results?) [horizontal bar chart]
```
```
(Which TOP 5 "Pattern_Barcode" has the highest number of 'Fail' results?) [vertical bar chart each "Pattern_Barcode" in bar]
```
```
(What is the fail rate percentage per "Nozel_Name"?) [vertical bar chart]
```

---

### Real Example Output

**Question:**
```
(Which TOP 5 "Pattern_Barcode" has the highest number of 'Fail' results?) [vertical bar chart each "Pattern_Barcode" in bar]
```

**Generated SQL:**
```sql
SELECT "Pattern_Barcode", COUNT(*) AS fail_count
FROM ai_data
WHERE "Componenet_Result" = 'Fail'
GROUP BY "Pattern_Barcode"
ORDER BY fail_count DESC
LIMIT 5
```

**Result:**
```
{'Pattern_Barcode': 'CE65AK9P5A08002434482412', 'fail_count': 4}
{'Pattern_Barcode': 'CE65AK9P5A08002500794284', 'fail_count': 4}
{'Pattern_Barcode': 'CE65AK9P5A08002429539582', 'fail_count': 4}
{'Pattern_Barcode': 'CE65AK9P5A08002429539583', 'fail_count': 4}
{'Pattern_Barcode': 'CE65AK9P5A08002500794285', 'fail_count': 4}
```

Chart saved automatically and opened in browser ✅

<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Chart</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
  <h2>Bar Chart</h2>
  <canvas id="myChart" width="500" height="300"></canvas>
  <script>
    const data = {
      labels: [
        'CE65AK9P5A08002434482412',
        'CE65AK9P5A08002500794284',
        'CE65AK9P5A08002429539582',
        'CE65AK9P5A08002429539583',
        'CE65AK9P5A08002500794285'
      ],
      datasets: [{
        label: 'Fail Count',
        data: [4, 4, 4, 4, 4],
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        borderColor: 'rgba(255, 99, 132, 1)'
      }]
    };

    const config = {
      type: 'bar',
      data: data,
      options: {
        plugins: {
          title: {
            display: true,
            text: 'Fail Count'
          }
        }
      }
    };

    new Chart('myChart', config);
  </script>
</body>
</html>


---

## 🔄 Version History

| Version | File | Notes |
|---|---|---|

| v1 | `04_rag_chat_ollama.py` + `03_generate_embeddings.py` | Basic RAG chatbot, SQL only | use question without ()

| v2 | `chatbotV2.py` + `build_faiss_sql.py` | Improved FAISS, better prompt, chart support | you need to make the question between ()


