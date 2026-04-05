import pandas as pd
from sqlalchemy import create_engine, text
import psycopg2

# -------------------------
# 1️⃣ CONFIGURATION
# -------------------------
DATA_PATH = r"C:\Users\Issa\source\repos\mygit\Glory_AI\dataset\Predictive-AI\final_dataset.csv"
DB_USER = "postgres"
DB_PASS = "****"
DB_HOST = "localhost"
DB_PORT = "*****"
DB_NAME = "****"
TABLE_NAME = "****"

# -------------------------
# 2️⃣ READ CSV
# -------------------------
print("📥 Reading CSV file...")
df = pd.read_csv(DATA_PATH)
print(f"Columns found: {df.columns.tolist()}")
print(f"First 5 rows:\n{df.head()}")

# -------------------------
# 3️⃣ CONNECT TO POSTGRES (default database to create new one)
# -------------------------
print(" Connecting to PostgreSQL to create database if needed...")
engine_master = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/postgres")
with engine_master.connect() as conn:
    conn.execute(text(f"COMMIT"))  # commit any pending transaction
    conn.execute(text(f"CREATE DATABASE {DB_NAME}"))
    print(f" Database '{DB_NAME}' created or already exists.")

# -------------------------
# 4️⃣ CONNECT TO TARGET DATABASE
# -------------------------
engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# -------------------------
# 5️⃣ CREATE TABLE (IF NOT EXISTS)
# -------------------------
with engine.connect() as conn:
    columns_sql = ", ".join([f'"{col}" TEXT' for col in df.columns])
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        {columns_sql}
    );
    """
    conn.execute(text(create_table_sql))
    print(f" Table '{TABLE_NAME}' is ready.")

# -------------------------
# 6️⃣ INSERT DATA
# -------------------------
print("💾 Inserting CSV data into PostgreSQL...")
df.to_sql(
    TABLE_NAME,
    engine,
    if_exists="replace",  # replace table with fresh data
    index=False
)
print(f" All CSV data inserted into '{TABLE_NAME}' successfully!")