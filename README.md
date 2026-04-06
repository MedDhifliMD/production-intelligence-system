# 🏭 AI-Powered Production Intelligence System

An end-to-end Machine Learning + RAG pipeline for predicting and analyzing manufacturing defects in PCB (Printed Circuit Board) production lines using real-world factory data from NPM-VF and Verification Station JSON databases.

---

## 📁 Project Structure

```
production-intelligence-system/
│
├── notebooks/                  # Interactive Step-by-Step Guides
│   ├── 01_EDA_Preprocessing.ipynb  # Data cleaning & Visualizations
│   ├── 02_Model_Development.ipynb  # XGBoost training & SHAP analysis
│   └── 03_RAG_Prototype.ipynb     # Natural Language Querying logic
│
├── src/                        # Modular Python Modules
│   ├── preprocessing.py        # Core feature engineering logic
│   ├── train_models.py         # Automated model training pipeline
│   ├── rag_engine.py           # Vector storage & LLM query logic
│   └── gradio_app.py           # Standalone Desktop/Web Application
│
├── models/                     # Saved Artifacts (.pkl)
├── JSON_to_CSV_Converter.py    # Raw MongoDB ingestion script
├── final_dataset.csv           # Generated master dataset (400MB+)
├── README.md                   # Project documentation
└── technical_report.md         # Comprehensive architectural report
```

---

## 🗃️ Dataset Schema (`final_dataset.csv`)

| Column | Source | Description |
|--------|--------|-------------|
| `Barcode` | NPM | Unique board identifier |
| `NPM_Date` | NPM | Timestamp of PCB manufacturing |
| `Verif_Date` | Verification | Timestamp of quality inspection |
| `Pattern_Barcode` | NPM | Sub-board pattern identifier (Join Key) |
| `Designator` | NPM | Component reference (e.g., C121, L850) |
| `Componenet_Result` | Both | `Pass` or `Fail` (Verification overrides NPM) |
| `Feede_ID` | NPM | Feeder machine that placed the component |
| `Nozel_Name` | NPM | Nozzle used during placement |
| `Coordinate_X` | NPM | X-axis position of component (mm) |
| `Coordinate_Y` | NPM | Y-axis position of component (mm) |
| `Has_Verification` | Derived | `True` if board was inspected |
| `DefectCode` | Verification | Type of defect detected |

---

## 🚀 How to Run

### 1. Environments & Prerequisites
Install required libraries:
```bash
pip install ijson pandas scikit-learn xgboost shap chromadb sentence-transformers gradio joblib transformers
```

### 2. Full Ingestion (Raw JSON to CSV)
If `final_dataset.csv` is missing, generate it from the raw JSON files:
```bash
python JSON_to_CSV_Converter.py
```

### 3. Automated Training
To train the models and save artifacts to the `models/` directory:
```bash
python src/train_models.py
```

### 4. Interactive Application
Launch the Gradio dashboard and RAG chatbot:
```bash
python src/gradio_app.py
```

---

## 🔬 Phase Overview

### Phase 1 — Data Engineering
- **Streaming Parser:** `ijson` handles 5 GB of nested JSON without memory overflow.
- **Aggregation:** Component-level data is rolled up into board-level features.

### Phase 2 — Predictive Modeling
- **Champion Model:** XGBoost (AUC 0.73).
- **Interpretability:** SHAP values reveal temporal defect clusters.

### Phase 3 — RAG Engine
- **Vector Search:** ChromaDB + `all-MiniLM-L6-v2`.
- **LLM Context:** Google FLAN-T5 generates human-readable defect reports.

### Phase 4 — Deployment
- **UI:** Gradio implementation with Risk Prediction, RAG Chat, and Analytics.
