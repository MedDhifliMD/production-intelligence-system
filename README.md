# AI Production Intelligence System

An end-to-end AI-powered system for manufacturing defect analysis, featuring predictive modeling, natural language querying (RAG), and interactive dashboards.

## 📁 Project Structure

```text
├── app.py                # Main Gradio application (Entry point)
├── src/
│   ├── preprocessing.py  # Data cleaning and feature engineering
│   ├── train_models.py   # Model training and artifact generation
│   └── rag_engine.py     # ChromaDB and LLM (RAG) logic
├── models/               # Saved model artifacts (.pkl)
├── final_dataset.csv     # (NOT IN REPO) Your manufacturing data
├── requirements.txt      # Dependency list
└── .gitignore            # Files excluded from version control
```

## 🚀 Quick Start

### 1. Setup Environment
Ensure you have Python 3.9+ installed. Clone this repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
Place your `final_dataset.csv` in the root directory of the project.

### 3. Train Models (Optional)
If you haven't saved your models yet, or want to retrain them on your local data:
```bash
python src/train_models.py
```
This will generate `.pkl` files in the `models/` directory.

### 4. Run the Application
Launch the interactive web interface:
```bash
python app.py
```
Wait for the RAG engine to index the data. Once ready, a local URL (and a public shareable URL if on Colab) will be displayed.

## 🛠 Features
- **🔮 Prediction**: Predict board pass/fail risk based on Feeder IDs or Component Designators.
- **💬 RAG Chat**: Ask questions about production history in plain English (uses FLAN-T5).
- **📊 Analytics**: View live defect distributions and hotspot heatmaps.

---
*Note: This project was originally developed in Google Colab and has been modularized for local and GitHub deployment.*
