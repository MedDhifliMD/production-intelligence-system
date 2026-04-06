# Technical Report — AI-Powered Production Intelligence System

**Author:** Capstone Student  
**Date:** April 2026  
**Version:** 1.0  

---

## Table of Contents
1. Introduction & Objectives
2. System Architecture Overview
3. Phase 1 — Data Engineering
4. Phase 2 — Predictive Modeling
5. Phase 3 — RAG & LLM Intelligence Layer
6. Phase 4 — Deployment
7. Results & Metrics Summary
8. Limitations & Known Issues
9. Next Steps

---

## 1. Introduction & Objectives

This project implements an end-to-end AI system for a PCB (Printed Circuit Board) electronics manufacturing facility. The system ingests raw factory data from two large MongoDB collections, trains predictive machine learning models, builds a natural language chatbot for production history queries, and deploys the complete system as an interactive web application.

### Primary Objectives
- **Predict** whether a circuit board will fail verification inspection *before* it arrives at the quality control station
- **Explain** which machine parameters (feeder, nozzle, coordinates, timing) are the root cause of manufacturing defects
- **Query** the production history using plain English natural language
- **Deploy** the system as a live, interactive prototype demonstrating all three capabilities simultaneously

### Datasets
| Dataset | Format | Size | Documents |
|---------|--------|------|-----------|
| NPM-VF (`Ai-project-data.Npm.json`) | MongoDB JSON Array | ~2.15 GB | Millions |
| Verification Station (`Verification-Station.json`) | MongoDB JSON Array | ~2.49 GB | Millions |

---

## 2. System Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                  Raw Data Layer (MongoDB JSON)           │
│         NPM-VF (~2.15 GB) + Verification (~2.49 GB)     │
└─────────────────────────┬───────────────────────────────┘
                          │ ijson Streaming Pipeline
┌─────────────────────────▼───────────────────────────────┐
│              Data Engineering (Phase 1)                  │
│   JSON_to_CSV_Converter.py → final_dataset.csv (15 cols) │
└───────────────────────┬─────────────────┬───────────────┘
                        │                 │
          ┌─────────────▼──────┐   ┌──────▼──────────────┐
          │   ML Pipeline      │   │   RAG Pipeline       │
          │   (Phase 2)        │   │   (Phase 3)          │
          │                    │   │                      │
          │  Preprocessing     │   │  ChromaDB Embedding  │
          │  Feature Eng.      │   │  Sentence Transformer│
          │  XGBoost 0.73 AUC  │   │  FLAN-T5 LLM        │
          │  SHAP Explainability│  │  Precision@5: 1.00   │
          └─────────────┬──────┘   └──────┬───────────────┘
                        │                 │
          ┌─────────────▼─────────────────▼───────────────┐
          │             Gradio Web App (Phase 4)           │
          │   Tab 1: Live Prediction  │  Tab 2: RAG Chat  │
          │   Tab 3: Dashboard        │                    │
          └────────────────────────────────────────────────┘
```

### Technology Stack

| Component | Technology | Justification |
|-----------|------------|---------------|
| JSON Streaming | `ijson` | Handles 5 GB files without crashing RAM |
| Data Manipulation | `pandas` | Industry standard for tabular data |
| ML Models | `xgboost`, `sklearn` | Best-in-class for tabular data |
| Explainability | `shap` | TreeExplainer natively integrates with XGBoost |
| Vector Database | `chromadb` | Lightweight, runs locally without a server |
| Sentence Embedding | `sentence-transformers` | Free, high-quality semantic vectors |
| LLM | `google/flan-t5-base` | Open-source, no API key required |
| Deployment | `gradio` | Standalone UI development |
| Model Persistence | `joblib` | Industry standard `.pkl` serialization |

---

## 3. Phase 1 — Data Engineering

### 3.1 The Core Challenge: Memory-Safe JSON Ingestion
The primary engineering challenge was loading 5 GB of deeply nested JSON into a flat tabular structure without crashing the computer's RAM. A standard `json.load()` call would require loading both files simultaneously into memory, consuming an estimated 10–15 GB of RAM.

**Decision:** Use `ijson` (Incremental JSON Parser) to stream the files as a rolling iterator. Each document is parsed, flattened, and immediately written to a CSV row before the next document is loaded.

### 3.2 Data Schema Design
The output `final_dataset.csv` was deliberately designed to be a flat, denormalized table. Each row represents **one component** placed on a circuit board.

---

## 4. Phase 2 — Predictive Modeling

### 4.1 Prediction Target Definition
**Prediction Question:** *"Given the physical and temporal characteristics of a board's manufacturing process, will this board fail verification inspection?"*

**Target (`y`):** `Board_Failed = 1` if any component on the board received a non-Pseudofehler defect code; `Board_Failed = 0` otherwise.

### 4.2 Feature Engineering
Temporal features (Hour, Day, Delay) were found to be the most significant predictors, indicating that defect patterns are often correlated with shift changes or machine warm-up periods.

---

## 5. Phase 3 — RAG & LLM Intelligence Layer

### 5.1 Architecture
The RAG (Retrieval-Augmented Generation) pipeline allows context-aware querying of production history. ChromaDB stores vectorized representations of machine logs, and FLAN-T5 synthesizes clear answers.

---

## 6. Phase 4 — Deployment

The system is deployed as a modular application in `src/gradio_app.py`, providing an integrated interface for risk prediction, data exploration, and natural language analytics.

---

## 7. Results & Metrics Summary
- **Ingestion:** Successfully flattened 5GB of JSON into a 400MB CSV.
- **ML Performance:** XGBoost achieved an AUC-ROC of 0.73.
- **Explainability:** Identifed shift-end fatigue as a key risk factor.
- **RAG:** 100% precision on benchmark retrieval tests.

---

*End of Technical Report — Page 10 of 10*  
*Appendix: Modular source code available in the accompanying repository*
