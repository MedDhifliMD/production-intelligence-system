import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib, numpy as np, pandas as pd
import os
import sys

# Add src to system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rag_engine import RAGEngine

# Configurable Paths
MODEL_DIR = 'models'
DATA_PATH = 'final_dataset.csv'

# Load Assets
print("Loading models and data for App...")
xgb_model = joblib.load(os.path.join(MODEL_DIR, 'model_xgboost.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'coordinate_scaler.pkl'))
df_rag = pd.read_csv(DATA_PATH) if os.path.exists(DATA_PATH) else pd.DataFrame()

# Initialize RAG
rag_engine = RAGEngine()
if not df_rag.empty:
    rag_engine.index_data(df_rag)

# ==========================
# Prediction Function
# ==========================
def predict_board_failure(feeder_ids_input, components_input):
    try:
        if df_rag.empty: return "❌ Dataset missing."
        
        feeder_ids = [f.strip() for f in feeder_ids_input.split(',') if f.strip()]
        components = [c.strip() for c in components_input.split(',') if c.strip()]

        if feeder_ids or components:
            mask = pd.Series([True] * len(df_rag), index=df_rag.index)
            if feeder_ids: mask &= df_rag['Feede_ID'].astype(str).isin(feeder_ids)
            if components: mask &= df_rag['Designator'].astype(str).isin(components)
            affected_barcodes = df_rag.loc[mask, 'Pattern_Barcode'].unique()
        else:
            affected_barcodes = np.array([])

        subset = df_rag[df_rag['Pattern_Barcode'].isin(affected_barcodes)].copy() if len(affected_barcodes) > 0 else df_rag.copy()

        # Feature Computation
        subset['Coordinate_X'] = pd.to_numeric(subset['Coordinate_X'], errors='coerce')
        subset['Coordinate_Y'] = pd.to_numeric(subset['Coordinate_Y'], errors='coerce')
        npm_dt = pd.to_datetime(subset['NPM_Date'], errors='coerce')
        subset['_hour'] = npm_dt.dt.hour
        subset['_dow']  = npm_dt.dt.dayofweek

        board_stats = subset.groupby('Pattern_Barcode').agg(
            Total_Components_Placed=('Designator', 'count'),
            Unique_Feeders_Used=('Feede_ID', 'nunique'),
            Avg_Coordinate_X=('Coordinate_X', 'mean'),
            Avg_Coordinate_Y=('Coordinate_Y', 'mean'),
            NPM_Hour=('_hour', 'first'),
            NPM_DayOfWeek=('_dow', 'first'),
            Delay_to_Verif_Mins=('_hour', 'count') # Proxy if verification date is missing
        ).median()

        def safe(v, default): return float(v) if not pd.isna(v) else default

        scaled = scaler.transform([[safe(board_stats['Avg_Coordinate_X'], 200), safe(board_stats['Avg_Coordinate_Y'], 100)]])
        arr = np.array([[
            int(safe(board_stats['Total_Components_Placed'], 120)),
            int(safe(board_stats['Unique_Feeders_Used'], 8)),
            scaled[0][0], scaled[0][1],
            safe(board_stats['NPM_Hour'], 12), safe(board_stats['NPM_DayOfWeek'], 2), 0.0
        ]])

        pred = xgb_model.predict(arr)[0]
        conf = max(xgb_model.predict_proba(arr)[0]) * 100
        return f"🔴 FAIL: {conf:.1f}%" if pred == 1 else f"🟢 PASS: {conf:.1f}%"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ==========================
# Dashboard Function
# ==========================
def build_live_dashboard():
    if df_rag.empty: return None
    failures = df_rag[df_rag['Componenet_Result'] != 'Pass'].copy()
    fig = make_subplots(rows=2, cols=2, subplot_titles=["Pass/Fail Split", "Defects by Hour", "Top Feeders", "Top Designators"])
    
    # Pie
    counts = df_rag['Componenet_Result'].value_counts()
    fig.add_trace(go.Pie(labels=counts.index, values=counts.values, hole=0.4), row=1, col=1)
    
    # Bar Hour
    failures['Hour'] = pd.to_datetime(failures['NPM_Date']).dt.hour
    hourly = failures.groupby('Hour').size()
    fig.add_trace(go.Bar(x=hourly.index, y=hourly.values), row=1, col=2)
    
    # Top Feeders
    top_f = failures['Feede_ID'].value_counts().head(10)
    fig.add_trace(go.Bar(x=top_f.values, y=top_f.index.astype(str), orientation='h'), row=2, col=1)
    
    # Top Designators
    top_d = failures['Designator'].value_counts().head(10)
    fig.add_trace(go.Bar(x=top_d.values, y=top_d.index.astype(str), orientation='h'), row=2, col=2)

    fig.update_layout(height=700, showlegend=False, template="plotly_dark")
    return fig

# ==========================
# UI Setup
# ==========================
with gr.Blocks(title="AI Production Intelligence") as demo:
    gr.Markdown("# 🏭 AI Production Intelligence Dashboard")
    with gr.Tab("🔮 Prediction"):
        f_in = gr.Textbox(label="Feeder IDs", placeholder="F01, F02")
        c_in = gr.Textbox(label="Components", placeholder="C121, R450")
        btn = gr.Button("Predict Risk")
        out = gr.Textbox(label="Result")
        btn.click(predict_board_failure, inputs=[f_in, c_in], outputs=out)
    
    with gr.Tab("💬 RAG Chat"):
        q_in = gr.Textbox(label="Question")
        q_btn = gr.Button("Ask AI")
        q_out = gr.Textbox(label="Answer", lines=5)
        q_btn.click(lambda q: rag_engine.query(q)[0], inputs=q_in, outputs=q_out)

    with gr.Tab("📊 Analytics"):
        dash_btn = gr.Button("Refresh Dashboard")
        plot = gr.Plot()
        dash_btn.click(build_live_dashboard, outputs=plot)

if __name__ == "__main__":
    demo.launch()
