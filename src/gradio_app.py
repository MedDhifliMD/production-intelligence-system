import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib, numpy as np, pandas as pd
import os
import sys

# Add parent directory to sys.path to import from src
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
from src.preprocessing import clean_and_feature_engineer, aggregate_to_boards

# ─── 1. SETUP & DATA LOADING ───────────────────────────────────────────────
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(BASE_PATH, '../models')
DATA_PATH = os.path.join(BASE_PATH, '../final_dataset.csv')

print("🚀 Initializing system...")
try:
    xgb_model = joblib.load(os.path.join(MODELS_PATH, 'model_xgboost.pkl'))
    scaler    = joblib.load(os.path.join(MODELS_PATH, 'coordinate_scaler.pkl'))
except FileNotFoundError:
    print("⚠️ Models not found in /models directory. Please run src/train_models.py first.")
    xgb_model = None
    scaler = None

if os.path.exists(DATA_PATH):
    # Load sample for dashboard and indexing
    cols = [
        'Barcode', 'NPM_Date', 'Verif_Date', 'Componenet_Result', 'Feede_ID', 'Designator',
        'Pattern_Barcode', 'Coordinate_X', 'Coordinate_Y', 'DefectCode', 'Nozel_Name', 'Has_Verification'
    ]
    df_rag = pd.read_csv(DATA_PATH, usecols=cols)
    df_rag['NPM_Date'] = pd.to_datetime(df_rag['NPM_Date'], errors='coerce')
    
    # Pre-calculate dataset means and GLOBAL STATS
    mean_x = df_rag['Coordinate_X'].mean()
    mean_y = df_rag['Coordinate_Y'].mean()
    
    print("📊 Calculating global failure statistics...")
    failures = df_rag[df_rag['Componenet_Result'].astype(str).str.strip() != 'Pass']
    top_failed_comps = failures['Designator'].value_counts().head(5).to_dict()
    top_failed_feeders = failures['Feede_ID'].value_counts().head(5).to_dict()
    total_failure_rate = (len(failures) / len(df_rag)) * 100
    
    global_stats_str = (
        f"Dataset Overview:\n"
        f"- Total Records: {len(df_rag):,}\n"
        f"- Global Failure Rate: {total_failure_rate:.2f}%\n"
        f"- Top 5 Failed Components (Designators): {', '.join([f'{k} ({v} failures)' for k, v in top_failed_comps.items()])}\n"
        f"- Top 5 Failed Feeders: {', '.join([f'{k} ({v} failures)' for k, v in top_failed_feeders.items()])}\n"
    )
    print(f"✅ Loaded {len(df_rag):,} records.")
else:
    print("❌ ERROR: final_dataset.csv not found!")
    df_rag = pd.DataFrame()
    mean_x, mean_y = 200, 100
    global_stats_str = "No data available."

# ─── 2. PREDICTION FUNCTION ──────────────────────────────────────────────
def predict_board_failure_dynamic(feeder_list_str, designator_list_str, npm_hour, day_of_week, delay):
    if xgb_model is None or scaler is None:
        return "❌ Error: Models not trained. Please run training script first."
    
    try:
        # 1. Parse inputs
        feeders = [f.strip() for f in feeder_list_str.split(',') if f.strip()]
        designators = [d.strip() for d in designator_list_str.split(',') if d.strip()]
        
        if not feeders and not designators:
            return "⚠️ Please enter at least one Feeder ID or Designator."

        # 2. Sanitize scenario inputs
        npm_hour = float(np.clip(pd.to_numeric(npm_hour, errors='coerce'), 0, 23))
        day_of_week = int(np.clip(pd.to_numeric(day_of_week, errors='coerce'), 0, 6))
        delay = float(max(pd.to_numeric(delay, errors='coerce'), 0))
        
        # Look up coordinates in data
        subset = df_rag.copy()
        if feeders:
            subset = subset[subset['Feede_ID'].astype(str).isin(feeders)]
        if designators:
            subset = subset[subset['Designator'].astype(str).isin(designators)]
            
        if not subset.empty:
            # Build board-level features from matching historical boards to align with training logic.
            matched_board_ids = subset['Pattern_Barcode'].dropna().unique()
            board_rows = df_rag[df_rag['Pattern_Barcode'].isin(matched_board_ids)].copy()

            board_rows['Coordinate_X'] = pd.to_numeric(board_rows['Coordinate_X'], errors='coerce')
            board_rows['Coordinate_Y'] = pd.to_numeric(board_rows['Coordinate_Y'], errors='coerce')
            board_rows['Componenet_Result'] = board_rows['Componenet_Result'].astype(str).str.strip()

            board_clean = clean_and_feature_engineer(board_rows)
            board_features, _ = aggregate_to_boards(board_clean)

            if not board_features.empty:
                total_comps = int(max(1, round(board_features['Total_Components_Placed'].median())))
                unique_feeders = int(max(1, round(board_features['Unique_Feeders_Used'].median())))
            else:
                total_comps = len(designators) if designators else 120
                unique_feeders = len(set(feeders)) if feeders else 8

            avg_x = pd.to_numeric(subset['Coordinate_X'], errors='coerce').mean()
            avg_y = pd.to_numeric(subset['Coordinate_Y'], errors='coerce').mean()
            if pd.isna(avg_x) or pd.isna(avg_y):
                avg_x, avg_y = mean_x, mean_y
        else:
            total_comps = len(designators) if designators else 120
            unique_feeders = len(set(feeders)) if feeders else 8
            avg_x, avg_y = mean_x, mean_y

        # 3. Predict
        # Scale coordinates
        scaled_coords = scaler.transform([[avg_x, avg_y]])

        # Prepare input array (7 features)
        arr = np.array([[
            total_comps, unique_feeders,
            scaled_coords[0][0], scaled_coords[0][1],
            npm_hour, day_of_week, delay
        ]])

        pred  = xgb_model.predict(arr)[0]
        probs = xgb_model.predict_proba(arr)[0]
        conf  = max(probs) * 100
        fail_prob = probs[1] * 100

        result = f"### 📊 Prediction Result\n---"
        if pred == 1:
            result += f"\n🔴 **HIGH RISK**\n- Confidence: {conf:.1f}%\n- Action: Flag for manual inspection."
        else:
            result += f"\n🟢 **LOW RISK**\n- Confidence: {conf:.1f}%\n- Action: Clear for production."
            
        result += (
            f"\n\n**Derived Features Used:**"
            f"\n- Computed Components: {total_comps}"
            f"\n- Computed Unique Feeders: {unique_feeders}"
            f"\n- Avg Coords: ({avg_x:.2f}, {avg_y:.2f})"
            f"\n- Model Fail Probability: {fail_prob:.1f}%"
        )

        if not subset.empty:
            pass_rate = (subset['Componenet_Result'].astype(str).str.strip() == 'Pass').mean() * 100
            result += (
                f"\n\n**Historical Match Context:**"
                f"\n- Matching rows: {len(subset)}"
                f"\n- Pass rate in matches: {pass_rate:.1f}%"
                f"\n- Note: a component-level Pass does not guarantee board-level Pass."
            )
        
        return result
    except Exception as e:
        return f"❌ Prediction Error: {str(e)}"

# ─── 3. ANALYTICS DASHBOARD ──────────────────────────────────────────────
def build_live_dashboard():
    if df_rag.empty:
        return None
    try:
        DARK, PANEL = '#0f0f23', '#16213e'
        fail_df = df_rag[df_rag['Componenet_Result'].astype(str).str.strip() != 'Pass'].copy()

        fig = make_subplots(
            rows=3, cols=2,
            column_widths=[0.45, 0.55],
            row_heights=[0.38, 0.32, 0.30],
            specs=[
                [{"type": "domain"}, {"type": "xy"}],
                [{"type": "xy"},     {"type": "xy"}],
                [{"type": "xy", "colspan": 2}, None],
            ],
            subplot_titles=["🍩 Pass / Fail Split", "⏰ Defects by Hour", "🔧 Top Failed Feeders", "🔩 Top Failed Components", "🛑 Top Defect Codes"],
            vertical_spacing=0.1, horizontal_spacing=0.08
        )

        c = df_rag['Componenet_Result'].value_counts()
        fig.add_trace(go.Pie(labels=c.index, values=c.values, hole=0.5, marker=dict(colors=['#00d4aa','#ff4757','#a29bfe'])), row=1, col=1)

        h = fail_df['NPM_Date'].dt.hour.value_counts().sort_index()
        fig.add_trace(go.Bar(x=h.index, y=h.values, marker_color='#a29bfe'), row=1, col=2)

        tf = fail_df['Feede_ID'].value_counts().head(10).sort_values()
        fig.add_trace(go.Bar(x=tf.values, y=tf.index.astype(str), orientation='h', marker_color='#ff4757'), row=2, col=1)

        td = fail_df['Designator'].value_counts().head(10).sort_values()
        fig.add_trace(go.Bar(x=td.values, y=td.index.astype(str), orientation='h', marker_color='#fd79a8'), row=2, col=2)

        tc = fail_df['DefectCode'].value_counts().head(10).sort_values()
        fig.add_trace(go.Bar(x=tc.index.astype(str), y=tc.values, marker_color='#ffdd59'), row=3, col=1)

        fig.update_layout(height=1000, paper_bgcolor=DARK, plot_bgcolor=PANEL, font=dict(color='white'), showlegend=False)
        return fig
    except Exception as e:
        print(f"Dashboard Error: {e}")
        return None

# ─── 4. RAG ENGINE ───────────────────────────────────────────────────────
from rag_engine import RAGEngine
rag_engine = RAGEngine()
if not df_rag.empty:
    print("Indexing data for RAG...")
    # Provide global stats to the engine
    rag_engine.set_global_stats(global_stats_str)
    # Index a larger sample (stratified) for the live app
    rag_engine.index_data(df_rag, sample_size=10000)

# ─── 5. GRADIO UI ──────────────────────────────────────────────────────────
custom_css = ".gradio-container { background: #0f0f23 !important; color: white !important; } .tab-nav button.selected { background: #6c5ce7 !important; color: white !important; }"

with gr.Blocks(title="AI Factory Intelligence", css=custom_css) as demo:
    gr.Markdown("# 🏭 AI Production Intelligence\n---")

    with gr.Tab("🔮 Risk Prediction"):
        gr.Markdown("### Enter Feeder and Component details to predict board failure risk")
        with gr.Row():
            with gr.Column():
                f_in = gr.Textbox(label="Feeder IDs (comma separated)", placeholder="e.g. 5, 8, 12", value="5")
                d_in = gr.Textbox(label="Designator Names (comma separated)", placeholder="e.g. C121, R450", value="C121")
            with gr.Column():
                hr = gr.Slider(0, 23, value=14, label="Production Hour")
                dw = gr.Slider(0, 6, value=2, label="Day of Week (0=Mon)")
                dl = gr.Number(label="Delay to Verification (min)", value=60)
        btn = gr.Button("🚀 Run AI Assessment", variant="primary")
        out = gr.Markdown(label="Risk Assessment Report")
        btn.click(predict_board_failure_dynamic, inputs=[f_in, d_in, hr, dw, dl], outputs=out)

    with gr.Tab("💬 RAG Chat"):
        gr.Markdown("### Ask questions about production history and defects")
        q_in = gr.Textbox(label="Your Question", placeholder="What are the main causes of failure?")
        q_btn = gr.Button("🔍 Ask AI", variant="primary")
        q_out = gr.Textbox(label="AI Answer", lines=5)
        q_btn.click(lambda q: rag_engine.query(q)[0], inputs=q_in, outputs=q_out)

    with gr.Tab("📊 Analytics Dashboard"):
        dash_btn = gr.Button("📈 Refresh Analytics", variant="primary")
        plot = gr.Plot()
        dash_btn.click(build_live_dashboard, outputs=plot)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
