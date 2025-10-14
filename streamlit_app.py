import json
from pathlib import Path
import streamlit as st
import time
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="🌊 Live Federated Learning Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS Styling ---
st.markdown(
    """
<style>
.main-header {
    font-size: 2.2rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 0.25rem;
}
.sub-header {
    text-align: center;
    color: #666;
    margin-top: 0;
    margin-bottom: 1.25rem;
}
.status-running { color: #28a745; font-weight: bold; }
.status-waiting { color: #ffc107; font-weight: bold; }
.metric-card {
    background-color: #f0f2f6;
    padding: 0.75rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
}
</style>
""",
    unsafe_allow_html=True,
)

# --- Page Headers ---
st.markdown('<h1 class="main-header">🌊 Live Federated Learning Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-time Privacy-Preserving Flood Detection AI</p>', unsafe_allow_html=True)

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Dashboard Controls")
auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh (rerun)", value=True)
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 1, 10, 2)

# Monitor state (start/stop)
if "monitor" not in st.session_state:
    st.session_state.monitor = auto_refresh

if st.sidebar.button("▶️ Start Monitor"):
    st.session_state.monitor = True
if st.sidebar.button("⏸️ Stop Monitor"):
    st.session_state.monitor = False

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 System Status")

# --- Paths & Placeholders ---
metrics_path = Path("metrics.json")
status_placeholder = st.sidebar.empty()
metrics_placeholder = st.sidebar.empty()
countdown_placeholder = st.sidebar.empty()
main_container = st.container()


# --- Helper Functions ---
def load_metrics():
    """Safely load metrics.json"""
    if not metrics_path.exists():
        return {}
    try:
        with metrics_path.open() as f:
            raw = f.read().strip()
            if not raw:
                return {}
            data = json.loads(raw)
        if "accuracies" not in data or not isinstance(data["accuracies"], list):
            return {}
        return data
    except Exception as e:
        status_placeholder.error(f"Error reading metrics.json: {e}")
        return {}


def format_pct(x):
    """Format floats as percentages"""
    try:
        return f"{x:.1%}"
    except Exception:
        return str(x)


def display_dashboard(data):
    """Render the dashboard based on metrics data"""
    accuracies = data.get("accuracies", [])
    last_updated = data.get("last_updated", None)

    # --- No data yet ---
    if not accuracies:
        status_placeholder.markdown('<p class="status-waiting">⏳ Waiting for federated learning to start...</p>', unsafe_allow_html=True)
        with main_container:
            st.info("🚀 **Ready to start!** Run `python simple_demo.py` to begin the federated learning demo.")
            st.markdown("### 🎯 What You'll See:")
            st.markdown(
                """
- 📱 **3 Clients** training locally on private flood data  
- 🌐 **Server** aggregating model weights (not images!)  
- 📈 **Live accuracy** updates as learning progresses  
- 🔒 **Privacy preserved** — only model weights shared  
                """
            )
        with metrics_placeholder:
            st.metric("Latest Accuracy", "—")
            st.metric("Rounds Completed", "0")
        return False, accuracies

    # --- Compute state ---
    rounds_completed = max(len(accuracies) - 1, 0)
    current_acc = accuracies[-1]
    best_acc = max(accuracies)
    training_complete = data.get("training_complete", False)

    # --- Sidebar Status ---
    if training_complete:
        status_placeholder.markdown('<p style="color: #28a745; font-weight: bold;">✅ Training Complete!</p>', unsafe_allow_html=True)
    else:
        status_placeholder.markdown('<p class="status-running">🟢 Federated Learning Active</p>', unsafe_allow_html=True)

    # --- Sidebar Metrics ---
    with metrics_placeholder:
        colA, colB = st.columns(2)
        with colA:
            st.metric("Current Round", rounds_completed)
            st.metric("Latest Accuracy", format_pct(current_acc))
        with colB:
            if len(accuracies) > 1:
                improvement = current_acc - accuracies[0]
                st.metric("Total Improvement", format_pct(improvement))
            else:
                st.metric("Total Improvement", "—")

        if last_updated:
            try:
                dt = datetime.fromisoformat(last_updated)
                st.caption(f"Last update: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
            except Exception:
                st.caption(f"Last update: {last_updated}")

    # --- Main Content Layout ---
    with main_container:
        col1, col2, col3 = st.columns([2, 1, 1])

        # Chart & Rounds
        with col1:
            st.subheader("📈 Federated Learning Progress")
            df = pd.DataFrame({"Round": list(range(len(accuracies))), "Global Test Accuracy": accuracies})
            df = df.set_index("Round")
            st.line_chart(df, height=420)

            st.subheader("🔍 Round-by-Round Details")
            for i, acc in enumerate(accuracies):
                if i == 0:
                    st.write(f"**Initial (global eval):** {format_pct(acc)}")
                else:
                    prev_acc = accuracies[i - 1]
                    change = acc - prev_acc
                    emoji = "📈" if change > 0 else "📊" if change == 0 else "📉"
                    st.write(f"**Round {i}:** {format_pct(acc)} {emoji} ({change:+.1%})")

        # Key Metrics
        with col2:
            st.subheader("🎯 Key Metrics")
            st.metric("Current Accuracy", format_pct(current_acc), f"{current_acc:.4f}")
            st.metric("Best Accuracy", format_pct(best_acc))
            st.metric("Rounds Completed", rounds_completed)

            if training_complete:
                st.success("✅ Training Complete!")
            else:
                st.info("🔄 Training in progress...")

        # Privacy / Export
        with col3:
            st.subheader("🔒 Privacy Stats")
            st.metric("Model Parameters", "1,054,050")
            st.metric("Clients", "3")
            st.metric("Data Shared", "0 images")
            st.write("**Only model weights transmitted!**")

            st.subheader("💾 Export Data")
            st.download_button(
                "📊 Download Metrics",
                data=json.dumps(data, indent=2),
                file_name="federated_learning_results.json",
                mime="application/json",
            )

            if st.checkbox("Show Raw Data"):
                st.json(data)

    return training_complete, accuracies


# --- Main Flow ---
data = load_metrics()
is_complete, accuracies = display_dashboard(data)

# --- Auto-refresh logic ---
if st.session_state.monitor and auto_refresh:
    if is_complete:
        countdown_placeholder.success("✅ Training complete. Auto-refresh paused.")
        st.session_state.monitor = False
    else:
        # Refresh countdown
        for i in range(refresh_rate, 0, -1):
            countdown_placeholder.metric("Next Refresh", f"{i}s")
            time.sleep(1)
        countdown_placeholder.empty()
        st.rerun()
else:
    # Static mode
    if not metrics_path.exists():
        st.info("metrics.json not found. Run the federated learning demo (python simple_demo.py) first.")
    else:
        if accuracies:
            st.caption(f"Snapshot: {len(accuracies)} datapoints — toggle Start Monitor for live updates.")
