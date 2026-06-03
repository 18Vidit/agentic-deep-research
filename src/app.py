import streamlit as st
import io
import sys
from contextlib import redirect_stdout
import google.generativeai as genai # So we can initialize custom models on the fly

st.set_page_config(page_title="Agentic Deep Research Console", layout="wide")

# Hide the default Streamlit menu and deploy button for a cleaner look
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

with st.status("Booting Research Engine...", expanded=True) as status:
    st.write("Connecting to persistent vector store...")
    # We import 'collection' but we no longer need the default 'model' from phase2
    from phase2_agent import ResearchAgent, collection 
    st.write("Index loaded.")
    status.update(label="System Ready", state="complete", expanded=False)

st.title("Agentic Deep Research Console")
st.markdown("Test the agentic RAG pipeline against the 2024-2026 arXiv corpus.")

# SIDEBAR
st.sidebar.header("Architecture Flags")
use_planner = st.sidebar.checkbox("Planner (Query Rewriting)", value=True)
use_reflector = st.sidebar.checkbox("Reflector (Self-Critique)", value=True)
use_verifier = st.sidebar.checkbox("Citation Verifier", value=True)

st.sidebar.divider()

# DYNAMIC MODEL SELECTOR WITH USER-FRIENDLY NAMES AND DESCRIPTIONS
st.sidebar.subheader("Engine Configuration")
selected_model_display = st.sidebar.selectbox(
    "LLM Backbone:",
    [
        "gemini-flash-lite-latest (Recommended: High Throughput / Stable)",
        "gemini-3.5-flash (Balanced: Speed & Intelligence)",
        "gemini-3-pro-preview (Warning: Strict Rate Limits of 2 RPM)"
    ],
    index=0,
    help="Select the underlying intelligence engine. Pro models offer deeper reasoning but strict rate limits. Lite models offer massive throughput."
)

# Strip out the parenthetical text so the backend just gets the raw API model name
actual_model_name = selected_model_display.split(" ")[0]

# Strip out the "(Recommended...)" text so we just have the raw API model name
actual_model_name = selected_model_display.split(" ")[0]

st.sidebar.divider()
st.sidebar.subheader("System Stats")
st.sidebar.text(f"Indexed Chunks: {collection.count()}")
st.sidebar.text(f"Active Engine:\n{actual_model_name}")

# MAIN LAYOUT
col1, col2 = st.columns([4, 1])
with col1:
    question = st.text_input("Research Query:", label_visibility="collapsed", placeholder="Enter a research question...")
with col2:
    run_btn = st.button("Execute Pipeline", use_container_width=True, type="primary")

if run_btn:
    if not question:
        st.warning("Please enter a query.")
    else:
        # Initialize the selected model dynamically
        custom_model = genai.GenerativeModel(actual_model_name)
        
        agent = ResearchAgent(
            model=custom_model, # ignores the default model and uses the user-selected one
            collection=collection, 
            max_steps=3,
            use_planner=use_planner,
            use_reflector=use_reflector,
            use_verifier=use_verifier
        )
        
        f = io.StringIO()
        with redirect_stdout(f):
            with st.spinner(f"Processing trajectory using {actual_model_name}..."):
                f = io.StringIO()
        with redirect_stdout(f):
            with st.spinner(f"Processing trajectory using {actual_model_name}..."):
                try:
                    final_answer = agent.run(question)
                except Exception as e:
                    error_str = str(e).lower()
                    # Check if it's a rate limit/quota error
                    if "429" in error_str or "quota" in error_str or "exhausted" in error_str:
                        # If they are using the Pro model, give them the exact architectural reason
                        if "pro" in actual_model_name.lower():
                            final_answer = (
                                "⚠️ **Rate Limit Exceeded (2 RPM)**\n\n"
                                "Because this Agentic RAG architecture uses a **Reflector** loop to continuously critique and re-search for evidence, "
                                "it makes multiple API calls per question. The Gemini Pro free tier restricts usage to 2 requests per minute, which is instantly exhausted by a single agentic loop.\n\n"
                                "**The Fix:** Please wait 60 seconds before trying again, or switch to the recommended `gemini-flash-lite-latest` model in the sidebar for unlimited high-throughput testing."
                            )
                        else:
                            # Standard rate limit error for other models
                            final_answer = f"⚠️ **API Quota Exceeded.** You may have exhausted your daily free tier.\n\nRaw Error: {e}"
                    else:
                        # Catch any other random Python crashes
                        final_answer = f"**System Error:** {e}"
        
        st.subheader("Synthesis")
        st.info(final_answer)
        
        # Hide the raw text dump inside a dropdown
        with st.expander("View Execution Trace"):
            st.code(f.getvalue(), language="text")