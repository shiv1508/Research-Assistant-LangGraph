import streamlit as st
import requests
import time
import uuid

# Configuration
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Pinecone Research Agent", layout="wide")

# --- SESSION MANAGEMENT ---
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())[:8]

# --- SIDEBAR: SETUP & UPLOAD ---
st.sidebar.header("📂 Project Setup")

# 1. Thread Management
project_name = st.sidebar.text_input("Project Name", value="alpha")
# We use this ID as the Pinecone Namespace to keep data separate
thread_id = f"{st.session_state.user_id}_{project_name}"
st.sidebar.caption(f"Thread ID: `{thread_id}`")

st.sidebar.divider()

# 2. PDF Uploader (The RAG Feature)
st.sidebar.subheader("📄 Knowledge Base")
uploaded_file = st.sidebar.file_uploader("Upload PDF (Optional)", type="pdf")

if uploaded_file:
    if st.sidebar.button("Upload to Pinecone"):
        with st.spinner("Indexing PDF..."):
            # Prepare the file and data for the API
            files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
            data = {"thread_id": thread_id}
            
            try:
                # Call the Backend /upload endpoint
                resp = requests.post(f"{API_URL}/upload", files=files, data=data)
                
                if resp.status_code == 200:
                    st.sidebar.success("✅ PDF Indexed!")
                    st.sidebar.json(resp.json()) # Show details (chunks, etc)
                else:
                    st.sidebar.error(f"Upload failed: {resp.text}")
            except requests.exceptions.ConnectionError:
                st.sidebar.error("🚨 Backend is offline.")

# --- MAIN UI ---
st.title("🌲 Pinecone Research Assistant")
st.markdown(
    """
    This agent uses **Pinecone** (Serverless Vector DB) for long-term memory.
    1. **Upload a PDF** in the sidebar.
    2. **Start a Task** below.
    3. The agent will search **both** the PDF and the Web.
    """
)

# 1. Start Research Task
with st.expander("🚀 Start New Research", expanded=True):
    task_input = st.text_input("Topic:", placeholder="e.g. Summarize the uploaded PDF and find 2024 updates online")
    
    if st.button("Start Agent"):
        if not task_input:
            st.warning("Please enter a topic.")
        else:
            with st.spinner("Initializing Agent..."):
                try:
                    requests.post(f"{API_URL}/start", json={"task": task_input, "thread_id": thread_id})
                    time.sleep(1) # Give backend a moment
                    st.rerun()
                except:
                    st.error("Backend connection failed.")

# 2. Status Polling & Interaction
st.divider()

if st.button("🔄 Refresh Status"):
    st.rerun()

try:
    # Check status from Backend
    status_resp = requests.get(f"{API_URL}/status/{thread_id}")
    
    if status_resp.status_code == 200:
        data = status_resp.json()
        status = data.get("status")
        draft = data.get("draft")

        if status == "empty":
            st.info("No active research. Start one above!")

        elif status == "paused":
            # --- HITL (Human-in-the-Loop) Interface ---
            st.warning("⚠️ Agent Paused: Human Review Required")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### 📄 Draft Report")
                st.markdown(draft)
                
            with col2:
                st.markdown("### 👮‍♂️ Your Feedback")
                feedback = st.text_area("Critique (Leave empty to approve)")
                
                c1, c2 = st.columns(2)
                if c1.button("✅ Approve"):
                    requests.post(f"{API_URL}/review", json={"thread_id": thread_id, "action": "approve"})
                    st.success("Approved! Finalizing...")
                    time.sleep(1)
                    st.rerun()
                    
                if c2.button("↩️ Revise"):
                    if not feedback:
                        st.error("Please enter feedback.")
                    else:
                        requests.post(f"{API_URL}/review", json={"thread_id": thread_id, "action": "revise", "feedback": feedback})
                        st.info("Feedback sent! Agent is re-researching...")
                        time.sleep(1)
                        st.rerun()

        elif status == "completed":
            st.balloons()
            st.success("✅ Research Task Complete")
            st.markdown("### 🏁 Final Report")
            st.markdown(draft)
            
    else:
        st.error("Failed to fetch status.")

except Exception as e:
    st.warning("Waiting for backend...")