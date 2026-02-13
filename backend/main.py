import os
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from agent import build_graph
import aiosqlite

# RAG Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import mlflow
from prometheus_client import Counter

load_dotenv("../.env")

DB_PATH = "database.sqlite"

# Prometheus metric: number of chunks uploaded
MLFLOW_UPLOAD_CHUNKS = Counter("upload_chunks_total", "Total number of chunks uploaded to vector store")

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with aiosqlite.connect(DB_PATH) as db_conn:
        checkpointer = AsyncSqliteSaver(db_conn)
        await checkpointer.setup()
        app.state.graph = build_graph(checkpointer)
        yield

app = FastAPI(title="Pinecone RAG Agent", lifespan=lifespan)

# --- MODELS ---
class TaskRequest(BaseModel):
    task: str
    thread_id: str

class FeedbackRequest(BaseModel):
    thread_id: str
    action: str
    feedback: str = ""

# --- ENDPOINTS ---

@app.post("/upload")
async def upload_file(thread_id: str = Form(...), file: UploadFile = File(...)):
    """Processes a PDF and uploads vectors to Pinecone Namespace."""
    try:
        # 1. Save Temp File
        temp_filename = f"temp_{file.filename}"
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. Process PDF
        loader = PyPDFLoader(temp_filename)
        docs = loader.load()
        
        # 3. Split Text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        print(f"--- PDF SPLIT: {len(splits)} chunks ---")
        
        # 4. Upsert to Pinecone (Cloud)
        # <--- CHANGED: Pinecone Logic
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        # Default to the existing index in your Pinecone project if unset
        index_name = os.getenv("PINECONE_INDEX_NAME", "langchain-pinecone-rag")

        # Start an MLflow run to track this upload
        mlflow_run = None
        try:
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
            mlflow_run = mlflow.start_run(run_name=f"upload-{thread_id}")
            mlflow.log_param("pinecone_index", index_name)
            mlflow.log_param("namespace", thread_id)
            mlflow.log_param("chunks_attempting", len(splits))

            # Attempt upload and provide a clearer error message if the index is missing
            PineconeVectorStore.from_documents(
                documents=splits,
                embedding=embeddings,
                index_name=index_name,
                namespace=thread_id  # Segregates data by user/thread
            )

            # Log success metrics
            mlflow.log_metric("chunks_uploaded", len(splits))
            MLFLOW_UPLOAD_CHUNKS.inc(len(splits))

        except Exception as e:
            err_msg = str(e)
            guidance = (
                f"Pinecone upload failed: {err_msg}.\n"
                "Check that your PINECONE_API_KEY and PINECONE_ENV (or equivalent) are set, "
                "and that the index exists. To use a different index, set the env var "
                "PINECONE_INDEX_NAME to a valid index (for example: 'langchain-pinecone-rag')."
            )
            print(guidance)
            if mlflow_run:
                mlflow.log_param("error", err_msg)
            raise HTTPException(status_code=500, detail=guidance)
        finally:
            if mlflow_run:
                mlflow.end_run()
        
        # Cleanup
        os.remove(temp_filename)
        return {"status": "success", "chunks": len(splits), "message": "PDF Uploaded to Pinecone"}
        
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

# ... (The rest of the endpoints: /start, /status, /review remain exactly the same) ...
# Just copy the existing /start, /status, and /review endpoints from the previous main.py
@app.post("/start")
async def start_research(req: TaskRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    # Initialize state
    initial_state = {"task": req.task, "revision_number": 0, "critique": None, "pdf_data": ""}
    async for event in app.state.graph.astream(initial_state, config):
        pass
    return {"status": "started"}

@app.get("/status/{thread_id}")
async def get_status(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = await app.state.graph.aget_state(config)
    except:
        return {"status": "error"}

    if not snapshot.values:
        return {"status": "empty"}
    
    next_step = snapshot.next
    is_paused = "human_review" in next_step if next_step else False
    
    return {
        "status": "paused" if is_paused else "completed",
        "draft": snapshot.values.get("draft_report", ""),
        "revision": snapshot.values.get("revision_number", 0)
    }

@app.post("/review")
async def review_task(req: FeedbackRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    if req.action == "approve":
        async for event in app.state.graph.astream(None, config):
            pass
        return {"status": "completed"}
    elif req.action == "revise":
        await app.state.graph.aupdate_state(config, {"critique": req.feedback})
        async for event in app.state.graph.astream(None, config):
            pass
        return {"status": "re-researching"}