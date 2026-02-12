import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnableConfig  # <--- FIX: Import this

# --- STATE DEFINITION ---
class AgentState(TypedDict):
    task: str
    web_data: str
    pdf_data: str
    draft_report: str
    critique: str
    revision_number: int

# --- NODES ---

# <--- FIX: Update type hint from 'dict' to 'RunnableConfig'
async def researcher_node(state: AgentState, config: RunnableConfig):
    """Searches Web AND Pinecone Vector DB."""
    
    # 1. Get Thread ID (Used as Pinecone Namespace)
    # RunnableConfig behaves like a dict, so this access pattern still works perfectly
    thread_id = config["configurable"].get("thread_id")
    
    query = state["task"]
    critique = state.get("critique")
    search_query = f"{query} - {critique}" if critique else query
    print(f"--- RESEARCHER: Searching '{search_query}' (Namespace: {thread_id}) ---")
    
    # A. WEB SEARCH
    tavily = TavilySearchResults(max_results=3)
    try:
        web_results = await tavily.ainvoke(search_query)
        web_text = "\n".join([f"- {d['content']}" for d in web_results])
    except:
        web_text = "Web search failed."

    # B. PINECONE SEARCH (RAG)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    index_name = os.getenv("PINECONE_INDEX_NAME", "research-agent")
    
    try:
        vector_store = PineconeVectorStore(
            index_name=index_name, 
            embedding=embeddings, 
            namespace=thread_id 
        )
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        docs = await retriever.ainvoke(search_query)
        
        if docs:
            pdf_text = "\n".join([f"[Source: PDF] {d.page_content}" for d in docs])
        else:
            pdf_text = "No relevant PDF context found."
            
    except Exception as e:
        print(f"Pinecone Error: {e}")
        pdf_text = "Vector DB unavailable."
    
    return {"web_data": web_text, "pdf_data": pdf_text}

async def writer_node(state: AgentState):
    """Drafts report using Web + PDF Data."""
    print("--- WRITER: Drafting report ---")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = ChatPromptTemplate.from_template(
        """
        You are a Senior Technical Analyst. Write a comprehensive report on: "{task}".
        
        --- PDF / LOCAL DATA (High Priority) ---
        {pdf_data}
        
        --- WEB DATA ---
        {web_data}
        
        --- FEEDBACK TO ADDRESS ---
        {critique}
        
        INSTRUCTIONS:
        1. Prioritize information from the PDF.
        2. If PDF and Web conflict, mention the discrepancy.
        3. Use Markdown with clear headers.
        """
    )
    chain = prompt | llm
    response = await chain.ainvoke({
        "task": state["task"],
        "web_data": state["web_data"],
        "pdf_data": state["pdf_data"],
        "critique": state.get("critique", "None")
    })
    
    return {
        "draft_report": response.content, 
        "revision_number": state.get("revision_number", 0) + 1
    }

async def human_review_node(state: AgentState):
    pass

# --- GRAPH BUILDER ---
def build_graph(checkpointer):
    workflow = StateGraph(AgentState)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("human_review", human_review_node)
    
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", "human_review")
    
    def should_continue(state):
        if state.get("critique"):
            return "researcher"
        return END

    workflow.add_conditional_edges("human_review", should_continue)
    
    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"]
    )