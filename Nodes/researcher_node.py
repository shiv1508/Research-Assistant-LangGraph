import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

# Import the specific tools we need
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

class ResearchState(TypedDict):

    question: str
    web_data: str
    arxiv_data: str
    final_report: str
    summary: str

# Web Search Tool (Tavily)
tavily_tool = TavilySearchResults(max_results=3)

# Academic Search Tool (ArXiv)
arxiv_wrapper = ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=2000)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

def researcher_node(state: ResearchState):
    """
    Takes the query from the state, searches both Web and ArXiv,
    and updates the state with the results.
    """
    print(f"--- RESEARCHING: {state['question']} ---")
    
    # 1. Search the Web (Tavily)
    print("Searching Web...")
    try:
        web_results = tavily_tool.invoke(state['question'])
        # Format the web results nicely
        web_context = "\n".join([f"- {d['content']} (Source: {d['url']})" for d in web_results])
    except Exception as e:
        web_context = f"Web search failed: {e}"

    # 2. Search Academic Papers (ArXiv)
    print("Searching ArXiv...")
    try:
        arxiv_context = arxiv_tool.invoke(state['question'])
    except Exception as e:
        arxiv_context = f"ArXiv search failed: {e}"

    # 3. Return updates to the State
    # We only return the keys we want to update.
    return {
        "web_data": web_context,
        "arxiv_data": arxiv_context
    }

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_openai import ChatOpenAI  # Uncomment if using OpenAI

# --- 1. Setup the LLM ---
# Ensure GOOGLE_API_KEY is set in your environment
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# --- 2. The Writer Node ---
def writer_node(state):
    """
    Synthesizes the raw research data into a structured markdown report.
    """
    print("--- WRITER: DRAFTING REPORT ---")
    
    # Create the prompt template
    writer_prompt = ChatPromptTemplate.from_template(
        """
        You are an expert technical writer. Write a comprehensive research report on the topic: "{question}".
        
        Use the following information to answer the user request:
        
        --- WEB DATA ---
        {web_data}
        
        --- ACADEMIC DATA (ArXiv) ---
        {arxiv_data}
        
        INSTRUCTIONS:
        1. Structure the report with clear Headers (Introduction, Key Findings, Technical Details, Conclusion).
        2. You MUST cite your sources. Use [Source URL] or [Paper Title] notation.
        3. If web and academic data conflict, note the discrepancy.
        4. Keep the tone professional and objective.
        """
    )
    
    # Chain: Prompt -> LLM
    chain = writer_prompt | llm
    
    # Execute
    response = chain.invoke({
        "question": state["question"],
        "web_data": state.get("web_data", "No web data found."),
        "arxiv_data": state.get("arxiv_data", "No academic papers found.")
    })
    
    return {"final_report": response.content}

# --- 3. The Summarizer Node ---
def summarizer_node(state):
    """
    Distills the long report into an executive summary.
    """
    print("--- SUMMARIZER: REFINING ---")
    
    summary_prompt = ChatPromptTemplate.from_template(
        """
        Read the following research report and provide a "Bottom Line Up Front" (BLUF) summary.
        
        --- REPORT ---
        {final_report}
        
        INSTRUCTIONS:
        1. Provide 3-5 bullet points of the most critical insights.
        2. Identify one "Key Takeaway" or action item.
        3. Max length: 200 words.
        """
    )
    
    chain = summary_prompt | llm
    
    response = chain.invoke({
        "final_report": state["final_report"]
    })
    
    return {"summary": response.content}

# Re-define the graph with the real nodes
workflow = StateGraph(ResearchState)

# Add the nodes we defined
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)
workflow.add_node("summarizer", summarizer_node)

# Set the Flow
workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "summarizer")
workflow.add_edge("summarizer", END)

# Compile
app = workflow.compile()

# --- Run the Full Agent ---
if __name__ == "__main__":
    print("Initializing Research Agent...")
    user_query = "What are the latest advancements in solid-state batteries for EVs?"
    
    # Run the graph
    result = app.invoke({"question": user_query})
    
    print("\n" + "="*50)
    print("EXECUTIVE SUMMARY")
    print("="*50)
    print(result["summary"])
    
    print("\n" + "="*50)
    print("FULL RESEARCH REPORT")
    print("="*50)
    print(result["final_report"])