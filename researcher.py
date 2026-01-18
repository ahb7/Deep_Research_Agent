import os
from typing import TypedDict, List, Annotated
import operator
import json
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_tavily import TavilySearch
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# 1. Define the Shared State 
class ResearchState(TypedDict):
    topic: str
    context: Annotated[List[str], operator.add]
    report: str
    iteration: int
    review_status: str
    feedback: str  

# 2. Initialize Hugging Face LLM
# We'll use Mistral-7B-Instruct or Llama-3-8B-Instruct (both excellent & free)
# Use Qwen 2.5 (very powerful) or Llama 3.1
repo_id = "Qwen/Qwen2.5-72B-Instruct"

# Initialize the base endpoint
llm_endpoint = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_new_tokens=2048,  
    temperature=0.1,
    provider="auto",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

# Wrap it in ChatHuggingFace
llm = ChatHuggingFace(llm=llm_endpoint)

search_tool = TavilySearch(max_results=3)

# 3. Define the Nodes
def search_node(state: ResearchState):
    """Action: Fetches data. Refines search if feedback exists."""
    base_topic = state["topic"]
    feedback = state.get("feedback", "")

    # If the Reviewer gave feedback, we narrow the search
    if feedback:
        search_query = f"{base_topic} focus on: {feedback}"
        print(f"\n🔍 [SEARCHER]: Reviewer requested more detail. Refining search to: '{search_query}'")
    else:
        search_query = base_topic
        print(f"\n🔍 [SEARCHER]: Initializing search for: '{search_query}'")

    # Execute search
    raw_results = search_tool.invoke(search_query)

    # Parsing logic to handle the Tavily JSON string
    try:
        if hasattr(raw_results, "content"):
            results_list = json.loads(raw_results.content)
        else:
            results_list = json.loads(raw_results)
        content = [res["content"] for res in results_list]
    except:
        content = [str(raw_results)]

    print(f"   -> Found {len(content)} snippets. Sending to Writer.")
    # We clear feedback here so the loop can refresh
    return {"context": content, "feedback": ""}

def writer_node(state: ResearchState):
    """Action: Synthesizes findings into a report."""
    print(f"\n✍️ [WRITER]: Synthesizing {len(state['context'])} total sources into a report...")

    combined_context = "\n\n".join(state["context"])
    topic = state["topic"]

    prompt = [
        SystemMessage(content="You are a senior technical analyst."),
        HumanMessage(content=f"""
            Write a detailed report on {topic}.
            Ensure the report has a clear Introduction, Body, and a final Conclusion.
            IMPORTANT: Do not stop mid-sentence. If you are reaching your limit, wrap up the final point quickly.
            CONTEXT:
            {combined_context}
        """)
    ]   

    response = llm.invoke(prompt)
    return {"report": response.content}

def reviewer_node(state: ResearchState):
    """Action: Dual-layer critique (Length + AI Reasoning)."""
    report_text = state["report"]
    iteration = state.get("iteration", 0)
    topic = state["topic"]

    print(f"\n⚖️  [REVIEWER]: Commencing Multi-Layer Evaluation (Iteration {iteration + 1})")

    # --- LAYER 1: QUANTITATIVE CHECK (Length) ---
    # Set to 2000 for a realistic challenge 
    MIN_LENGTH = 2000

    if len(report_text) < MIN_LENGTH and iteration < 2:
        feedback_msg = f"Inadequate length ({len(report_text)} chars). Need more technical data."
        print(f"   ❌ LAYER 1 REJECTED: Report too brief.")
        return {
            "review_status": "re-search",
            "iteration": iteration + 1,
            "feedback": feedback_msg
        }

    print(f"   ✅ LAYER 1 PASSED: Length requirement met.")

    # --- LAYER 2: QUALITATIVE CHECK (AI Reasoning) ---
    # We ask the LLM to act as a "Technical Auditor"
    audit_prompt = f"""
    You are a Senior Technical Auditor. Review this report on '{topic}'.
    Check for:
    1. Technical Depth: Does it go beyond surface-level facts?
    2. Missing Gaps: Are there obvious missing technical details?

    REPORT:
    {report_text}

    If the report is perfect, reply with 'PASSED'.
    If it needs more detail, reply with a 1-sentence instruction starting with 'NEED_MORE:'.
    """

    print("   🔍 [REVIEWER]: Analyzing content for technical gaps...")
    # Use your existing LLM to "Judge" the content
    audit_response = llm.invoke(audit_prompt)
    audit_content = audit_response.content if hasattr(audit_response, 'content') else str(audit_response)

    if "NEED_MORE:" in audit_content and iteration < 2:
        feedback_msg = audit_content.replace("NEED_MORE:", "").strip()
        print(f"   ❌ LAYER 2 REJECTED: AI Judge identified gaps.")
        print(f"   💬 FEEDBACK: {feedback_msg}")
        return {
            "review_status": "re-search",
            "iteration": iteration + 1,
            "feedback": feedback_msg
        }

    # --- FINAL APPROVAL ---
    print("   ✅ LAYER 2 PASSED: Content quality approved.")
    print("   🏆 [REVIEWER]: Finalizing high-quality research dossier.")
    return {"review_status": "good", "iteration": 3}

def router(state: ResearchState):
    """Conditional logic to decide next step."""
    if state["iteration"] < 3:
        return "searcher"
    # Must return the string key that matches the mapping dictionary
    return "end"

# --- UPDATED GRAPH ---
workflow = StateGraph(ResearchState)

workflow.add_node("searcher", search_node)
workflow.add_node("writer", writer_node)
workflow.add_node("reviewer", reviewer_node)

workflow.add_edge(START, "searcher")
workflow.add_edge("searcher", "writer")
workflow.add_edge("writer", "reviewer")

workflow.add_conditional_edges(
    "reviewer",
    router,
    {
        "searcher": "searcher",  # If router returns "searcher", go to searcher node
        "end": END               # If router returns END, exit the graph
    }
)

# Create a memory checkpointer
memory = MemorySaver()

# Compile the graph with the checkpointer
app = workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "research_1"}}
    # Change your inputs to something broad to test the loop
    inputs = {"topic": "AI in healthcare", "iteration": 0}
    #inputs = {"topic": "Impact of Open Source LLMs on Enterprise Security 2026", "iteration": 0}

    print("\n" + "="*50)
    print("AGENTIC RESEARCH STARTING")
    print("="*50)
    # Change your inputs to something broad to test the loop
    for chunk in app.stream(inputs, config=config, stream_mode="updates"):
        for node_name, output in chunk.items():
            if "review_status" in output and output["review_status"] == "re-search":
                print(f"\n🔁 LOOP TRIGGERED: Reviewer sent Writer back to Searcher.")
                print(f"   Reason: {output.get('feedback')}")
            print(f"\n[AGENT ACTIVE]: {node_name}")
            print("-" * 20)

            # 1. Check if the Researcher finished
            if "context" in output:
                # Show the first 100 characters of each search snippet
                for i, snippet in enumerate(output["context"]):
                    print(f"  > Search Result {i+1}: {snippet[:150]}...")

            # 2. Check if the Writer finished
            if "report" in output:
                print("  > Status: Draft report generated.")
                # Optional: print the first few lines of the draft
                print(f"  > Preview: {output['report'][:200]}...")

            # 3. Check if the Reviewer finished
            if "iteration" in output:
                if output["iteration"] < 3:
                    print("  > Review Verdict: INSUFFICIENT DATA. Restarting Loop.")
                else:
                    print("  > Review Verdict: QUALITY APPROVED.")

    # Fetch and print final result
    final_state = app.get_state(config).values
    print("\n" + "="*50)
    print("FINAL CONSOLIDATED REPORT")
    print("="*50 + "\n")
    print(final_state.get("report"))


