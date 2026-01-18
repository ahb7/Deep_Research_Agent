import streamlit as st
import time
from researcher import app  # Ensure your compiled graph is named 'app' in researcher.py

st.set_page_config(page_title="Deep Research Agent", page_icon="🕵️‍♂️", layout="wide")

st.title("🕵️‍ Deep Research Agent")
st.markdown("*Advanced Multi-Agent Research with Self-Correction Loops*")

# Sidebar for session management
with st.sidebar:
    st.header("Settings")
    if st.button("Clear Research History"):
        st.session_state.messages = []
        st.rerun()

# Topic Input
topic = st.text_input("What would you like me to research deep into?", placeholder="e.g., Security risks of DeepSeek-V3 in 2026")

if st.button("Start Research"):
    if not topic:
        st.warning("Please enter a topic.")
    else:
        # Initial State with thread_id for persistence
        inputs = {"topic": topic, "iteration": 0}
        config = {"configurable": {"thread_id": f"research_{int(time.time())}"}}
        
        # Placeholders for the UI
        log_container = st.container()
        
        with st.status("🚀 Agents Orchestrating...", expanded=True) as status:
            # We use stream_mode="updates" to catch the node-by-node handoffs
            for chunk in app.stream(inputs, config=config, stream_mode="updates"):
                for node_name, output in chunk.items():
                    
                    if node_name == "searcher":
                        st.write("🔍 **Researcher:** Scouring technical databases via Tavily...")
                        if output.get("context"):
                            st.caption(f"Found {len(output['context'])} new data points.")
                    
                    elif node_name == "writer":
                        st.write("✍️ **Writer:** Drafting the technical dossier...")
                    
                    elif node_name == "reviewer":
                        if output.get("review_status") == "re-search":
                            st.error(f"⚠️ **Reviewer Rejected Draft:** {output.get('feedback')}")
                            st.write("🔄 *Agentic Loop Triggered: Returning to search phase...*")
                        else:
                            st.success("✅ **Reviewer Approved:** Quality standards met.")
                            time.sleep(2)
            
            status.update(label="Research Complete!", state="complete", expanded=False)

        # Final Result Rendering
        final_state = app.get_state(config).values
        report = final_state.get("report", "Error generating report.")
        
        st.divider()
        st.subheader("Final Research Dossier")
        st.markdown(report)
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("💾 Download .MD", report, file_name="research_report.md")
        with col2:
            if st.button("📋 Copy to Clipboard"):
                st.toast("Report copied (simulated)!")
                
