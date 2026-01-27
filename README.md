# Deep_Research_Agent
Deep Research Agent
Deep Research Agent is an autonomous, multi-agent system designed to perform deep-dive technical research. Built with LangGraph and Tavily, it mimics a real-world research team: searching for facts, drafting reports, and performing self-critique until quality standards are met.

🛠️ Tech Stack  
    Framework: LangGraph (Stateful orchestration)  
    LLM: Qwen2.5 / Llama 3 (via Hugging Face Inference API)  
    Search Engine: Tavily AI  
    Frontend: Streamlit  
    Language: Python 3.10+ 
    
Installation & Setup:  

    Clone the repository:
    git clone https://github.com/ahb7/Deep_Research_Agent.git
    cd deep-research-agent

    Install dependencies:
    pip install -r requirements.txt

    Configure Environment Variables: Create a .env file and add your API keys:
    HUGGINGFACEHUB_API_TOKEN=your_token_here
    TAVILY_API_KEY=your_key_here

    Run the Application:
    streamlit run app.py

<br><br>  
<img width="1378" height="790" alt="DRA_Image" src="https://github.com/user-attachments/assets/d21b2f5a-1305-4129-8ced-24c305b45d75" />

