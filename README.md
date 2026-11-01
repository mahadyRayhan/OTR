# 🚀 Multi-Agent System

### Setup
1. Install dependencies: `pip install google-generativeai python-dotenv`
2. Add your Google API Key to the `.env` file.
3. Run the system: `python main.py`

### Testing RAG Strategies
You can force the RAG agent to use a specific cognitive framework by appending `--strategy=<name>` to your question.
* Example: `What is bioinformatics? --strategy=tot`
* Example: `How does BLAST work? --strategy=got`