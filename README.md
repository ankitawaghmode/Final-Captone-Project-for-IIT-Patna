# IT Help Desk Assistant

AI-powered IT support assistant built with Streamlit, LangGraph, Gemini, ChromaDB, and SQLite.

## Setup

1. Create a `.env` file from `.env Example`.
2. Add your Gemini API key:

```env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001
```

3. Install dependencies:

```powershell
python -m venv myenv
myenv/scripts/activate$RESOURCE_GROUP="it-helpdesk-rg"
$LOCATION="eastus"
$ACR_NAME="ithelpdeskacr12345"   # must be globally unique
$APP_NAME="it-helpdesk"
$ENV_NAME="it-helpdesk-env"
pip install -r requirements.txt
```

4. Run the application:

```powershell
streamlit run app.py
```

Open `http://localhost:8501`.

## Docker

```powershell
docker build -t it-helpdesk:latest .
docker run --name it-helpdesk --env-file .env -p 8501:8501 it-helpdesk:latest
```

## MCP Server

Run this separately only when using an MCP client:

```powershell
python -m src.mcp_server
```

Keep `.env` private. Never commit or share API keys.
