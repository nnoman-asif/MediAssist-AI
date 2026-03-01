# MediAssist AI

An AI-powered medical assistance chatbot that combines large language models with a retrieval-augmented generation (RAG) pipeline. MediAssist ingests medical PDFs, indexes them, and uses the knowledge to provide informed responses to health-related questions -- via text or voice.

## Features

- **Bilingual support** -- English and Urdu (text and voice)
- **Voice chat** -- Speak your question, hear the answer via neural text-to-speech
- **Knowledge-grounded responses** -- Backed by indexed medical documents (PDF upload)
- **Hospital finder** -- Recommends nearby hospitals using Google Maps
- **MLOps tracking** -- All interactions logged with MLflow
- **Agentic architecture** -- LangGraph agent with tool calling

## Tech Stack

| Component | Technology |
|---|---|
| LLM | OpenAI |
| Agent Framework | LangGraph |
| Speech-to-Text | OpenAI Whisper large-v3 |
| Text-to-Speech | Microsoft Edge Neural TTS |
| RAG | LangChain + FAISS + sentence-transformers |
| Location | Google Maps Places API |
| MLOps | MLflow |
| UI | Gradio |

## Setup

### Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA (recommended, 8 GB+ VRAM)
- OpenAI API key
- Google API key

### Installation

```bash
git clone https://github.com/<your-username>/MediAssist.git
cd MediAssist

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
```

### Configuration

Copy the example env file and fill in your credentials:

```bash
copy .env.example .env       # Windows
# cp .env.example .env       # Linux/macOS
```

Edit `.env` with your API keys:

```
OPENAI_API_KEY=your-key-here
OPENAI_MODEL=your-model-here

GOOGLE_MAPS_API_KEY=your-maps-key      # for hospital finder
```

### Running

```bash
python -m app.main
```

On first run, Whisper large-v3 (~3 GB) and the sentence-transformer embedding model will be downloaded to the `models/` directory.

## Usage

1. **Upload PDFs** -- Go to the Knowledge Base tab, upload medical reference PDFs. The system indexes them automatically.
2. **Text chat** -- Type a medical question in English or Urdu in the Chat tab.
3. **Voice chat** -- Click the microphone, speak your question, and the system will transcribe, answer, and read the response back.
4. **Hospital finder** -- If the chatbot recommends a hospital, it will ask for your location and provide nearby options.

## License

This project is licensed under the [MIT License](LICENSE).
