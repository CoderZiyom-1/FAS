**Fashion Stylist AI Agent** (Multimodal RAG System) (Work-inprogress)

Fashion Stylist AI is a multimodal AI system that provides intelligent outfit styling suggestions using:

1. Computer Vision

2. Color Analysis

3. CLIP Image Embeddings

4. Retrieval-Augmented Generation (RAG)

5. Local LLM (Mistral via Ollama)

Users upload an outfit image and receive professional styling advice grounded in fashion knowledge.

**Tech Stack**

1. CLIP (Image Understanding)

2. LangChain

3. FAISS Vector Database

4. Ollama (Mistral LLM)

5. Streamlit


**Color Palette Extraction**

**Architecture**


1️⃣ Image Upload

2️⃣ Color Palette Extraction

3️⃣ CLIP Visual Embedding

4️⃣ Fashion Knowledge Retrieval (RAG)

5️⃣ LLM Style Reasoning

6️⃣ Final Styling Advice


**Features**

1. Outfit color detection
2. Visual understanding using CLIP
3. AI fashion recommendations
4. Knowledge-grounded styling advice
5. Local AI inference (privacy-friendly)
6. Streamlit interactive UI


**Setup Instructions**

1. Install Ollama


https://ollama.com


Pull model:

ollama pull mistral

2. Install dependencies

pip install -r requirements.txt

3. Ingest Fashion Knowledge PDFs

Update path in:

data_ingestion.py


Run:

python data_ingestion.py

4. Run App

streamlit run app.py

**Use Cases**

AI wardrobe assistant

E-commerce styling

Personal fashion AI

Outfit compatibility analysis

**Future Improvements**

Occasion detection

Body-type recommendations

Trend-aware suggestions


**Author**

Moyiz Khan
