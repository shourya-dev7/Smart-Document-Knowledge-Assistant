# Smart Document Knowledge Assistant

A web-based document question-answering system that uses Retrieval-Augmented Generation (RAG) to provide context-aware answers from uploaded PDF and TXT documents.

## Problem

Students and researchers often work with large collections of lecture notes, research papers, lab manuals, and other documents. Traditional keyword search can be slow and may fail to understand the context of a question.

## Solution

The Smart Document Knowledge Assistant allows users to:

- Upload multiple PDF and TXT documents
- Extract and split document text into chunks
- Convert chunks into vector embeddings
- Perform semantic similarity search
- Retrieve the top relevant document chunks
- Send the retrieved context and user question to an LLM
- Ask follow-up questions
- Display the document snippets used to generate the answer

## Technology Stack

- Python
- Streamlit
- PyMuPDF
- Sentence Transformers
- Scikit-learn
- Google Gemini API

## RAG Pipeline

```text
Upload Documents
       ↓
Text Extraction
       ↓
Text Chunking
       ↓
Sentence Embeddings
       ↓
Semantic Similarity Search
       ↓
Top Relevant Chunks
       ↓
Gemini LLM
       ↓
Grounded Answer + Sources


## Features

Multiple PDF/TXT file upload
Semantic document search
Top 3–5 relevant chunks retrieval
Context-aware Gemini responses
Follow-up questions
Conversation history
Referenced document snippets
Relevance filtering
Document and chunk information
Clear conversation option


## Project Structure

Smart-Document-Knowledge-Assistant/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── embedding_test.py
├── gemini_test.py
├── similarity_test.py
├── test.txt
└── documents/


## How to Run Locally

Install the required packages:
pip install -r requirements.txt
Set the Gemini API key as an environment variable.


## Windows PowerShell

$env:GEMINI_API_KEY="YOUR_API_KEY"
Run the application:

python -m streamlit run app.py


## Security

The Gemini API key is not stored in the source code. It is read from the GEMINI_API_KEY environment variable.

Sensitive files and local environment files are excluded using .gitignore.