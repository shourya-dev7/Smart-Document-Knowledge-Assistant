# Smart Document Knowledge Assistant

## 1. Project Overview

The Smart Document Knowledge Assistant is a web-based document question-answering system designed to help students quickly find and understand information from their lecture notes, research articles, lab manuals, and other study documents.

The system uses Retrieval-Augmented Generation (RAG) to answer questions based on information retrieved from the documents uploaded by the user.

The application accepts PDF and TXT files, searches their content semantically, retrieves the most relevant sections, and provides the retrieved context to a Large Language Model (LLM) to generate a grounded answer.

---

## 2. Problem Statement

Students and researchers often work with large collections of unstructured documents such as lecture notes, research papers, lab manuals, and study material.

Traditional keyword-based searching can become slow and ineffective when:

- The user does not remember the exact wording used in the document.
- The question uses synonyms or different wording.
- Relevant information is spread across different parts of a document.
- The user wants to ask a natural-language question instead of searching for keywords.

Therefore, there is a need for a system that can understand the meaning of a question, retrieve relevant information from uploaded documents, and provide a concise answer based only on that information.

---

## 3. Objective

The objective of this project is to build a web-based Knowledge Assistant that:

- Accepts PDF and TXT documents.
- Extracts text from the uploaded documents.
- Splits documents into smaller chunks.
- Converts the chunks into vector embeddings.
- Performs semantic similarity search.
- Retrieves the most relevant document chunks.
- Sends the retrieved context and user question to an LLM.
- Generates an answer grounded in the uploaded documents.
- Supports follow-up questions.
- Displays the document snippets used to generate the answer.

---

## 4. Proposed Solution

The proposed system uses a Retrieval-Augmented Generation (RAG) pipeline.

Instead of directly sending the entire document to the LLM, the application first searches the uploaded documents and identifies the most relevant chunks.

These retrieved chunks are then provided to the Gemini LLM along with the user's question.

The LLM is instructed to answer only using the retrieved document context.

This approach helps keep the generated response connected to the information contained in the user's documents.

---

## 5. System Workflow

```text
User Uploads Documents
          ↓
     Text Extraction
          ↓
      Text Chunking
          ↓
   Generate Embeddings
          ↓
   Semantic Similarity
        Search
          ↓
 Retrieve Top Relevant
        Chunks
          ↓
 Combine Retrieved
       Context
          ↓
      Gemini LLM
          ↓
 Grounded Answer
          ↓
 Display Answer + Sources
6. Technology Stack
Programming Language

Python

Python is used to implement the document processing, semantic search, retrieval pipeline, and interaction with the LLM.

User Interface

Streamlit

Streamlit is used to create the web-based interactive interface.

PDF Processing

PyMuPDF

PyMuPDF is used to extract text from PDF documents while preserving page information.

Embeddings

Sentence Transformers

The all-MiniLM-L6-v2 model is used to convert document chunks and search queries into numerical vector representations called embeddings.

Similarity Search

Scikit-learn

Cosine similarity is used to compare the embedding of the user's question with the embeddings of document chunks.

Large Language Model

Google Gemini API

Gemini is used to generate the final answer using the retrieved document context.

## RAG Implementation

The application follows a Retrieval-Augmented Generation (RAG) approach so that the language model generates answers using information retrieved from the uploaded documents.

### 1. Document Processing

Uploaded PDF files are processed using PyMuPDF, which extracts text page by page. TXT files are read directly.

Each extracted document is divided into chunks of approximately 400 characters with an overlap of approximately 150 characters.

The overlap helps reduce context loss when relevant information occurs near a chunk boundary.

Each chunk stores metadata such as:

- Source file name
- Page number for PDF files
- Chunk number
- Detected problem section, when applicable

### 2. Embedding Generation

Each document chunk is converted into a numerical vector using the `all-MiniLM-L6-v2` Sentence Transformer model.

These vectors represent the semantic meaning of the chunks and allow the system to compare the meaning of a user's question with the meaning of document content.

### 3. Semantic Retrieval

When the user submits a question, the question is also converted into an embedding.

Cosine similarity is then calculated between the question embedding and all document chunk embeddings.

A relevance threshold of `0.20` is used to remove chunks with very low semantic similarity.

### 4. Combined Retrieval Ranking

The system ranks relevant chunks using a combined retrieval score.

The initial score is the semantic similarity score. A small keyword-based boost is then added when important words from the question occur in a chunk.

The system also provides an additional section-specific boost when the user explicitly mentions a detected section such as `PROBLEM 02`.

The final ranking therefore considers:

- Semantic similarity
- Keyword matches
- Explicit section references

The system selects up to five relevant chunks that satisfy the semantic similarity threshold.

### 5. Context-Aware Generation

The retrieved chunks are combined into a document context containing the source file, page number, chunk number, and extracted text.

This context, together with the user's question, is sent to the Gemini API.

The prompt instructs Gemini to answer only from the provided document context and to state that the answer could not be found if the required information is not present in the retrieved content.

This allows the language model to generate a natural-language answer while keeping the uploaded documents as the primary knowledge source.

## Why RAG?

A standard language model can generate answers from its pre-trained knowledge, but that knowledge does not necessarily contain the specific lecture notes, lab manuals, or documents uploaded by a student.

RAG solves this problem by separating the process into two stages:

1. Retrieve relevant information from the user's documents.
2. Provide the retrieved information to the language model for answer generation.

This allows the application to answer questions based on the user's own documents instead of relying only on the model's general knowledge.

It also makes the retrieved sources visible to the user, allowing them to inspect the document content used to generate the answer.

## Follow-up Questions

The application supports follow-up questions within the same conversation.

A controlled set of explicit follow-up phrases is detected, including phrases such as:

- "second requirement"
- "first requirement"
- "previous requirement"
- "this requirement"
- "that requirement"
- "previous one"
- "above one"

When one of these phrases is detected, the previous question and previous answer are combined with the current question before performing semantic retrieval.

This helps the retrieval system understand references to information discussed immediately before.

The previous conversation is also provided to Gemini so that the generated response can remain consistent with the ongoing conversation.

## Source Referencing

After generating an answer, the application displays the document chunks used during retrieval.

For each retrieved chunk, the interface shows:

- Source file name
- Page number for PDF documents
- Chunk number
- Semantic similarity score
- Combined retrieval score
- Retrieved document text

This provides transparency into which parts of the uploaded documents were considered relevant to the user's question.

The displayed snippets also allow the user to verify the answer against the original document content.

## Relevance Filtering

The system uses a semantic similarity threshold of `0.20` to filter out chunks that have very low similarity to the user's question.

After filtering, the remaining chunks are ranked using the combined retrieval score, which includes semantic similarity, keyword matching, and an additional boost for explicitly mentioned document sections.

Up to five chunks that satisfy the semantic similarity threshold are selected and passed to the generation stage.

If no chunk satisfies the relevance requirement, the system does not call the language model and instead informs the user that relevant information could not be found in the uploaded documents.

## Testing

The application was tested using both relevant and irrelevant questions to verify document retrieval, follow-up handling, multiple-document support, and grounding behavior.

| Test | Input / Scenario | Expected Behavior | Result |
|---|---|---|---|
| Direct document question | "What are the core requirements of Problem 02?" | Retrieve relevant Problem 02 content and generate an answer from the document | Passed |
| Follow-up question | "What does the second requirement mean?" | Use the previous conversation to understand the reference and retrieve the relevant requirement | Passed |
| Multiple-document search | Question about qubits with both a TXT file and PDF uploaded | Retrieve the relevant TXT content instead of unrelated PDF content | Passed |
| Out-of-document question | "Who is the Prime Minister of India?" | Do not answer using outside knowledge when the information is absent from the uploaded documents | Passed |
| Source display | Any successful question | Display source file, chunk information, similarity score, and retrieved text | Passed |

These tests confirmed that the application can retrieve relevant document content, maintain basic conversational context, and handle questions for which relevant information is not available in the uploaded documents.

12. Error Handling

The application includes handling for several possible problems:

Missing Gemini API key
Gemini API generation errors
Empty or unreadable documents
Questions with no relevant document content
Empty questions

## Security

The Gemini API key is not stored directly in the source code.

During local development, the application reads the key from the `GEMINI_API_KEY` environment variable.

For deployment, the API key is stored using Streamlit Community Cloud Secrets rather than being committed to the GitHub repository.

The project also includes a `.gitignore` file that excludes environment files, Python cache files, virtual environments, and Streamlit secrets.

A repository scan was performed to check that the Gemini API key was not accidentally included in the project files.

## Project Structure

```text
Smart-Document-Knowledge-Assistant/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── embedding_test.py
├── gemini_test.py
├── similarity_test.py
└── test.txt

## Deployment

The application was deployed using Streamlit Community Cloud.

The deployed application is connected to the GitHub repository containing the project source code.

The Gemini API key is configured through Streamlit Community Cloud Secrets and is not included in the public repository.

Live application:

https://smart-document-knowledge-assistant-zfvbiiaj4cezucgnzpnb6m.streamlit.app/

## Limitations

The current implementation has several limitations:

- PDF processing currently depends on machine-readable text. Scanned PDFs requiring OCR are not supported.
- Uploaded documents are used within the current application session rather than being stored in a persistent document database.
- The chunking strategy is based on character length and overlap rather than document-aware semantic sections.
- Follow-up retrieval currently handles a controlled set of explicit reference phrases rather than every possible conversational reference.
- The application currently retrieves up to five relevant chunks, subject to the semantic similarity threshold.

## Future Improvements

Possible improvements include:

- Adding OCR support for scanned PDF documents.
- Using more advanced document-aware chunking based on paragraphs, headings, and sections.
- Adding persistent document storage so users can maintain a knowledge base across sessions.
- Improving conversational reference resolution for more natural follow-up questions.
- Adding document deletion and management features.
- Supporting additional document formats.
- Improving retrieval using more advanced hybrid search and reranking techniques.

## Conclusion

The Smart Document Knowledge Assistant demonstrates a practical implementation of Retrieval-Augmented Generation for student documents.

The application combines PDF/TXT text extraction, overlapping chunking, Sentence Transformer embeddings, cosine similarity, keyword-based retrieval improvements, section-aware ranking, and Gemini-based response generation.

The system also provides conversational follow-up support, relevance filtering, and visible source snippets so users can understand which document content was used for an answer.

The project is deployed as a functional Streamlit web application and demonstrates how RAG can make large collections of student documents easier to search and interact with using natural-language questions.