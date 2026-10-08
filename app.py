import streamlit as st
import fitz
import os

from google import genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.button("Clear Conversation"):
    st.session_state.chat_history = []
    st.success("Conversation cleared.")

def split_into_chunks(text, chunk_size=400, overlap=150):
    words = text.split()

    chunks = []
    start = 0

    while start < len(words):

        current_chunk = []
        current_length = 0
        index = start

        while index < len(words):
            word = words[index]

            if current_length + len(word) + 1 <= chunk_size:
                current_chunk.append(word)
                current_length += len(word) + 1
                index += 1
            else:
                break

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        if index >= len(words):
            break

        overlap_words = 0
        overlap_length = 0

        for i in range(index - 1, start - 1, -1):

            word_length = len(words[i]) + 1

            if overlap_length + word_length <= overlap:
                overlap_length += word_length
                overlap_words += 1
            else:
                break

        start = index - overlap_words

    return chunks

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def create_embeddings(chunk_texts):
    model = load_embedding_model()
    return model.encode(chunk_texts)


@st.cache_resource
def load_gemini_client():
    return genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY")
    )


st.title("Smart Document Knowledge Assistant")

st.write("Upload your documents and ask questions about them.")

uploaded_files = st.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)


if uploaded_files:

    st.success(f"{len(uploaded_files)} file(s) uploaded.")
    st.info(f"{len(uploaded_files)} document(s) are currently being used.")
    st.write("**Documents in knowledge base:**")

    for file in uploaded_files:
        st.write(f"- {file.name}")

    all_chunks = []

    for file in uploaded_files:

        if file.name.endswith(".pdf"):
            document = fitz.open(
                stream=file.read(),
                filetype="pdf"
            )

            pages = []

            for page_number, page in enumerate(document, start=1):
                pages.append({
                    "text": page.get_text(),
                    "page_number": page_number
                })

        elif file.name.endswith(".txt"):
            text = file.read().decode("utf-8")

        if file.name.endswith(".pdf"):

            chunk_number = 1
            current_section = None

            for page in pages:

                page_text = page["text"]

                # Detect problem section
                if "PROBLEM 01" in page_text:
                    current_section = "PROBLEM 01"

                elif "PROBLEM 02" in page_text:
                    current_section = "PROBLEM 02"

                elif "PROBLEM 03" in page_text:
                    current_section = "PROBLEM 03"

                elif "PROBLEM 04" in page_text:
                    current_section = "PROBLEM 04"


                chunks = split_into_chunks(page_text)

                for chunk in chunks:

                   all_chunks.append({
                       "text": chunk,
                       "source": file.name,
                       "page_number": page["page_number"],
                       "chunk_number": chunk_number,
                       "section": current_section
                   })

                   chunk_number += 1

            st.write(f"### {file.name}")
            st.write(f"Number of chunks: {chunk_number - 1}")


        elif file.name.endswith(".txt"):

            chunks = split_into_chunks(text)

            for i, chunk in enumerate(chunks):

                all_chunks.append({
                    "text": chunk,
                    "source": file.name,
                    "page_number": None,
                    "chunk_number": i + 1,
                    "section": None         
                })

            st.write(f"### {file.name}")
            st.write(f"Number of chunks: {len(chunks)}")

    st.write("---")

    with st.form("question_form"):

        question = st.text_input(
            "Ask a question about your documents:"
        )

        submitted = st.form_submit_button("Ask")


    if submitted and question.strip():

        # -----------------------------
        # STEP 1: Semantic Search
        # -----------------------------

        model = load_embedding_model()

        chunk_texts = [
            chunk["text"]
            for chunk in all_chunks
        ]

        if not chunk_texts:
            st.warning(
                "No readable text was found in the uploaded documents."
            )
            st.stop()

        chunk_embeddings = create_embeddings(chunk_texts)

        search_query = question

        # Detect clear follow-up questions
        question_lower = question.lower()

        follow_up_phrases = [
            "second requirement",
            "first requirement",
            "third requirement",
            "fourth requirement",
            "previous requirement",
            "above requirement",
            "this requirement",
            "that requirement",
            "previous one",
            "above one",
            "this one",
            "that one"
        ]

        is_follow_up = False

        for phrase in follow_up_phrases:
            if phrase in question_lower:
                is_follow_up = True
                break

        if is_follow_up and st.session_state.chat_history:

            last_chat = st.session_state.chat_history[-1]

            search_query = f"""
            Previous question:
            {last_chat["question"]}

            Previous answer:
            {last_chat["answer"]}

            Current follow-up question:
            {question}
            """

        question_embedding = model.encode([search_query])

        similarities = cosine_similarity(
            question_embedding,
            chunk_embeddings
        )[0]


        # -----------------------------
        # STEP 2: Relevance Check
        # -----------------------------

        relevance_threshold = 0.20

        candidate_indices = [
            index
            for index in similarities.argsort()[::-1]
            if similarities[index] >= relevance_threshold
        ]

        if not candidate_indices:
            st.warning(
                "I couldn't find relevant information in the uploaded documents."
            )
            st.stop()

        # -----------------------------
        # STEP 3: Combined Retrieval Score
        # -----------------------------

        question_lower = question.lower()

        # Start with semantic similarity
        combined_scores = similarities.copy()


        # Give a small boost when query words
        # appear inside a chunk.
        question_words = question_lower.split()

        for index, chunk in enumerate(all_chunks):

            chunk_text_lower = chunk["text"].lower()

            keyword_matches = 0

            for word in question_words:

                if len(word) > 2 and word in chunk_text_lower:
                    keyword_matches += 1

            combined_scores[index] += keyword_matches * 0.02


        # ------------------------------------------------
        # Extra boost for an explicitly mentioned section
        # ------------------------------------------------

        for index, chunk in enumerate(all_chunks):

            if chunk["section"] is not None:

                section_name = chunk["section"].lower()

                if section_name in question_lower:
                    combined_scores[index] += 0.30


        # Sort chunks using the combined score
        ranked_indices = combined_scores.argsort()[::-1]

        # -----------------------------
        # STEP 4: Select Top 3–5 Relevant Chunks
        # -----------------------------

        top_indices = []

        for index in ranked_indices:

            if similarities[index] >= relevance_threshold:

                top_indices.append(index)

            if len(top_indices) == 5:
                break


        if not top_indices:

            st.warning(
                "I couldn't find relevant information in the uploaded documents."
            )
            st.stop()
        
        # -----------------------------
        # STEP 2: Collect Top 5 Chunks
        # -----------------------------

        retrieved_chunks = []

        for index in top_indices:

            chunk = all_chunks[index]

            retrieved_chunks.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "page_number": chunk["page_number"],
                "chunk_number": chunk["chunk_number"],
                "similarity": similarities[index],
                "retrieval_score": combined_scores[index]
            })


        # -----------------------------
        # STEP 3: Create Context
        # -----------------------------

        context = ""

        for i, chunk in enumerate(retrieved_chunks):

            context += f"""
SOURCE {i + 1}
File: {chunk["source"]}
Page: {chunk["page_number"]}
Chunk: {chunk["chunk_number"]}

{chunk["text"]}

-------------------------
"""


        # -----------------------------
        # STEP 4: Ask Gemini
        # -----------------------------

        api_key = os.environ.get("GEMINI_API_KEY")

        if not api_key:
            st.error(
                "Gemini API key is not configured. "
                "Please set the GEMINI_API_KEY environment variable."
            )
            st.stop()

        client = load_gemini_client()

        conversation = ""

        for chat in st.session_state.chat_history:
            conversation += f"""
        Previous Question: {chat["question"]}
        Previous Answer: {chat["answer"]}

        """


        prompt = f"""
        You are a document-based knowledge assistant.

        Answer the user's question ONLY using the information
        provided in the document context below.

        Do not use outside knowledge.

        Use the previous conversation only to understand
        follow-up questions and references such as "it", "they",
        "that", or "the previous one".

        If the answer cannot be found in the provided document
        context, say:

        "I couldn't find the answer in the uploaded documents."

        Keep the answer clear and concise.

        PREVIOUS CONVERSATION:
        {conversation}

        DOCUMENT CONTEXT:
        {context}

        CURRENT USER QUESTION:
        {question}
        """


        try:
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt
            )

        except Exception as error:
            st.error(
                "The AI service could not generate a response. "
                "Please try again."
            )
            st.stop()


        # -----------------------------
        # STEP 5: Display Answer
        # -----------------------------

        st.session_state.chat_history.append({
            "question": question,
            "answer": response.text
        })

        st.write("### Conversation History")

        for chat in st.session_state.chat_history:
            st.write(f"**Question:** {chat['question']}")
            st.write(f"**Answer:** {chat['answer']}")


        # -----------------------------
        # STEP 6: Display Sources
        # -----------------------------

        st.write("### Referenced Sources")

        for chunk in retrieved_chunks:

            if chunk["page_number"] is not None:

                st.write(
                    f"**{chunk['source']}** "
                    f"| Page {chunk['page_number']} "
                    f"| Chunk {chunk['chunk_number']} "
                    f"| Semantic Similarity: {chunk['similarity']:.3f} "
                    f"| Retrieval Score: {chunk['retrieval_score']:.3f}"
                )

            else:

                st.write(
                    f"**{chunk['source']}** "
                    f"| Chunk {chunk['chunk_number']} "
                    f"| Semantic Similarity: {chunk['similarity']:.3f} "
                    f"| Retrieval Score: {chunk['retrieval_score']:.3f}"
                )

            st.info(chunk["text"])