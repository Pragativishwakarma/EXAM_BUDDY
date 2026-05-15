import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pypdf import PdfReader
from dotenv import load_dotenv
import os


st.set_page_config(
    page_title="Study Buddy RAG",
    page_icon="📚",
    layout="wide"
)

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("Gemini API key not found!")
    st.stop()


try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        "gemini-2.0-flash"
    )
except Exception as e:
    st.error(f"Gemini setup error: {e}")
    st.stop()



@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()


st.markdown(
    """
    <style>
    .main {
        background-color: #0E1117;
    }

    .stTextInput > div > div > input {
        border-radius: 10px;
    }

    .stButton button {
        border-radius: 10px;
        width: 100%;
        font-size: 16px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)



st.title("📚 Study Buddy RAG")
st.write("Upload PDFs and ask questions from your notes.")



uploaded_files = st.file_uploader(
    "Upload PDF Files",
    type=["pdf"],
    accept_multiple_files=True
)


def extract_text_from_pdfs(files):

    full_text = ""

    for file in files:

        pdf_reader = PdfReader(file)

        for page in pdf_reader.pages:

            text = page.extract_text()

            if text:
                full_text += text + "\n"

    return full_text


def chunk_text(text, chunk_size=500):

    chunks = []

    for i in range(0, len(text), chunk_size):

        chunks.append(text[i:i + chunk_size])

    return chunks


def create_vector_store(chunks):

    embeddings = embedding_model.encode(chunks)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    return index



def retrieve_relevant_chunks(query, chunks, index, top_k=3):

    query_embedding = embedding_model.encode([query])

    distances, indices = index.search(
        np.array(query_embedding),
        top_k
    )

    retrieved_chunks = [chunks[i] for i in indices[0]]

    return retrieved_chunks



if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "index" not in st.session_state:
    st.session_state.index = None



if uploaded_files:

    with st.spinner("Processing PDFs..."):

        raw_text = extract_text_from_pdfs(uploaded_files)

        chunks = chunk_text(raw_text)

        index = create_vector_store(chunks)

        st.session_state.chunks = chunks
        st.session_state.index = index

    st.success("PDFs processed successfully!")


question = st.text_input("Ask a question from your notes")


if st.button("Get Answer"):

    if not uploaded_files:
        st.warning("Please upload PDFs first.")

    elif not question:
        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching and generating answer..."):

            retrieved_chunks = retrieve_relevant_chunks(
                question,
                st.session_state.chunks,
                st.session_state.index
            )

            context = "\n\n".join(retrieved_chunks)

            prompt = f"""
            You are a helpful study assistant.

            Use the following context to answer the question.

            Context:
            {context}

            Question:
            {question}

            Answer in a simple and detailed manner.
            """

            try:

                response = model.generate_content(prompt)

                st.subheader("📖 Answer")
                st.write(response.text)

                with st.expander("📚 Retrieved Context"):
                    st.write(context)

            except Exception as e:
                st.error(f"Error generating response: {e}")