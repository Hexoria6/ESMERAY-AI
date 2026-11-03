
import streamlit as st
import requests
import io
import time
import random

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="ESMERAY AI",
    page_icon="✦",
    layout="wide"
)

GROQ_MODEL = "openai/gpt-oss-20b"

GROQ_API_URL = (
    "https://api.groq.com/openai/v1/chat/completions"
)


# ==========================================
# DESIGN - LAVENDER AND PURPLE
# ==========================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg, #f3edff, #e6dcff, #f7f2ff
    );
    color: #211536;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1050px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3, p, label {
    color: #211536;
}

/* MAIN TITLE */

.nova-title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    color: #58358c;
    margin-bottom: 5px;
    letter-spacing: 2px;
}

.nova-subtitle {
    text-align: center;
    color: #765a9e;
    font-size: 16px;
    margin-bottom: 30px;
}

/* WELCOME SECTION */

.welcome {
    text-align: center;
    padding: 32px 15px;
    border: 1px solid #c8b4ef;
    border-radius: 22px;
    background: linear-gradient(
        135deg, #e5d7ff, #d5c2fa
    );
    margin-bottom: 25px;
}

.welcome h2 {
    font-size: 30px;
    color: #45236f;
    margin-bottom: 8px;
}

.welcome p {
    color: #62468a;
}

/* CHAT MESSAGES */

div[data-testid="stChatMessage"] {
    border: 1px solid #cbb9ed;
    border-radius: 16px;
    padding: 14px;
    margin-bottom: 14px;
    background: #ffffff;
}

/* FORCE CHAT TEXT TO BLACK */

div[data-testid="stChatMessage"],
div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] li,
div[data-testid="stChatMessage"] span,
div[data-testid="stChatMessage"] strong,
div[data-testid="stChatMessage"] em,
div[data-testid="stChatMessage"] h1,
div[data-testid="stChatMessage"] h2,
div[data-testid="stChatMessage"] h3,
div[data-testid="stChatMessage"] code {
    color: #171020 !important;
}

/* USER MESSAGE */

div[data-testid="stChatMessage"]:has(
    div[data-testid="chatAvatarIcon-user"]
) {
    background: #e6dcff;
    border: 1px solid #bca4e9;
}

div[data-testid="stChatMessage"]:has(
    div[data-testid="chatAvatarIcon-user"]
) p {
    color: #171020 !important;
}

/* BUTTONS */

.stButton > button {
    width: 100%;
    min-height: 48px;
    border-radius: 12px;
    border: 1px solid #a88bd7;
    background: #d8c6fa;
    color: #35204f;
    transition: 0.2s;
}

.stButton > button:hover {
    background: #b99ce9;
    border-color: #8057c5;
    color: #211536;
}

/* CHAT INPUT */

.stChatInput textarea {
    color: #171020 !important;
    background: #ffffff !important;
}

.stChatInput textarea::placeholder {
    color: #89799e !important;
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: #e5d9f8;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #302044 !important;
}

/* FILE UPLOADER */

div[data-testid="stFileUploader"] {
    border: 1px dashed #9c7acb;
    border-radius: 12px;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# SESSION STATE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents" not in st.session_state:
    st.session_state.documents = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# ==========================================
# DOCUMENT PROCESSING
# ==========================================

def extract_text(uploaded_file):

    try:

        name = uploaded_file.name.lower()

        if name.endswith(".pdf"):

            reader = PdfReader(
                io.BytesIO(uploaded_file.getvalue())
            )

            return "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )

        if name.endswith(".txt"):

            return uploaded_file.getvalue().decode(
                "utf-8",
                errors="replace"
            )

    except Exception as error:

        st.warning(
            f"Could not read {uploaded_file.name}: {error}"
        )

    return ""


def split_into_chunks(
    text,
    chunk_size=900,
    overlap=150
):

    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(text[start:end])

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def retrieve_context(question):

    chunks = st.session_state.documents

    if not chunks:
        return ""

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000
        )

        matrix = vectorizer.fit_transform(
            chunks + [question]
        )

        scores = cosine_similarity(
            matrix[-1],
            matrix[:-1]
        ).flatten()

        best_indices = scores.argsort()[-4:][::-1]

        selected = [
            chunks[i]
            for i in best_indices
            if scores[i] > 0
        ]

        return "\n\n".join(selected)

    except Exception:

        return ""


# ==========================================
# GROQ API REQUEST WITH RETRIES
# ==========================================

def call_model(messages, api_key):

    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1500
    }

    for attempt in range(3):

        try:

            response = requests.post(
                GROQ_API_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                },
                json=payload,
                timeout=60
            )

        except requests.exceptions.Timeout:

            if attempt < 2:

                time.sleep(2 ** attempt)
                continue

            return None, "The request timed out."

        except requests.exceptions.RequestException:

            if attempt < 2:

                time.sleep(2 ** attempt)
                continue

            return None, "Could not connect to Groq AI."

        if response.status_code == 200:

            try:

                data = response.json()

                answer = data["choices"][0]["message"]["content"]

                if answer:

                    return answer.strip(), None

                return None, "The model returned an empty answer."

            except (ValueError, KeyError, IndexError, TypeError):

                return None, "Unexpected AI response."

        if response.status_code in (
            408, 429, 500, 502, 503, 504
        ):

            if attempt < 2:

                delay = (2 ** attempt) + random.uniform(0, 1)

                time.sleep(delay)
                continue

        try:

            detail = response.json().get(
                "error", {}
            ).get(
                "message", "Unknown API error"
            )

        except Exception:

            detail = "Unexpected API response."

        return None, f"Groq API error {response.status_code}: {detail}"

    return None, "Please try again later."


# ==========================================
# ESMERAY AI ASSISTANT - GROQ
# ==========================================

def ask_esmeray(question):

    try:

        api_key = st.secrets["GROQ_API_KEY"]

        context = retrieve_context(question)

        if context:

            prompt = f"""
Use these document excerpts when relevant.
If the answer is not present in the documents,
clearly say so. Do not invent document facts.

DOCUMENT EXCERPTS:

{context}

USER QUESTION:

{question}
"""

        else:

            prompt = question

        system_instruction = """
You are ESMERAY, a helpful, friendly and intelligent AI assistant.

Your responsibilities:

1. Give clear and accurate answers.
2. Explain difficult topics in simple language.
3. Help with learning, writing and brainstorming.
4. Be friendly and respectful.
5. When documents are supplied, prioritize their contents.
6. Never pretend to know information that was not provided.
7. If a question is about uploaded documents,
   distinguish document information from general knowledge.
"""

        messages = [
            {
                "role": "system",
                "content": system_instruction
            }
        ]

        # Add previous conversation history.

        previous_messages = st.session_state.messages[:-1]

        for message in previous_messages[-10:]:

            messages.append({
                "role": (
                    "user"
                    if message["role"] == "user"
                    else "assistant"
                ),
                "content": message["content"]
            })

        # Add current question.

        messages.append({
            "role": "user",
            "content": prompt
        })

        answer, error = call_model(
            messages,
            api_key
        )

        if answer:
            return answer

        return error or "ESMERAY could not generate an answer."

    except KeyError:

        return (
            "GROQ_API_KEY is missing. "
            "Please check Streamlit Secrets."
        )

    except Exception:

        return (
            "Something went wrong. "
            "Please try asking again."
        )


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("## ✦ ESMERAY AI")

    st.caption("Your intelligent assistant")

    st.divider()

    st.markdown("### 📁 Your documents")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:

        if st.button("Process uploaded documents"):

            all_chunks = []

            for uploaded_file in uploaded_files:

                extracted = extract_text(uploaded_file)

                all_chunks.extend(
                    split_into_chunks(extracted)
                )

            st.session_state.documents = all_chunks

            st.success(
                f"Processed {len(uploaded_files)} file(s). "
                f"Created {len(all_chunks)} text chunks."
            )

    if st.session_state.documents:

        st.caption(
            f"{len(st.session_state.documents)} document chunks available"
        )

    st.divider()

    if st.button("＋ New conversation"):

        st.session_state.messages = []

        st.session_state.pending_prompt = None

        st.rerun()

    if st.button("Clear documents"):

        st.session_state.documents = []

        st.success("Documents cleared.")


# ==========================================
# MAIN HOME SCREEN
# ==========================================

st.markdown(
    '<div class="nova-title">✦ ESMERAY AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="nova-subtitle">'
    'Your intelligent AI assistant'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================
# WELCOME SCREEN
# ==========================================

if not st.session_state.messages:

    st.markdown("""
    <div class="welcome">
        <h2>Hello Nakshatra! ✨</h2>
        <p>
            Welcome to ESMERAY AI.
            What would you like to explore today?
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 💡 Suggested questions")

    suggestions = [
        "Explain a difficult topic in simple words",
        "Summarize my uploaded PDF",
        "Answer questions from my documents",
        "Help me brainstorm new ideas"
    ]

    columns = st.columns(2)

    for index, suggestion in enumerate(suggestions):

        with columns[index % 2]:

            if st.button(
                suggestion,
                key=f"suggestion_{index}"
            ):

                st.session_state.pending_prompt = suggestion

                st.rerun()


# ==========================================
# DISPLAY CHAT HISTORY
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ==========================================
# CHAT INPUT
# ==========================================

typed_question = st.chat_input(
    "Ask ESMERAY anything..."
)

question = (
    st.session_state.pending_prompt
    or typed_question
)

if question:

    st.session_state.pending_prompt = None

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("ESMERAY is thinking..."):

            answer = ask_esmeray(question)

        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
