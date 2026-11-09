import re

import streamlit as st
from pypdf import PdfReader

# --- CONSTANTS ---
MODELS = ["gpt-4o-mini", "gpt-4o"]
SUMMARY_CHARS = 10000   # text sent for summary / ELI5 / glossary
CHAT_CHARS = 8000       # retrieved text sent for each chat question
CHUNK_CHARS = 1200      # size of pieces used for keyword search
MIN_TEXT_CHARS = 200    # below this, the PDF is treated as scanned
SIM_EXCERPT_CHARS = 600

STOPWORDS = {
    "this", "that", "with", "from", "have", "were", "they", "their", "there",
    "which", "these", "those", "into", "also", "than", "then", "when", "what",
    "about", "been", "will", "would", "could", "should", "using", "used", "paper",
    "study", "does", "data", "such", "over", "more", "some", "other", "only",
}

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="PDF ELI5 & Research Studio",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR STYLING ---
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        color: #1E3A8A;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #F3F4F6;
        border-radius: 4px;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563EB !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=80)
    st.title("Studio Settings")
    st.markdown("Configure your AI backend and preferences.")

    api_key = st.text_input(
        "OpenAI API Key (Optional)", type="password",
        help="Leave blank to use simulation mode (no AI, uses only the PDF text).",
    )
    model_choice = st.selectbox("Select AI Model", MODELS)
    use_api = bool(api_key and api_key.startswith("sk-"))

    if use_api:
        st.success(f"Mode: OpenAI ({model_choice})")
    else:
        st.info("Mode: Simulation (no AI). Add an API key for AI answers.")

    st.markdown("---")
    st.markdown("### 💡 Project Guide")
    st.markdown("""
    1. **Upload** any academic paper or technical PDF.
    2. **Explore** the Executive Summary, ELI5 breakdown, and Jargon glossary.
    3. **Chat** directly with the document using the Q&A tab.
    """)
    st.markdown("---")
    st.caption("College Project Edition • Built with Streamlit")


# --- PDF EXTRACTION ---
def extract_pages(uploaded_file):
    """Return a list of (page_number, text) tuples, or None on error."""
    try:
        uploaded_file.seek(0)
        reader = PdfReader(uploaded_file)
        return [(i + 1, (page.extract_text() or "").strip())
                for i, page in enumerate(reader.pages)]
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None


def ocr_pages(uploaded_file):
    """Optional OCR for scanned PDFs. Needs pytesseract, pdf2image and system Tesseract/Poppler."""
    try:
        import pytesseract
        from pdf2image import convert_from_bytes
    except ImportError:
        return None
    try:
        uploaded_file.seek(0)
        images = convert_from_bytes(uploaded_file.read(), dpi=200)
        return [(i + 1, pytesseract.image_to_string(img).strip())
                for i, img in enumerate(images)]
    except Exception as e:
        st.warning(f"OCR failed: {e}")
        return None


def full_text(pages):
    return "".join(f"\n--- Page {n} ---\n{t}" for n, t in pages if t)


def make_chunks(pages):
    """Split each page into pieces of about CHUNK_CHARS, keeping the page number."""
    chunks = []
    for n, text in pages:
        for start in range(0, len(text), CHUNK_CHARS):
            piece = text[start:start + CHUNK_CHARS].strip()
            if piece:
                chunks.append((n, piece))
    return chunks


# --- RETRIEVAL (keyword based, no AI needed) ---
def keywords(text):
    return {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in STOPWORDS}


def relevant_chunks(chunks, query, max_chars):
    """Pick the chunks that share the most words with the query, within a size limit.
    Returned in document order."""
    q = keywords(query)
    scored = sorted(
        ((len(q & keywords(piece)), idx, n, piece) for idx, (n, piece) in enumerate(chunks)),
        key=lambda s: (-s[0], s[1]),
    )
    picked, total = [], 0
    for overlap, idx, n, piece in scored:
        if overlap == 0:
            break  # nothing else matches the query
        if total + len(piece) > max_chars:
            continue
        picked.append((idx, n, piece))
        total += len(piece)
    picked.sort()
    return picked


# --- OPENAI ---
def call_openai(api_key, model, system_prompt, user_prompt):
    """Return the model's text, or None if the call failed."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.warning(f"API call failed ({e}). Showing simulation output for this section.")
        return None


SYSTEM_PROMPT = "You are an expert academic research assistant specializing in simplifying complex papers."

SECTION_INSTRUCTIONS = {
    "executive": "Write an executive summary of the document below: the problem, the method, the main findings and the limitations. Use markdown headings and bullet points.",
    "eli5": "Explain the document below like I'm 5 years old. Use simple everyday analogies and short paragraphs. Use markdown.",
    "glossary": "List the important technical terms in the document below as a markdown table with columns 'Term' and 'Simplified Definition'.",
}


# --- SIMULATION OUTPUT (built from the PDF itself, no AI) ---
def sim_executive(text):
    excerpt = text[:1200].strip()
    return ("### Opening text from the document\n\n"
            "*Simulation mode: this is the start of the paper, not an AI summary. "
            "Add an OpenAI key in the sidebar for a real summary.*\n\n"
            f"> {excerpt.replace(chr(10), ' ')}…")


def sim_eli5():
    return ("### ELI5 (Explain Like I'm 5)\n\n"
            "*Simulation mode: analogies need an AI model.* "
            "Add an OpenAI key in the sidebar and this tab will explain the paper in simple words.")


def sim_glossary(text):
    # Words written in capitals (acronyms) that appear often are likely key terms.
    counts = {}
    for term in re.findall(r"\b[A-Z]{2,}[A-Za-z0-9-]*\b", text):
        counts[term] = counts.get(term, 0) + 1
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:15]
    if not top:
        return "No acronyms or capitalised terms were found. Add an OpenAI key for a full glossary."
    rows = "\n".join(f"| **{t}** | {c} | *Needs AI for a definition* |" for t, c in top)
    return ("*Simulation mode: these are the most frequent capitalised terms in the PDF. "
            "Add an OpenAI key for definitions.*\n\n"
            "| Term | Occurrences | Simplified Definition |\n|---|---|---|\n" + rows)


def build_analysis(text, model, api_key):
    """Return {'executive', 'eli5', 'glossary'} markdown strings.
    Uses OpenAI when a key is given; each section falls back to simulation on failure."""
    source = text[:SUMMARY_CHARS]
    simulated = {
        "executive": sim_executive(text),
        "eli5": sim_eli5(),
        "glossary": sim_glossary(text),
    }
    if not api_key:
        return simulated
    result = {}
    for section, instruction in SECTION_INSTRUCTIONS.items():
        out = call_openai(api_key, model, SYSTEM_PROMPT, f"{instruction}\n\n{source}")
        result[section] = out if out else simulated[section]
    return result


def answer_question(query, chunks, api_key, model):
    picked = relevant_chunks(chunks, query, CHAT_CHARS if api_key else 3 * CHUNK_CHARS)
    if not picked:
        if not api_key:
            return "I could not find any passage in this document that matches your question. Try using words from the paper."
        # With AI, still answer from the start of the document (the AI can say if the answer is missing).
        picked = [(i, n, piece) for i, (n, piece) in enumerate(chunks[:CHAT_CHARS // CHUNK_CHARS])]

    if api_key:
        context = "\n\n".join(f"[Page {n}]\n{piece}" for _, n, piece in picked)
        user_prompt = (
            "Based on the following document context, answer the user's question clearly. "
            "If the context does not contain the answer, say so.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}"
        )
        out = call_openai(api_key, model, SYSTEM_PROMPT, user_prompt)
        if out:
            return out

    # Simulation: show the most relevant passages with their page numbers.
    lines = [f"Here are the passages from the document most related to **'{query}'**:\n"]
    for _, n, piece in picked[:3]:
        snippet = piece[:SIM_EXCERPT_CHARS].replace("\n", " ")
        lines.append(f"> {snippet}…  \n> *(page {n})*\n")
    lines.append("*Simulation mode: this is text from the PDF, not an AI answer. "
                 "Add an OpenAI key in the sidebar for a real answer.*")
    return "\n".join(lines)


# --- MAIN INTERFACE ---
st.markdown('<p class="main-header">📚 AI Research Paper & PDF "ELI5" Studio</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload dense technical papers, extract insights instantly, and break down complex concepts with AI.</p>', unsafe_allow_html=True)
st.markdown("---")

uploaded_file = st.file_uploader("Upload your research paper or technical PDF", type=["pdf"])

if uploaded_file is None:
    st.info("👈 Please upload a PDF file using the file uploader above to begin your analysis session.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📋 Executive Summaries")
        st.write("Instantly distill 20-page papers into structured bullet points and key takeaways.")
    with col2:
        st.markdown("### 🧸 ELI5 Mode")
        st.write("Understand intricate algorithms and theories using intuitive, everyday analogies.")
    with col3:
        st.markdown("### 💬 Document Chat")
        st.write("Interactively query specific sections, formulas, or results without manual skimming.")
    st.stop()

with st.spinner("Extracting text..."):
    pages = extract_pages(uploaded_file)
if pages is None:
    st.stop()

if sum(len(t) for _, t in pages) < MIN_TEXT_CHARS:
    with st.spinner("Little or no text found. Trying OCR (this can be slow)..."):
        ocr = ocr_pages(uploaded_file)
    if ocr and sum(len(t) for _, t in ocr) >= MIN_TEXT_CHARS:
        pages = ocr
        st.info("The text was read with OCR, so some words may be wrong.")
    else:
        st.error(
            "This PDF has no readable text layer, so it is probably a scanned image. "
            "Upload a text-based PDF, or install OCR support as described in the README."
        )
        st.stop()

text = full_text(pages)
chunks = make_chunks(pages)
doc_key = (uploaded_file.name, uploaded_file.size)

# A new document resets the chat and the analysis.
if st.session_state.get("doc_key") != doc_key:
    st.session_state.doc_key = doc_key
    st.session_state.messages = []
    st.session_state.pop("analysis", None)

st.success(f"Successfully loaded **{uploaded_file.name}** (~{len(text)} characters extracted).")

analysis_key = (doc_key, model_choice, api_key if use_api else None)
if st.session_state.get("analysis_key") != analysis_key:
    with st.spinner("Analyzing document..."):
        st.session_state.analysis = build_analysis(text, model_choice, api_key if use_api else None)
    st.session_state.analysis_key = analysis_key
analysis = st.session_state.analysis

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Executive Summary",
    "🧸 ELI5 & Analogies",
    "📖 Jargon Glossary",
    "💬 Chat with Paper",
])

with tab1:
    st.subheader("Executive Academic Summary")
    st.markdown(analysis["executive"])

with tab2:
    st.subheader("Simplified Explanation & Real-world Analogies")
    st.markdown(analysis["eli5"])

with tab3:
    st.subheader("Key Jargon & Terminology Glossary")
    st.markdown(analysis["glossary"])

with tab4:
    st.subheader("Interactive Q&A: Ask Anything About This Paper")
    st.markdown("Ask specific questions regarding methodology, datasets, or conclusions.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("e.g., What datasets were used in this study?"):
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner("Searching the document..."):
            response = answer_question(query, chunks, api_key if use_api else None, model_choice)

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})