import streamlit as st
import pdfplumber
import uuid
from langgraph_backend import workflow
from langchain_core.messages import HumanMessage

# ---------------------------
# Session State Setup
# ---------------------------
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = {}
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = str(uuid.uuid4())
if st.session_state['thread_id'] not in st.session_state['chat_threads']:
    st.session_state['chat_threads'][st.session_state['thread_id']] = []
if 'pdf_content' not in st.session_state:
    st.session_state['pdf_content'] = ""

# ---------------------------
# Helper Functions
# ---------------------------
def generate_thread_id():
    return str(uuid.uuid4())

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'][thread_id] = []

def extract_pdf_text(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def clean_response(response_text, user_input):
    if user_input.lower() in response_text.lower():
        response_text = response_text.replace(user_input, "").strip()
    prefixes = ["the name is", "the mobile number is", "mobile number:", "phone number:", 
                "the answer is", "based on the pdf", "according to the pdf", "from the pdf"]
    for prefix in prefixes:
        if response_text.lower().startswith(prefix):
            response_text = response_text[len(prefix):].strip().lstrip(' :,-')
    return response_text.strip()

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("Message History")
if st.sidebar.button("➕ New Chat"):
    st.session_state['thread_id'] = generate_thread_id()
    add_thread(st.session_state['thread_id'])
    st.session_state['pdf_content'] = ""
    st.rerun()

st.sidebar.header("My Conversations")
for tid in list(st.session_state['chat_threads'].keys()):
    if st.sidebar.button(f"Chat {tid[:8]}...", key=tid):
        st.session_state['thread_id'] = tid
        st.rerun()
st.sidebar.markdown(f"**Current:** {st.session_state['thread_id'][:8]}...")

# ---------------------------
# Main UI
# ---------------------------
st.title("📄 PDF Chatbot (LangGraph + Google Generative AI)")

# PDF uploader
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    st.session_state['pdf_content'] = extract_pdf_text(uploaded_file)
    st.success("PDF uploaded and processed successfully!")

# Display previous messages
current_thread = st.session_state['chat_threads'][st.session_state['thread_id']]
for message in current_thread:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# User input
user_input = st.chat_input("Ask something about the PDF…")
if user_input and user_input.strip():
    current_thread.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Stream response
    response_placeholder = st.empty()
    full_response = ""
    try:
        for event in workflow.stream(
            {"messages":[HumanMessage(content=user_input)], "pdf_content": st.session_state['pdf_content']},
            config={"configurable":{"thread_id": st.session_state['thread_id']}},
            stream_mode="values"
        ):
            if "messages" in event and event["messages"]:
                for message in event["messages"]:
                    if hasattr(message, "content") and message.content:
                        clean_text = clean_response(message.content, user_input)
                        if clean_text and clean_text != full_response:
                            full_response = clean_text
                            response_placeholder.markdown(full_response)
        if full_response:
            current_thread.append({"role": "assistant", "content": full_response})
        else:
            response_placeholder.markdown("Not found")
            current_thread.append({"role": "assistant", "content": "Not found"})
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
