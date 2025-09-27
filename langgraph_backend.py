from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not set in .env file.")

# Initialize model
def initialize_model():
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=GOOGLE_API_KEY
    )
    return model

model = initialize_model()

# Define chat state
class ChatState(dict):
    messages: list
    pdf_content: str

# Chat node
def chat_node(state: ChatState):
    messages = state["messages"]
    pdf_content = state.get("pdf_content", "")

    system_message = SystemMessage(content=f"""
    You are a PDF assistant. Answer ONLY the user's current question using the PDF content.
    DO NOT include previous chat context or explanations.
    Return ONLY the exact answer.
    If the answer is not in the PDF, reply 'Not found'.

    PDF Content: {pdf_content[:3000]}
    """)

    # Use only the last user message
    last_user_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_message = msg
            break

    all_messages = [system_message, last_user_message] if last_user_message else [system_message] + messages
    response = model.invoke(all_messages)
    return {"messages": [AIMessage(content=response.content)]}

# Build workflow graph
checkpoint = InMemorySaver()
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
workflow = graph.compile(checkpointer=checkpoint)
