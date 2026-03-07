# langchain_chatbot.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
import os

# Load OpenAI key from environment
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# === Define sample tools ===
def get_weather(city: str = "New York", date: str = "tomorrow") -> str:
    return f"The weather in {city} on {date} will be sunny."

def book_meeting(person: str = "Alice", date: str = "next Monday") -> str:
    return f"Meeting booked with {person} on {date}."

# === Wrap them as LangChain Tools ===
tools = [
    Tool(
        name="get_weather",
        func=lambda q: get_weather(),
        description="Gets the weather forecast for a city on a given date"
    ),
    Tool(
        name="book_meeting",
        func=lambda q: book_meeting(),
        description="Books a meeting with a person on a specific date"
    )
]

# === LLM and Memory ===
llm = ChatOpenAI(model_name="gpt-4", temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history")

# === Agent setup ===
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# === Input schema ===
class ChatRequest(BaseModel):
    user_input: str

# === Endpoint ===
@app.post("/chat")
async def chat(chat: ChatRequest):
    response = agent.run(chat.user_input)
    return {"response": response}