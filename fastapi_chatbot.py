from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Dict, Callable, Any
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


app = FastAPI()

# === Function Registry ===
def get_weather(city: str, date: str) -> str:
    return f"The weather in {city} on {date} will be sunny."

def book_meeting(person: str, date: str) -> str:
    return f"Meeting booked with {person} on {date}."

function_registry: Dict[str, Callable[..., str]] = {
    "get_weather": get_weather,
    "book_meeting": book_meeting
}

# === Request Schema ===
class ChatRequest(BaseModel):
    user_input: str
    history: list[Dict[str, str]] = []

# === Utility Functions ===
def extract_function_call(prompt: str) -> Dict[str, Any]:
    """ Very simple intent/function detection logic (placeholder for actual NLU) """
    if "weather" in prompt:
        return {"function": "get_weather", "args": {"city": "New York", "date": "tomorrow"}}
    if "meeting" in prompt:
        return {"function": "book_meeting", "args": {"person": "Alice", "date": "next Monday"}}
    return {"function": None, "args": {}}

# === Chat Endpoint ===
@app.post("/chat")
async def chat_endpoint(chat: ChatRequest):
    history = chat.history
    user_input = chat.user_input

    # Detect function call
    call = extract_function_call(user_input)
    func_name, args = call["function"], call["args"]

    if func_name and func_name in function_registry:
        result = function_registry[func_name](**args)
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": result})
        return {"response": result, "history": history}

    # Fall back to LLM response
    messages = [{"role": "system", "content": "You are a helpful assistant."}] + history + [
        {"role": "user", "content": user_input}
    ]

    completion = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
    response = completion["choices"][0]["message"]["content"]
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": response})
    return {"response": response, "history": history}
