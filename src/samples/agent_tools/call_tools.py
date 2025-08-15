# pip install openai

from __future__ import annotations

import datetime as dt
import json
import random
from typing import Dict, Any

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",  # ollama default
    api_key="ollama",  # any non-empty string works
)

MODEL = "gpt-oss:20b"  # try others: "qwen2.5", "phi3", etc.


def get_weather(city: str) -> Dict[str, Any]:
    """Fake weather service for demo."""
    # In real life: call a real API. Here we mock.
    cond = random.choice(["sunny", "cloudy", "rainy", "windy"])
    temp_c = round(random.uniform(20, 30), 1)
    return {
        "city": city,
        "condition": cond,
        "temp_c": temp_c,
        "observed_at": dt.datetime.now().isoformat(timespec="seconds"),
    }


def add(a: float, b: float) -> Dict[str, Any]:
    return {"a": a, "b": b, "sum": a + b}


TOOL_IMPLS = {
    "get_weather": get_weather,
    "add": add,
}

# --- 3) Advertise tools to the model (OpenAI tool schema) ---
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city (mocked demo).",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g., 'Seoul'",
                    },
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers and return the sum.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
    },
]

# --- 4) A helpful system prompt for models that aren't fine-tuned for tools ---
SYSTEM_PROMPT = """\
You are a helpful assistant. If a tool is relevant, ALWAYS call it with JSON arguments.
Return ONLY one tool call at a time. After tool results, write a concise final answer.
If user asks for math, use the `add` tool when appropriate. For weather questions, use `get_weather`.
"""


def chat_once(messages, tools=None, tool_choice="auto"):
    return client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,  # let model decide when to call tools
        temperature=0,
    )


def run_chat(user_prompt: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    while True:
        resp = chat_once(messages, tools=TOOLS, tool_choice="auto")
        choice = resp.choices[0]
        msg = choice.message

        # If the model made a tool call, execute it and feed back the result.
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            # (Many open-source models will put arguments as a JSON string.)
            for call in tool_calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    # Fallback: try to coerce simple cases
                    args = {}

                if name not in TOOL_IMPLS:
                    tool_result = {"error": f"Unknown tool: {name}"}
                else:
                    try:
                        tool_result = TOOL_IMPLS[name](**args)
                    except TypeError as e:
                        tool_result = {"error": f"Bad arguments for {name}: {e}"}
                    except Exception as e:
                        tool_result = {"error": f"Tool {name} failed: {e}"}

                # Append tool result for the model to use
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,  # echo back the id
                        "name": name,
                        "content": json.dumps(tool_result),
                    }
                )

            # Loop continues so the model can produce the final answer using tool results.
            continue

        # No tool calls -> we have the assistant's final answer
        print(msg.content)
        break


if __name__ == "__main__":
    # Try different prompts:
    # 1) Weather (should trigger get_weather then produce final text)
    print("=== Weather example ===")
    run_chat("What's the weather in Seoul right now? Keep it short.")

    # 2) Math (should trigger add)
    print("\n=== Math example ===")
    run_chat("Add 12.5 and 7.25 and explain the result briefly.")

    # 3) No tools
    print("\n=== no tool example ===")
    run_chat("Recommend current movies")
