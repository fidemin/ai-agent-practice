import asyncio

import websockets
from ollama import Client

ollama_client = Client(host="http://localhost:11434")

messages = [
    {"role": "system", "content": "You are too much talker."},
]


async def echo(websocket):
    async for message in websocket:
        print(f"Received message: {message}")
        messages.append({"role": "user", "content": message})
        response = await asyncio.to_thread(
            ollama_client.chat,
            model="gpt-oss:20b",
            messages=messages,
        )

        response_content = response["message"]["content"]
        messages.append(
            {
                "role": "assistant",
                "content": response_content,
            }
        )
        await websocket.send(response_content)
        print(f"Sent message: {response_content}")


async def main():
    async with websockets.serve(
        echo,
        "localhost",
        8765,
        ping_interval=20,
        ping_timeout=60,
        close_timeout=10,
    ):
        print("WebSocket server started at ws://localhost:8765")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
