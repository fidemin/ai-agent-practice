import asyncio

import websockets


async def hello():
    async with websockets.connect("ws://localhost:8765") as websocket:
        while True:
            user_input = input("User >> ")
            await websocket.send(user_input)

            response = await websocket.recv()
            print(f"Assistant >>: {response}")


if __name__ == "__main__":
    asyncio.run(hello())
