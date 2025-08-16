import asyncio

import semantic_kernel as sk
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.ollama import (
    OllamaChatCompletion,
    OllamaChatPromptExecutionSettings,
)
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelArguments
from semantic_kernel.prompt_template import PromptTemplateConfig, InputVariable

from src.samples.agent_tools.plugins.tmdb import TMDbService

kernel = sk.Kernel()
SERVICE_ID = "ollama-gpt"
MODEL_ID = "gpt-oss:20b"
chat = OllamaChatCompletion(
    ai_model_id=MODEL_ID,
    host="http://localhost:11434",  # default Ollama endpoint
    service_id=SERVICE_ID,  # <-- pick any id; reuse below
)
kernel.add_service(chat)

chatbot_execution_settings = OllamaChatPromptExecutionSettings(
    service_id=SERVICE_ID,
    ai_model_id=MODEL_ID,
    max_tokens=5120,
    temperature=0.7,
    top_p=0.8,
    tool_choice="auto",
    function_choice_behavior=FunctionChoiceBehavior.Auto(
        filters={
            "excluded_plugins": ["ChatBot"]
        }  # exclude ChatBot plugin because chatbot do not need to use this plugin as tools
    ),
)

prompt_template_config = PromptTemplateConfig(
    template="{{$user_input}}",
    execution_settings=chatbot_execution_settings,
    input_variables=[InputVariable(name="user_input", description="User's request")],
)

history = ChatHistory()

history.add_system_message("You recommend movies and TV shows")
history.add_user_message("Hi, who are you?")
history.add_assistant_message(
    "I am movie recommend bot. I want to try to find what user wants"
)


kernel.add_function(
    plugin_name="ChatBot",
    function_name="chat",
    prompt_template_config=prompt_template_config,
)

plugin = kernel.add_plugin(TMDbService, plugin_name="TMDbService")


async def chat() -> bool:
    try:
        user_input = input("User:> ")
    except KeyboardInterrupt:
        print("\n\nExiting chat...")
        return False
    except EOFError:
        print("\n\nExiting chat...")
        return False

    if user_input == "exit":
        print("\n\nExiting chat...")
        return False
    arguments = KernelArguments(
        user_input=user_input,
        history=("\n").join([f"{msg.role}: {msg.content}" for msg in history]),
    )
    result = await kernel.invoke(
        arguments=arguments,
        plugin_name="ChatBot",
        function_name="chat",
        chat_history=history,
    )
    print(f"GPT Agent:> {result}")
    return True


async def main() -> None:
    chatting = True
    while chatting:
        chatting = await chat()


if __name__ == "__main__":
    asyncio.run(main())
