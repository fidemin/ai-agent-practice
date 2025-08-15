import asyncio

import semantic_kernel as sk
from semantic_kernel.connectors.ai.ollama import (
    OllamaChatPromptExecutionSettings,
    OllamaChatCompletion,
)
from semantic_kernel.functions import KernelArguments
from semantic_kernel.prompt_template import PromptTemplateConfig, InputVariable

kernel = sk.Kernel()
chat = OllamaChatCompletion(
    ai_model_id="gpt-oss:20b",  # e.g. `ollama pull gpt-oss:20b`
    host="http://localhost:11434",  # default Ollama endpoint
    service_id="ollama-gpt",  # <-- pick any id; reuse below
)
kernel.add_service(chat)


def get_recommend_prompt_config():
    execution_settings = OllamaChatPromptExecutionSettings(
        service_id="ollama-gpt",
        ai_model_id="gpt-oss:20b",
        max_tokens=5120,
        temperature=0.7,
    )

    prompt = """
    system:
    You have vast knowledge about everything.
    I can recommend anything if subject, genre, format, custom are given.
    
    user:
    Recommend {{$format}} with subject {{$subject}} and genre {{$genre}}.
    It also has customized info: {{$custom}}
    """

    prompt_tmpl_config = PromptTemplateConfig(
        template=prompt,
        name="tldr",
        template_format="semantic-kernel",
        input_variables=[
            InputVariable(
                name="format",
                description="format for recommendation. e.g. movie, music",
                is_required=True,
            ),
            InputVariable(
                name="subject",
                description="subject for recommendation. e.g. boxing",
                is_required=True,
            ),
            InputVariable(
                name="genre",
                description="genre for recommendation. e.g. action",
                is_required=True,
            ),
            InputVariable(
                name="custom",
                description="custom infomation for recommendation.",
                is_required=True,
            ),
        ],
        execution_settings=execution_settings,
    )
    return prompt_tmpl_config


async def main():
    prompt_template_config = get_recommend_prompt_config()
    recommend_function = kernel.add_function(
        prompt_template_config=prompt_template_config,
        function_name="Recommend_Anything",
        plugin_name="Recommendation",
    )

    recommendation = await kernel.invoke(
        recommend_function,
        KernelArguments(
            subject="boxing",
            format="movie",
            genre="drama",
            custom="Should be sad ending",
        ),
    )
    print(recommendation)


if __name__ == "__main__":
    asyncio.run(main())
