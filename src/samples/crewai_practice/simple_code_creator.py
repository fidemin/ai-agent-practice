import os

import chromadb
import numpy as np
from chromadb.utils import embedding_functions
from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
model = os.getenv("MODEL", "ollama/gpt-oss:20b")

embedding_model = SentenceTransformer("BAAI/bge-m3")  # multilingual, multi-function


def _embed_fn(texts):
    vecs = embedding_model.encode(
        texts, normalize_embeddings=True
    )  # L2-normalize for better performance
    return [v.tolist() for v in np.atleast_2d(vecs)]


class ChromaEF(embedding_functions.EmbeddingFunction):
    def __call__(self, inputs):
        # list[str] -> list[list[float]]
        return _embed_fn(inputs)


client = chromadb.Client()
collection = client.get_or_create_collection(
    name="agent_memory",
    embedding_function=ChromaEF(),
)

# ---- CrewAI agents/tasks with memory ----
# IMPORTANT: pass a memory_config so CrewAI won't try to init OpenAI embeddings.
memory_config = {
    "provider": "chroma",
    "client": client,  # pass the client so it can reuse the same store
    "collection": collection,  # optionally pass the ready collection
    # Fallbacks in case your CrewAI version expects keys instead:
    "collection_name": "crewai-memory",
    "embedding_fn": _embed_fn,  # some versions use 'embedding_fn' or 'embed_fn'
}


llm = LLM(
    model=model,
    base_url=base_url,
    temperature=0.7,
)

simple_code_programmer = Agent(
    role="simple code programmer",
    goal="create python code about {topic}",
    verbose=True,
    memory=True,
    backstory=(
        "You are python programmer who can only write simple code."
        "The lines of code should be less than 100."
    ),
    allow_delegation=True,
    llm=llm,
    memory_config=memory_config,
)

simple_code_reviewer = Agent(
    role="simple code reviewer",
    goal="review only simple code",
    verbose=True,
    memory=True,
    backstory=("You are simple code reviewer." "You can only review simple code."),
    allow_delegation=False,
    llm=llm,
    memory_config=memory_config,
)

simple_code_creation_task = Task(
    description="You need to create python code less than 100 lines for {topic}.",
    expected_output="python codes less than 100 lines",
    agent=simple_code_reviewer,
)

simple_code_review_task = Task(
    description=(
        "The codes given should be related to {topic}."
        " If the codes are not related to the {topic}, you say 'you are wrong'"
        " You can only review codes less than 100 lines."
        " If the lines of code is over 100, you say 'it sucks'"
    ),
    expected_output="code reviews about {topic}",
    agent=simple_code_reviewer,
    async_execution=False,
    output_file="reviewed_code.md",
)

crew = Crew(
    agents=[simple_code_programmer, simple_code_reviewer],
    tasks=[simple_code_creation_task, simple_code_review_task],
    process=Process.sequential,
    memory=False,
    cache=True,
    max_rpm=100,
    share_crew=True,
)

if __name__ == "__main__":
    result = crew.kickoff(inputs={"topic": "New York City"})
    print(result)
