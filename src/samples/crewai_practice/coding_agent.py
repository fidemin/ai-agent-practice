import os
from textwrap import dedent

import chromadb
from chromadb.utils import embedding_functions
from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
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

    print("Welcome to game crew.")
    print("---------------------")
    query = input("What kind of game do you want to create?")

    senior_engineering_agent = Agent(
        role="Senior software engineer",
        goal="Generate software based on requirements.",
        backstory=dedent(
            """
            You are very talented and tech-driven senior software engineer.
            You are specialized on python programming language.
            You always do your best to create perfect code.
            """
        ),
        allow_delegation=False,
        verbose=True,
        llm=llm,
    )

    qa_engineer_agent = Agent(
        role="Software QA engineer",
        goal="Find errors in code given and fix it.",
        backstory=dedent(
            """
            You are QA engineer who is specialized in finding errors in codes.
            You check missing import, declaring variable, syntax error, etc.
            You also check secure vulnerability and logical error.
            """
        ),
        allow_delegation=False,
        verbose=True,
        llm=llm,
    )

    chief_qa_engineer_agent = Agent(
        role="Chief software QA engineer",
        goal="Check the code works as it intends.",
        backstory=dedent(
            """
            You are Chief QA software engineer.
            You are responsible for the code works as it intends.
            You should check errors in code and guarantee the quality of code
            """
        ),
        allow_delegation=True,
        verbose=True,
        llm=llm,
    )

    code_task = Task(
        description=f"""
        Create game based on python code based on following instructions in XML instruction tag.
        
        <instruction>
        {query}
        </instruction>
        """,
        expected_output="Final result should be only python code without other contents.",
        agent=senior_engineering_agent,
    )

    qa_task = Task(
        description=f"""
        Find errors in given code."
        You should check logical error, syntax error, missing imports, variables, unmatched parenthesis, security vulnerability.
        
        The context is code is following instruction.
        <instruction>
        {query}
        </instruction>
        """,
        expected_output="Make a list of found problems",
        agent=qa_engineer_agent,
    )

    evaluate_task = Task(
        description=f"""
        Help creating python game based on following instruction in instruction tag.
        <instruction>
        {query}
        </instruction>
        
        Review the code works as it intends and fix code based on problems given.
        """,
        expected_output="Final result should be only python code without other contents.",
        agent=chief_qa_engineer_agent,
    )

    crew = Crew(
        agents=[
            senior_engineering_agent,
            qa_engineer_agent,
            chief_qa_engineer_agent,
        ],
        tasks=[code_task, qa_task, evaluate_task],
        verbose=True,
        process=Process.sequential,
    )

    result = crew.kickoff()

    print("----------------------")
    print(result)
