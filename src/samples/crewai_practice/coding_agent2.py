import os
from textwrap import dedent

from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()
    base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model = os.getenv("MODEL", "ollama/gpt-oss:20b")

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

    from crewai.agents.agent_builder.utilities.base_token_process import TokenProcess

    TokenProcess()

    # Ollama manager not working
    # manager_llm = LLM(
    #     model=model,
    #     base_url=base_url,
    #     api_key="NA",  # litellm requires something
    #     temperature=0.1,  # keep manager deterministic
    #     timeout=300,  # give local model time
    #     max_retries=3,
    #     stream=False,  # avoid stream parsing issues for manager
    #     max_tokens=5120,
    #     # Expand context and nudge decoding on Ollama
    #     extra_body={
    #         "options": {
    #             "num_ctx": 8192,  # bigger context helps with hierarchy prompts
    #             "repeat_penalty": 1.05,
    #         }
    #     },
    # )

    # only Actual openai call is working..
    # manager_llm = ChatOpenAI(
    #     model="gpt-4.1",
    #     temperature=0,
    # )

    crew = Crew(
        agents=[
            senior_engineering_agent,
            qa_engineer_agent,
            chief_qa_engineer_agent,
        ],
        tasks=[code_task, qa_task, evaluate_task],
        verbose=True,
        process=Process.hierarchical,
        manager_llm=manager_llm,
    )

    result = crew.kickoff()

    print("----------------------")
    print(result)
