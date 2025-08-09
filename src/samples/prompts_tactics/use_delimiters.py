from openai import OpenAI


if __name__ == "__main__":
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="notneeded")

    completion = client.chat.completions.create(
        model="gpt-oss:20b",
        messages=[
            {
                "role": "system",
                "content": "Compare two sentence in <statement> xml tag. Summarize it",
            },
            {
                "role": "user",
                "content": """
                <statement> This is my favorite language python!! </statement>
                <statement> This is my worst language C!! </statement>
                """,
            },
        ],
        temperature=0.7,
    )

    print(completion.choices[0].message)
