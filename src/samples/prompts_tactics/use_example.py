from openai import OpenAI


if __name__ == "__main__":
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="notneeded")

    completion = client.chat.completions.create(
        model="gpt-oss:20b",
        messages=[
            {
                "role": "system",
                "content": """
                Answer to my question with humor.
                
                Example:
                    User:
                        What is python?
                    Assistant:
                        Python is computer programming language. Python is pie!!
                """,
            },
            {
                "role": "user",
                "content": """
                What is Java?
                """,
            },
        ],
        temperature=0.7,
    )

    print(completion.choices[0].message.content)
