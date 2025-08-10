from openai import OpenAI


if __name__ == "__main__":
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="notneeded")

    completion = client.chat.completions.create(
        model="gpt-oss:20b",
        messages=[
            {
                "role": "system",
                "content": """
                Follow steps below.
                Step 1. Summarize only sentence in statement xml tag. requires prefix 'Summary: '.
                Step 2. Translate summary in step 1 to Korean. requires prefix 'Translation: '. 
                """,
            },
            {
                "role": "user",
                "content": """
                Hi, I am Fide.
                <statement> This is my favorite language python!! </statement>
                """,
            },
        ],
        temperature=0.7,
    )

    print(completion.choices[0].message.content)
