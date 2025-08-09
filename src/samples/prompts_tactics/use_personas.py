from openai import OpenAI


if __name__ == "__main__":
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="notneeded")

    completion = client.chat.completions.create(
        model="gpt-oss:20b",
        messages=[
            {
                "role": "system",
                "content": "You are junior software engineer.",
            },
            {
                "role": "system",
                "content": "What is your favorite language?",
            },
        ],
        temperature=0.7,
    )

    print(completion.choices[0].message)
