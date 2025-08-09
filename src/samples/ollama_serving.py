from openai import OpenAI


if __name__ == "__main__":
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="notneeded")

    completion = client.chat.completions.create(
        model="gpt-oss:20b",
        messages=[
            {
                "role": "system",
                "content": "항상 라임을 맞춰서 응답하세요.",
            },
            {
                "user": "system",
                "content": "네 소개를 해줘.",
            },
        ],
        temperature=0.7,
    )

    print(completion.choices[0].message)
