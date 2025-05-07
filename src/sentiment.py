import openai, os

openai.api_key = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = """
You are a retail data analyst. For each line return JSON with:
sentiment (love, like, neutral, dislike, hate) and any product dimension
mentioned (size, flavor, material, shipping, CX, price).
"""

def tag_sentiment(batch: list[str]) -> str:
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(batch)},
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content
