### M3 MODEL: Prompt-Engineered ChatGPT
import openai

openai.api_key = "YOUR_API_KEY"

def classify_fallacy(text):
    prompt = f"""
You are a logical fallacy expert. Given the following argument, identify the logical fallacy it commits. Choose from: Ad Hominem, Strawman, Slippery Slope, Red Herring, Appeal to Emotion, or None.

Argument: "{text}"

Answer with only the name of the fallacy.
"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response["choices"][0]["message"]["content"].strip()
 