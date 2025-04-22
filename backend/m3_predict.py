from openai import OpenAI
from dotenv import load_dotenv
import os


load_dotenv()

# Get the API key
api_key = os.getenv("AI_API_KEY")
client = OpenAI(api_key=api_key)


def get_fallacy_response(text_comment):

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": "You are a logic expert. For each statement, choose exactly one logical fallacy from the list below that best matches the reasoning error. If there is no fallacy, reply with \"none\".\n\nChoose from:\n- False Causality\n- Fallacy of Logic\n- Circular Reasoning\n- Equivocation\n- Faulty Generalization\n- Fallacy of Relevance\n- False Dilemma\n- Ad Populum\n- Fallacy of Extension\n- Fallacy of Credibility\n- Appeal to Emotion\n- Intentional Fallacy\n- Ad Hominem\n- none\n\nExamples:\n"
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Senator Randall isn't lying when she says she cares about her constituents—she wouldn't lie to people she cares about."
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Circular Reasoning"
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "I love eating burgers"
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "none"
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "If we ban Hummers because they are bad for the environment, eventually the government will ban all cars, so we should not ban Hummers."
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "False Dilemma"
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": text_comment
                    }
                ]
            }
        ],
        text={
            "format": {
                "type": "text"
            }
        },
        reasoning={},
        tools=[],
        temperature=0.1,
        max_output_tokens=20,
        top_p=0.8,
        store=True
    )
    print(response.output_text)
    return response.output_text