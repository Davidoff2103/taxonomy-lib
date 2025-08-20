from langchain_community.utilities import SerpAPIWrapper
from dotenv import load_dotenv
import os
import json
import re
from openai import OpenAI
from typing import Literal

load_dotenv()

def propose_taxonomy(field: str, description:str) -> list[str]:
    client = OpenAI(base_url=os.getenv("SERVER_URL"), api_key=os.getenv("API_KEY"))
    examples = """
    Example 1:
    Field: "Color"
    Taxonomy: ["Red", "Blue", "Green"]

    Example 2:
    Field: "Marital status"
    Taxonomy: ["Single", "Married", "Divorced"]
    """

    message = f"""
    Below are some examples of taxonomies for discrete fields:

    {examples}

    Now, for this new field:
    Field: "{field}"
    Description: "{description}"

    Propose a taxonomy.
    """

    response = client.chat.completions.create(
        model="Qwen/Qwen3-32B",
        messages=[
            {"role": "system", "content": "You are an assistant responsible for proposing taxonomies based on examples. You must respond only with the list of proposed taxonomies."},
            {"role": "user", "content": message}
        ]
    )

    return json.loads(response.choices[0].message.content.split("</think>", 1)[1].strip())

def apply_taxonomy(discrete_fields: list[str], taxonomy: list[str]):
    client = OpenAI(base_url=os.getenv("SERVER_URL"), api_key=os.getenv("API_KEY"))

    content = f"Discrete fields:\n{discrete_fields}\n\nTaxonomies:\n{taxonomy}\n\nClassify each field into one of the taxonomies. Respond only with key-value pairs in JSON format."

    messages = [
                {"role": "system", "content": "You are an assistant that classifies discrete dataset fields into taxonomies. Only use the web_search tool if you do not have enough information to assign a taxonomy confidently and you think you can find useful information on internet. Respond only with a JSON object containing field: taxonomy pairs."},
                {"role": "user", "content": content}
            ]
    response = client.chat.completions.create(
        model="Qwen/Qwen3-32B",
        messages=messages,
        tools = [{
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Get additional information about a discrete field value",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query to search about"
                        }
                    },
                    "required": ["query"]
                }
            }
        }]
    )

    messages.append({
        "role": "assistant",
        "tool_calls": response.choices[0].message.tool_calls
    })

    completion_tool_calls = response.choices[0].message.tool_calls
    for call in completion_tool_calls:
        print(call)
        args = json.loads(call.function.arguments)
        result = web_search(**args)
        print(result)
        messages.append({
            "role": "tool",
            "content": json.dumps({
                "query": args["query"],
                "result": result
            }),
            "tool_call_id": call.id,
            "name": call.function.name
        })
    response = client.chat.completions.create(
        model="Qwen/Qwen3-32B",
        messages=messages
    )
    return json.loads(re.sub(r"<think>.*?</think>\s*", "", response.choices[0].message.content, flags=re.DOTALL).strip())


def analyze_text_field(field_name: str, field_value: str, task: Literal["label", "summarize"] = "label"):
    client = OpenAI(base_url=os.getenv("SERVER_URL"), api_key=os.getenv("API_KEY"))

    if task not in {"label", "summarize"}:
        raise ValueError("task must be 'label' or 'summarize'")


    if task == "label":
        user_prompt = (
            f'Please classify the following text from the field "{field_name}" '
            f'into one concise label such as sentiment or topic.\n\nText: "{field_value}"\n\nLabel:'
        )
    else:
        user_prompt = (
            f'Please provide a very brief summary of the following text from the field "{field_name}".\n\n'
            f'Text: "{field_value}"\n\nSummary:'
        )


    response = client.chat.completions.create(
        model="Qwen/Qwen3-32B",
        messages=[
            {"role": "system", "content": "You are an assistant that analyzes natural language text fields. You have to respond with only the label or only the summary, without any additional text."},
            {"role": "user", "content": user_prompt}
        ]
    )

    return re.sub(r"<think>.*?</think>\s*", "", response.choices[0].message.content, flags=re.DOTALL).strip()


def web_search(query):
    """Search the web for additional information about the query topic, or if the question is outside the scope of the RAG tool."""
    serp_api_wrapper = SerpAPIWrapper(serpapi_api_key=os.getenv("SERPAPI_API_KEY"))
    return serp_api_wrapper.run(query)


def main():
    print(propose_taxonomy("SystemEnergySource", "The primary source or type of energy utilized by a system to operate"))

if __name__ == "__main__":
    main()
