from googlesearch import search
from dotenv import load_dotenv
from pydantic import field_validator, BaseModel, ValidationError
from tqdm import tqdm
import asyncio
import os
import faiss
import json
import re
import numpy as np

from itertools import islice
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from typing import Literal

load_dotenv()


class TaxonomyModel(BaseModel):
    discrete_fields: list[str]
    taxonomy: list[str]
    partial_result: dict

    @field_validator("partial_result")
    def keys_must_match_data(cls, v, info):
        fields = info.data.get("discrete_fields", [])
        taxonomy = info.data.get("taxonomy", [])
        if set(v.keys()) != set(fields):
            raise ValueError(
                f"Las claves de la lista {fields} no coinciden exactamente con las del diccionario {list(v.keys())}"
            )
        elif not set(v.values()).issubset(set(taxonomy + ["null"] + [None])):
            raise ValueError(
                f"Algunos valores del diccionario {list(v.values())} no están en la lista {taxonomy}"
            )
        print("Validation succeed.\n")
        return v


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
    # Qwen/Qwen3-32B
    response = client.chat.completions.create(
        model="Qwen/Qwen3-32B",
        messages=[
            {"role": "system", "content": "You are an assistant responsible for proposing taxonomies based on examples. You must respond only with the list of proposed taxonomies."},
            {"role": "user", "content": message}
        ]
    )

    return json.loads(response.choices[0].message.content.split("</think>", 1)[1].strip())

def normalize_street_name(street: str):
    s = street.split()
    MAPPING = {
        "CL": "Carrer",
        "BJ": "Baixada",
        "PZ": "Plaça",
        "AV": "Avinguda",
        "PJ": "Passatge",
        "PS": "Passeig",
        "RB": "Rambla",
        "TT": "Torrent"
    }

    parts = street.strip().split(maxsplit=1)
    first_word = parts[0]
    rest = parts[1]
    normalized_first = MAPPING.get(first_word, first_word)
    return f"{normalized_first} {rest}".strip()

def apply_taxonomy_similarity(discrete_fields: list[str], taxonomy: list[str], category_type: str = None):
    embedder = HuggingFaceEmbeddings(
        model_name="all-mpnet-base-v2",
        model_kwargs={"device": "cuda"}
    )

    vectordb = Chroma.from_texts(
        texts=taxonomy,
        embedding=embedder,
        persist_directory="./chroma",
        collection_metadata={"hnsw:space": "l2"}
    )
#    vectordb = FAISS.from_texts(
#        texts=taxonomy,
#        embedding=embedder
#    )

    results = {}
    to_check = 0
    for field in discrete_fields:
        if category_type == "streets":
            result = vectordb.similarity_search_with_score(normalize_street_name(field), k=1)
        else:
            result = vectordb.similarity_search_with_score(field, k=1)

        if result:
            match, score = result[0]
            if score >= 0.35:
                to_check += 1
                results[field] = {"match": match.page_content, "to_check": True}
            else:
                results[field] = {"match": match.page_content}
            print(f"{field} → {match.page_content} (Score: {score:.2f})")
        else:
            print(f"{field} → None")
    print(round(to_check*100/len(discrete_fields), 2), f"% must be checked ({to_check}/{len(discrete_fields)})")
    return results

def chunks(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

def reasoning(client, part, taxonomy, classification_description):
    content = f"Discrete fields values:\n{part}\n\nTaxonomies:\n{taxonomy}\n\nClassification description: {classification_description}\n\n"
    messages = [
            {
                "role": "system",
                "content": (
                    "You are an assistant that classifies discrete dataset fields values into one of the allowed classification values.\n\n"
                    "Rules:\n"
                    "- You MUST choose exactly one value from the allowed classification list for each field value.\n"
                    "- First, if you do not have enough information to assign a classification to certain field values confidently, "
                    "you must generate a tool call for the tool web_search to get information about each of them.\n"
                    "- If no suitable classification exists for a field value, output the value `null` (JSON null) for that.\n"
                    "- Never invent or output any classification value not included in the list.\n"
                    "Output format:\n"
                    "Return ONLY a valid JSON object with \"field\": \"taxonomy\" pairs for all provided values, nothing else.\n"
                    "Before providing the answer, you must check all discrete fields values provided by the user are present in your response,"
                    "written exactly as originally (including typos).\n\n"
                )
            },
            {"role": "user", "content": content}
        ]
    response = client.chat.completions.create(
        model="Qwen/Qwen3-32B",
        messages=messages,
        tools = [{
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Get information about a discrete field value when you do not have enough information to classify it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query to search about, consisting of the concatenation of the discrete field value and the classification description (provided by the user)."
                        }
                    },
                    "required": ["query"]
                }
            }
        }],
        # temperature=0
        # extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    )
    # print(response.choices[0].message.content)

    completion_tool_calls = response.choices[0].message.tool_calls

    if completion_tool_calls:
        messages.append({
            "role": "assistant",
            "tool_calls": completion_tool_calls
        })
        print(completion_tool_calls)
        for call in completion_tool_calls:
            args = json.loads(call.function.arguments)
            result = web_search(**args)
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
    print(re.sub(r"<think>.*?</think>\s*", "", response.choices[0].message.content, flags=re.DOTALL).strip())
    res = re.sub(r"<think>.*?</think>\s*", "", response.choices[0].message.content, flags=re.DOTALL).strip()
    if "```json" in res:
        res = re.sub(r"```json\s*(.*?)\s*```", r"\1", res, flags=re.DOTALL).strip()
    partial_result = json.loads(res)
    return partial_result


def apply_taxonomy_reasoning(discrete_fields: list[str], taxonomy: list[str], classification_description: str):
    client = OpenAI(base_url=os.getenv("SERVER_URL"), api_key=os.getenv("API_KEY"))

    chunk_size = 15
    x_taxonomy = {}
    total_chunks = (len(discrete_fields) + chunk_size - 1) // chunk_size

    for idx, part in enumerate(tqdm(islice(chunks(discrete_fields, chunk_size), total_chunks), total=total_chunks, desc="Classifying chunks"), start=1):
#        print(f"Processing chunk {idx}/{total_chunks} ({len(part)} items)...")
        while True:
            try:
                partial_result = reasoning(client, part, taxonomy, classification_description)
                validated = TaxonomyModel(partial_result=partial_result, taxonomy=taxonomy, discrete_fields=part)
                x_taxonomy.update(validated.partial_result)
                break
            except ValidationError as e:
                print(f"Validation failed for chunk {idx}, retrying...\n{e}")

    # return response.choices[0].message.content
    # return json.loads(re.sub(r"<think>.*?</think>\s*", "", response.choices[0].message.content, flags=re.DOTALL).strip())
    return x_taxonomy


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


def translate_taxonomy_reasoning(src_lang, dest_lang, headers):
    client = OpenAI(base_url=os.getenv("SERVER_URL"), api_key=os.getenv("API_KEY"))
    response = client.chat.completions.create(
        model="Qwen/Qwen3-32B",
        messages=[
            {"role": "system", "content": "You are an assistant that translate column names. You have to respond with a JSON where the key is the original text and the value is the translated text."},
            {"role": "user", "content": f"Translate the following list from {src_lang} to {dest_lang}: {json.dumps(headers, ensure_ascii=False)}"}
        ]
    )
    return json.loads(re.sub(r"<think>.*?</think>\s*", "", response.choices[0].message.content, flags=re.DOTALL).strip())


def web_search(query):
    #serp_api_wrapper = SerpAPIWrapper(serpapi_api_key=os.getenv("SERPAPI_API_KEY"))
    #return serp_api_wrapper.run(query)
    results = []
    for r in search(query, advanced=True, num_results=5):
        results.append({
            "url": r.url,
            "title": r.title,
            "description": r.description
        })

    return results


def main():
    print(propose_taxonomy("SystemEnergySource", "The primary source or type of energy utilized by a system to operate"))

if __name__ == "__main__":
    main()
