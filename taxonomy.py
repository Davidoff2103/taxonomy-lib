import glob
import json
import os
import re
from itertools import islice
from typing import Literal

from dotenv import load_dotenv
from googlesearch import search
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from pydantic import field_validator, BaseModel, ValidationError
from tqdm import tqdm
from utils import normalize_street_name

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME")
SERVER_URL = os.getenv("SERVER_URL")
API_KEY = os.getenv("API_KEY")


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
                f"The keys in the list {fields} do not exactly match the keys in the dictionary {list(v.keys())}"
            )
        elif not set(v.values()).issubset(set(taxonomy + ["null"] + [None])):
            raise ValueError(
                f"Some values of the dictionary {list(v.values())} are not in the list {taxonomy}"
            )
        return v


class TranslationModel(BaseModel):
    headers: list[str]
    translations: dict

    @field_validator("translations")
    def keys_must_match_data(cls, v, info):
        headers = info.data.get("headers", [])
        if set(v.keys()) != set(headers):
            raise ValueError(
                f"The keys in the list {headers} do not exactly match the keys in the dictionary {list(v.keys())}"
            )
        return v


def deprecate_cache_file(file: str = None):
    if os.path.isfile(file):
        os.remove(file)
        print(f"Removed file: {file}")
    else:
        files = glob.glob("*.tmp")
        if files:
            for f in files:
                try:
                    os.remove(f)
                    print(f"Removed file: {f}")
                except Exception as e:
                    print(f"Cannot remove {f}: {e}")
        else:
            print("No files to remove")


def propose_taxonomy(field: str, description: str, discrete_fields: list[str] = None) -> list[str]:
    client = OpenAI(base_url=SERVER_URL, api_key=API_KEY)
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

    if discrete_fields:
        message += f"\nTake the following list as discrete fields to classify: {discrete_fields}."

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": (
                "You are an assistant responsible for proposing taxonomies based on examples."
                "You must respond only with the list of proposed taxonomies."
            )},
            {"role": "user", "content": message}
        ]
    )
    return json.loads(re.sub(r"<[^>]+>.*?</[^>]+>\s*", "", response.choices[0].message.content, flags=re.DOTALL).strip())


def apply_taxonomy_similarity(discrete_fields: list[str], taxonomy: list[str], category_type: str = None):
    embedder = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDER_MODEL"),
        model_kwargs={"device": "cuda"}
    )

    vectordb = Chroma.from_texts(
        texts=taxonomy,
        embedding=embedder,
        persist_directory="./chroma",
        collection_metadata={"hnsw:space": "l2"}
    )

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
    content = (f"Discrete fields values:\n{part}\n\nTaxonomies:\n{taxonomy}\n\n"
               f"Classification description: {classification_description}\n\n")
    messages = [
            {
                "role": "system",
                "content": (
                    "You are an assistant that classifies discrete dataset fields values into one "
                    "of the allowed classification values.\n\n"
                    "Rules:\n"
                    "- You MUST choose exactly one value from the allowed classification list for each field value.\n"
                    "- First, if you do not have enough information to assign a classification "
                    "to certain field values confidently, "
                    "you must generate a tool call for the tool web_search to get information about each of them.\n"
                    "- If no suitable classification exists for a field value, "
                    "output the value `null` (JSON null) for that.\n"
                    "- Never invent or output any classification value not included in the list.\n"
                    "Output format:\n"
                    "Return ONLY a valid JSON object with \"field\": \"taxonomy\" pairs for all provided values, "
                    "nothing else.\n"
                    "Check all discrete fields values provided by the user are present in your response, "
                    "written exactly as originally (including typos).\n"
                    "Be especially careful with the quotes of the same type that were used to define the string, "
                    "you can't forget any.\n\n"
                )
            },
            {"role": "user", "content": content}
        ]
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools = [{
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Get information about a discrete field value when you do not have "
                               "enough information to classify it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query to search about, consisting of the concatenation of the discrete "
                                           "field value and the classification description (provided by the user)."
                        }
                    },
                    "required": ["query"]
                }
            }
        }],
        # temperature=0
        # extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    )

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
            model=MODEL_NAME,
            messages=messages
        )
    res = re.sub(r"<[^>]+>.*?</[^>]+>\s*", "", response.choices[0].message.content, flags=re.DOTALL).strip()
    print(res)
    if "```json" in res:
        res = re.sub(r"```json\s*(.*?)\s*```", r"\1", res, flags=re.DOTALL).strip()
    partial_result = json.loads(res)
    return partial_result


def load_checkpoint(tmp_file: str):
    if os.path.exists(tmp_file):
        with open(tmp_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(tmp_file, data):
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def apply_taxonomy_reasoning(discrete_fields: list[str], taxonomy: list[str],
                             classification_description: str, hash_file: str = None):
    client = OpenAI(base_url=SERVER_URL, api_key=API_KEY)

    chunk_size = 15
    x_taxonomy = {}
    total_chunks = (len(discrete_fields) + chunk_size - 1) // chunk_size

    if hash_file:
        tmp_file = hash_file + "_cache.tmp"
        x_taxonomy = load_checkpoint(tmp_file)
        already_done = set(x_taxonomy.keys())

    for idx, part in enumerate(tqdm(islice(chunks(discrete_fields, chunk_size), total_chunks), total=total_chunks,
                                    desc="Classifying chunks"), start=1):
        # print(f"Processing chunk {idx}/{total_chunks} ({len(part)} items)...")
        if hash_file and all(field in already_done for field in part):
            print(f"Chunk {idx} already processed, skipping...")
            continue

        while True:
            try:
                partial_result = reasoning(client, part, taxonomy, classification_description)
                partial_result = {str(k): v for k, v in zip(part, partial_result.values())}
                validated = TaxonomyModel(partial_result=partial_result, taxonomy=taxonomy, discrete_fields=part)
                x_taxonomy.update(validated.partial_result)

                if hash_file:
                    save_checkpoint(tmp_file, x_taxonomy)
                    print(f"✅ Chunk {idx} validated and cached on {tmp_file}.")
                break
            except ValidationError as e:
                print(f"Validation failed for chunk {idx}, retrying...\n{e}")

    return x_taxonomy


def analyze_text_field(field_name: str, field_value: str, task: Literal["label", "summarize"] = "label"):
    client = OpenAI(base_url=SERVER_URL, api_key=API_KEY)

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
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": (
                "You are an assistant that analyzes natural language text fields."
                "You have to respond with only the label or only the summary, without any additional text."
            )},
            {"role": "user", "content": user_prompt}
        ]
    )

    return re.sub(
        r"<[^>]+>.*?</[^>]+>\s*", "", response.choices[0].message.content, flags=re.DOTALL).strip()


def translate_taxonomy_reasoning(src_lang, dest_lang, headers):
    client = OpenAI(base_url=SERVER_URL, api_key=API_KEY)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system",
             "content": (
                  "You are an assistant that translate column names. "
                  "You have to respond with a JSON where the key is the original text and "
                  "the value is the translated text.\n"
                  "Check all headers provided by the user are present in your response, "
                  "written exactly as originally (including typos).\n"
             )},
            {"role": "user", "content": f"Translate the following list from {src_lang} to {dest_lang}: "
                                        f"{json.dumps(headers, ensure_ascii=False)}"}
        ]
    )
    try:
        translations = json.loads(
            re.sub(r"<[^>]+>.*?</[^>]+>\s*", "", response.choices[0].message.content,
                   flags=re.DOTALL).strip())
        validated = TranslationModel(translations=translations, headers=headers)
        print("✅ Translation validated.")
        return validated.translations
    except ValidationError as e:
        print(f"Validation failed for translation...\n{e}")


def web_search(query):
    results = []
    for r in search(query, advanced=True, num_results=5):
        if r.url and r.title and r.description:
            results.append({
                "url": r.url,
                "title": r.title,
                "description": r.description
            })

    return results
