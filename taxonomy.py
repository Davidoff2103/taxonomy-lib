from dotenv import load_dotenv
import os
import json
import re
import textdistance
import unicodedata

from itertools import islice
#from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
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
        "CL": "CARRER",
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

def apply_cities_taxonomy(discrete_fields: list[str], taxonomy: list[str]):
    embedder = HuggingFaceEmbeddings(
        model_name="all-mpnet-base-v2",
        model_kwargs={"device": "cuda"}
    )
    #vectordb = Chroma.from_texts(
    #    texts=taxonomy,
    #    embedding=embedder,
    #    persist_directory="./chroma_streets",
    #    collection_metadata={"hnsw:space": "cosine"}
    #)
    vectordb = FAISS.from_texts(
        texts=taxonomy,
        embedding=embedder
    )

    results = {}
    to_check = 0
    for street in discrete_fields:
        result = vectordb.similarity_search_with_score(normalize_street_name(street), k=1)
        if result:
            matched_street, score = result[0]
            if score >= 0.35:
                to_check += 1
                results[street] = {"matched_street": matched_street.page_content, "to_check": True}
            else:
                results[street] = {"matched_street": matched_street.page_content}
            print(f"{street} → {matched_street.page_content} (Score: {score:.2f})")
        else:
            print(f"{street} → None")
    print(round(to_check*100/len(discrete_fields), 2), f"must be checked ({to_check}/{len(discrete_fields)})")
    return results

def chunks(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

def apply_taxonomy(discrete_fields: list[str], taxonomy: list[str]):
    client = OpenAI(base_url=os.getenv("SERVER_URL"), api_key=os.getenv("API_KEY"))

    chunk_size = 30
    x_taxonomy = {}
    total_chunks = (len(discrete_fields) + chunk_size - 1) // chunk_size

    for idx, part in enumerate(islice(chunks(discrete_fields, chunk_size), 5), start=1):
        print(f"Processing chunk {idx}/{total_chunks} ({len(part)} items)...")
        content = f"Discrete fields:\n{part}\n\nTaxonomies:\n{taxonomy}\n\n"
        response = client.chat.completions.create(
            model="Qwen/Qwen3-32B",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant that classifies discrete dataset fields into one of the allowed classification values.\n\n"
                        "Rules:\n"
                        "- You MUST choose exactly one value from the allowed classification list for each field.\n"
                        "- If no suitable classification exists for a field, output the value `null` (JSON null) for that field.\n"
                        "- Never invent or output any classification value not included in the list.\n"
                        "- If unsure between two, use these defaults:\n"
                        "  - BJ → \"Baixada\"\n"
                        "  - CL → \"Carrer\"\n"
                        "  - PZ → \"Plaça\"\n"
                        "  - AV → \"Avinguda\"\n"
                        "  - PJ → \"Passatge\"\n"
                        "  - PS → \"Passeig\" or \"Pas\"\n"
                        "  - RB → \"Rambla\"\n"
                        "  - TT → \"Torrent\"\n"
                        "- You MUST write the taxonomy exactly as it appears in the allowed list, including accents, case, articles and prepositions.\n"
                        "- Do NOT reorder, remove, or change any words from the taxonomy value.\n\n"
                        "Output format:\n"
                        "Return ONLY a valid JSON object with \"field\": \"taxonomy\" pairs, nothing else.\n\n"
                    )
                },
                {"role": "user", "content": content}
            ],
            temperature=0
            # extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
        print(re.sub(r"<think>.*?</think>\s*", "", response.choices[0].message.content, flags=re.DOTALL).strip())
        partial_result = json.loads(re.sub(r"<think>.*?</think>\s*", "", response.choices[0].message.content, flags=re.DOTALL).strip())
        x_taxonomy.update(partial_result)

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


def main():
    print(propose_taxonomy("SystemEnergySource", "The primary source or type of energy utilized by a system to operate"))

if __name__ == "__main__":
    main()
