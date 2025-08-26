# Taxonomy Library README

This module provides functions for taxonomy creation, text classification, and header translation using AI models and vector similarity search.

## Usage

Import the module with:
```python
from bee_taxonomy import taxonomy
```

## Main Functions

### 1. `taxonomy.propose_taxonomy(field: str, description: str, discrete_fields: list[str] = None)`
**Purpose**: Generate taxonomy suggestions using OpenAI
**Parameters**:
- `field`: Name of the field to categorize
- `description`: Description of the field's purpose
- `discrete_fields`: Optional specific values to consider

**Example**:
```python
taxonomy.propose_taxonomy(
    field="Color",
    description="Vehicle paint color classification",
    discrete_fields=["Red", "Blue", "Green", "Custom"]
)
# Returns: ["Red", "Blue", "Green", "Other"]
```

### 2. `taxonomy.apply_taxonomy_similarity(discrete_fields: list[str], taxonomy: list[str], category_type: str = None)`
**Purpose**: Classify values using semantic similarity with vector database
**Parameters**:
- `discrete_fields`: Values to classify
- `taxonomy`: List of allowed classification terms
- `category_type`: Special processing for categories like 'streets'

**Example**:
```python
taxonomy.apply_taxonomy_similarity(
    discrete_fields=["Rd", "Street", "Ave"],
    taxonomy=["Road", "Street", "Avenue"],
    category_type="streets"
)
# Returns: {'Rd': {'match': 'Road', 'score': 0.92}, ...}
```

### 3. `taxonomy.apply_taxonomy_reasoning(discrete_fields: list[str], taxonomy: list[str], classification_description: str, hash_file: str = None)`
**Parameters**:
- `discrete_fields`: List of values to classify
- `taxonomy`: List of allowed categories
- `classification_description`: Context for classification
- `hash_file`: Optional file hash for progress tracking

**Example**:
```python
taxonomy.apply_taxonomy_reasoning(
    discrete_fields=["Quick Brown Fox", "Lazy Dog"],
    taxonomy=["Animal", "Object", "Action"],
    classification_description="Classify animal-related phrases"
)
# Returns: {'Quick Brown Fox': 'Animal', 'Lazy Dog': 'Animal'
```

### 4. `taxonomy.translate_headers_reasoning(src_lang, dest_lang, headers)`
**Parameters**:
- `src_lang`: Source language code
- `dest_lang`: Target language code
- `headers`: List of headers to translate

**Example**:
```python
taxonomy.translate_headers_reasoning(
    src_lang="en",
    dest_lang="es",
    headers=["Street Name", "Zip Code"]
)
# Returns: {'Street Name': 'Nombre de la Calle', 'Zip Code': 'Código Postal'
```

### 5. `taxonomy.analyze_text_field(field_name: str, field_value: str, task: Literal["label", "summarize"] = "label")`
**Parameters**:
- `field_name`: Name of the text field
- `field_value`: Text to analyze
- `task`: "label" for classification or "summarize" for text summary

**Example**:
```python
taxonomy.analyze_text_field(
    field_name="Product Description",
    field_value="This ergonomic chair provides lumbar support and adjustable height",
    task="label"
)
# Returns: "Office Furniture"
```

## Environment Variables
- `MODEL_NAME`: Hugging Face model identifier
- `SERVER_URL`: Base URL for OpenAI-compatible API
- `API_KEY`: Authentication token for the API
- `EMBEDDER_MODEL`: Embedding model for semantic similarity

## Features
- Validation workflow with Pydantic models
- Progress checkpointing for large datasets
- Google search integration for ambiguous classifications