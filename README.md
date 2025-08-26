Here's a README.md explaining the main functions in taxonomy.py:

# Taxonomy Library README

This module provides functions for taxonomy creation, text classification, and header translation using AI models and vector similarity search.

## Main Functions

### 1. `propose_taxonomy(field: str, description: str, discrete_fields: list[str] = None)`
**Purpose**: Generate taxonomy suggestions using OpenAI
**Parameters**:
- `field`: Name of the field to categorize
- `description`: Description of the field's purpose
- `discrete_fields`: Optional specific values to consider

**Example**:
```python
propose_taxonomy(
    field="Color",
    description="Vehicle paint color classification",
    discrete_fields=["Red", "Blue", "Green", "Custom"]
)
# Returns: ["Red", "Blue", "Green", "Other"]
```

### 2. `apply_taxonomy_similarity(discrete_fields: list[str], taxonomy: list[str], category_type: str = None)`
**Purpose**: Classify values using semantic similarity with vector database
**Parameters**:
- `discrete_fields`: Values to classify
- `taxonomy`: List of allowed classification terms
- `category_type`: Special processing for categories like 'streets'

**Example**:
```python
apply_taxonomy_similarity(
    discrete_fields=["Rd", "Street", "Ave"],
    taxonomy=["Road", "Street", "Avenue"],
    category_type="streets"
)
# Returns: {'Rd': {'match': 'Road', 'score': 0.92}, ...}
```

### 3. `apply_taxonomy_reasoning(discrete_fields: list[str], taxonomy: list[str], classification_description: str, hash_file: str = None)`
**Purpose**: Classify values in chunks with validation and checkpointing
**Parameters**:
- `discrete_fields`: List of values to classify
- `taxonomy`: List of allowed categories
- `classification_description`: Context for classification
- `hash_file`: Optional file hash for progress tracking

**Example**:
```python
apply_taxonomy_reasoning(
    discrete_fields=["Quick Brown Fox", "Lazy Dog"],
    taxonomy=["Animal", "Object", "Action"],
    classification_description="Classify animal-related phrases"
)
# Returns: {'Quick Brown Fox': 'Animal', 'Lazy Dog': 'Animal'
```

### 4. `translate_headers_reasoning(src_lang, dest_lang, headers)`
**Purpose**: Translate column headers between languages
**Parameters**:
- `src_lang`: Source language code
- `dest_lang`: Target language code
- `headers`: List of headers to translate

**Example**:
```python
translate_headers_reasoning(
    src_lang="en",
    dest_lang="es",
    headers=["Street Name", "Zip Code"]
)
# Returns: {'Street Name': 'Nombre de la Calle', 'Zip Code': 'Código Postal'
```

### 5. `analyze_text_field(field_name: str, field_value: str, task: Literal["label", "summarize"] = "label")`
**Purpose**: Analyze text for labeling or summarization
**Parameters**:
- `field_name`: Name of the text field
- `field_value`: Text to analyze
- `task`: "label" for classification or "summarize" for text summary

**Example**:
```python
analyze_text_field(
    field_name="Product Description",
    field_value="This ergonomic chair provides lumbar support and adjustable height",
    task="label"
)
# Returns: "Office Furniture"
```

## Key Features
- Uses Hugging Face embeddings for semantic similarity
- Implements validation models with Pydantic
- Includes progress checkpointing for large datasets
- Handles special cases like street name normalization
- Uses Google search for ambiguous classifications

## Environment Variables
- `MODEL_NAME`: Hugging Face model identifier
- `SERVER_URL`: Base URL for OpenAI-compatible API
- `API_KEY`: Authentication token for the API