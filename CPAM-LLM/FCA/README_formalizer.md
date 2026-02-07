# CP Problem Formalizer

A tool for transforming natural language descriptions of Constraint Programming (CP) problems into structured formal descriptions using RAG-FCA (Retrieval-Augmented Generation with Formal Concept Analysis) and reference examples.

## Features

- **Reference-Based Generation**: Uses pre-defined reference examples as templates for consistent formalization
- **Formal Concept Analysis**: Organizes problems into concept lattices based on attributes
- **LLM-Powered**: Leverages large language models for intelligent rewriting and adaptation
- **Structured Output**: Generates standardized formal descriptions with consistent sections
- **Multi-Domain Support**: Handles various CP domains (manufacturing, logistics, facility location, etc.)

## Requirements

```
pandas
requests
openpyxl
```

## Usage

### Basic Example

```python
from cp_formalizer import ProblemFormalizer, load_reference_examples
import pandas as pd

# Load dataset
df = pd.read_excel('your_dataset.xlsx')

# Load reference examples (using specific row indices)
reference_indices = [0, 61, 77, 108, 124]
references = load_reference_examples(df, reference_indices)

# Initialize formalizer
formalizer = ProblemFormalizer(
    api_base="https://api.chatanywhere.tech/v1",
    reference_examples=references
)

# Generate formal descriptions
for idx, row in df.iterrows():
    formal_desc = formalizer.generate_formalized_description(row)
    df.loc[idx, 'CP formal language generated'] = formal_desc

# Save results
df.to_excel('formalized_output.xlsx', index=False)
```

### Input Format

Expected Excel columns:
- `Number`: Problem ID
- `Problem types`: Problem category (e.g., "Aircraft Skin Processing", "VRP")
- `Constraint`: Constraint type (e.g., "base", "temporal")
- `CP natural language`: Natural language problem description
- `CP code`: Python implementation code
- `CP formal language`: Existing formal description (for reference examples)

## Output Structure

Generated formal descriptions follow this format:

1. **Decision Variables**: Core decision elements
2. **Parameters and Input Data**: Static input parameters and problem dimensions
3. **Constraints**: All problem constraints (base + additional variants)
4. **Objective Function**: Optimization objective
5. **Data Specification**: Dataset format and structure (when applicable)

## Supported Problem Types

- Aircraft Skin Processing (job shop scheduling)
- Battery Pack Design (energy systems)
- Charging Station Location (facility location)
- DNA Sequence Design (molecular biology)
- Vehicle Routing Problem (logistics)
- And more...

## Configuration

Key parameters:
- `api_base`: LLM API endpoint URL
- `api_key`: Authentication key for LLM service
- `reference_examples`: Dictionary of reference problems with formal descriptions
- `temperature`: LLM generation temperature (default: 0.5 for consistency)
- `max_tokens`: Maximum tokens for LLM response (default: 3500)

## License

MIT
