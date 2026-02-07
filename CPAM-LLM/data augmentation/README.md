# CP Data Augmentation

A data augmentation tool for Constraint Programming (CP) datasets using chaotic mapping-based perturbations combined with LLM-powered reframing.

## Features

- **Chaotic Mapping**: Uses logistic chaotic mapping to generate diverse perturbation parameters
- **Multi-level Perturbation**:
  - Natural language description perturbation (synonym substitution, phrase insertion, sentence reordering)
  - Code perturbation (variable renaming, comment modification, spacing adjustment)
  - LLM-based semantic reframing
- **Code Validation**: Automatic syntax checking for generated code samples
- **Batch Processing**: Support for processing entire datasets with multiple iterations

## Requirements

```
pandas
numpy
requests
openpyxl
```

## Usage

### Basic Example

```python
from cp_data_augmentation import DataAugmentationPipeline
import pandas as pd

# Load your dataset
df = pd.read_excel('your_dataset.xlsx')

# Initialize pipeline
pipeline = DataAugmentationPipeline(
    api_key='your_api_key',
    x0=0.4,
    r=3.9,
    alpha_max=0.3,
    beta_max=0.3,
    gamma_max=0.3,
    use_llm=True
)

# Augment dataset
augmented_df = pipeline.augment_dataset(
    df=df,
    num_iterations=2,
    samples_per_iteration=50
)

# Save results
augmented_df.to_excel('augmented_dataset.xlsx', index=False)
```

### Configuration Parameters

- `api_key`: API key for LLM service
- `x0`: Initial value for chaotic mapping (0, 1)
- `r`: Chaotic parameter [3.5, 4]
- `alpha_max`: Maximum natural language perturbation rate
- `beta_max`: Maximum model perturbation rate
- `gamma_max`: Maximum code perturbation rate
- `use_llm`: Enable/disable LLM reframing

## Input Format

Expected Excel columns:
- `Number`: Sample ID
- `CP natural language`: Problem description
- `CP code`: Python implementation
- `Constraint`: Constraint type
- `Problem types`: Problem category
- `CP formal language`: Formal model (optional)

## Output

The augmented dataset includes:
- Original samples
- Generated augmented samples with metadata
- Perturbation parameters (alpha, beta, gamma) for each sample
- Validation statistics

## License

MIT
