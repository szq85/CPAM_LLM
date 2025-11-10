# RAG-FCA System for Optimization Problem Processing

A complete implementation of Retrieval-Augmented Generation via Formal Concept Analysis (RAG-FCA) for processing constrained optimization problems.

## Overview

This system implements the RAG-FCA framework as described in the research paper, providing:

- **Formal Knowledge Representation**: Constructs formal backgrounds K=(G,M,I) to mathematically represent optimization problems
- **Concept Lattice Construction**: Builds hierarchical knowledge structures using Formal Concept Analysis
- **Semantic Disambiguation**: Resolves ambiguous terminology using Jaccard similarity on concept lattices
- **Implicit Constraint Discovery**: Automatically infers missing constraints using rough set theory
- **LLM Integration**: Generates formal problem descriptions using Claude API

## Architecture

### Core Components

1. **FormalBackground**: Represents the formal background K=(G,M,I)
   - Objects (G): Optimization problem instances
   - Attributes (M): Problem characteristics
   - Incidence Relation (I): Binary/fuzzy relations between objects and attributes

2. **ConceptLattice**: Constructs and manages the concept lattice L(K)
   - Extracts formal concepts using closure operators
   - Implements partial order relations
   - Provides Jaccard similarity matching

3. **RoughSetProcessor**: Handles incomplete information
   - Computes indiscernibility relations
   - Calculates lower/upper approximations
   - Discovers certain and possible constraints

4. **RAGFCASystem**: Main system orchestrator
   - Attribute extraction from natural language
   - Semantic disambiguation
   - Query augmentation
   - LLM-based formal description generation

## Installation

```bash
# Install required packages
pip install numpy

# Optional: For Claude API integration
pip install anthropic
```

## Usage

### Basic Usage

```python
from rag_fca_system import RAGFCASystem

# Initialize the system
system = RAGFCASystem()

# Define your optimization problem in natural language
problem_description = """
We have a reconfigurable photovoltaic energy storage system composed of 
multiple parallel battery modules and series columns that need optimization. 
The goal is to select the activated battery module configuration for each 
period based on different demands in order to minimize the total energy loss, 
while ensuring that the constraints of parallel activation consistency, 
system voltage range, and maximum module current are met.
"""

# Process the problem
result = system.process_optimization_problem(problem_description)

# Display formal description
import json
print(json.dumps(result, indent=2))
```

### Output Format

The system generates a structured JSON output containing:

```json
{
  "decision_variables": [
    {
      "name": "x_ipt",
      "type": "binary",
      "dimension": "i ∈ {1,...,N_modules}, p ∈ {1,...,N_parallel}, t ∈ {1,...,T}",
      "description": "Binary variable indicating whether battery module i in parallel branch p is activated at time period t"
    }
  ],
  "objective": {
    "type": "minimize",
    "expression": "∑_{t=1}^{T} E_loss(t)",
    "description": "Minimize total energy loss across all time periods"
  },
  "constraints": [
    {
      "name": "parallel_consistency",
      "type": "implicit",
      "expression": "∑_{i=1}^{N_modules} x_ipt = ∑_{i=1}^{N_modules} x_ip't",
      "description": "All parallel branches must have the same number of activated modules"
    }
  ],
  "parameters": [
    {
      "name": "N_modules",
      "description": "Number of battery modules in series per column"
    }
  ],
  "metadata": {
    "extracted_attributes": [...],
    "processing_method": "RAG-FCA"
  }
}
```

### Claude API Integration

To use the real Claude API instead of demo responses:

```python
from api_integration_example import RAGFCASystemWithAPI
import os

# Set your API key
os.environ["ANTHROPIC_API_KEY"] = "your-api-key-here"

# Initialize system with API
system = RAGFCASystemWithAPI()

# Process problems with real LLM
result = system.process_optimization_problem(problem_description)
```

### Knowledge Base Management

#### Export Knowledge Base
```python
system.export_knowledge_base("my_knowledge_base.json")
```

#### Import Knowledge Base
```python
system.import_knowledge_base("my_knowledge_base.json")
```

#### Update with New Problems
```python
new_attributes = {
    "portfolio_optimization": 1.0,
    "risk_minimization": 1.0,
    "return_constraint": 1.0,
    "continuous_variables": 1.0
}
system.update_knowledge_base("portfolio_opt_1", new_attributes)
```

## Pre-configured Problem Domains

The system comes pre-configured with knowledge about several optimization problem domains:

1. **DNA Sequence Design** (C1)
   - Attributes: GC_content_constraint, base_pair_complementarity, sequence_design

2. **Energy Storage Systems** (C2)
   - Attributes: battery_voltage_range, energy_loss_minimization, module_configuration, parallel_consistency

3. **Vehicle Routing** (C3)
   - Attributes: capacity_constraint, distance_minimization, routing_problem, vehicle_assignment

4. **Facility Location** (C4)
   - Attributes: capacity_constraint, demand_satisfaction, location_selection, facility_location

5. **Scheduling Problems** (C5)
   - Attributes: temporal_constraint, makespan_minimization, task_assignment, precedence_constraint

## Key Features

### 1. Semantic Disambiguation

The system uses Jaccard similarity to disambiguate ambiguous terms:

```
score(Q,B) = |Q ∩ B| / |Q ∪ B|
```

For fuzzy queries:
```
score_f(Q,B) = Σ min(μ_Q(m), μ_B(m)) / Σ max(μ_Q(m), μ_B(m))
```

### 2. Implicit Constraint Discovery

Using rough set theory, the system discovers:

- **Certain Constraints**: Attributes that definitely apply (lower approximation)
- **Possible Constraints**: Attributes that possibly apply (upper approximation)

```python
certain, possible = system.augment_query(query_attributes)
```

### 3. Concept Lattice Operations

Implements formal concept analysis operations:

- Prime operations: A' and B'
- Double prime operations: A'' and B''
- Meet and join operations: ∧ and ∨
- Partial order relations: ≤

## Mathematical Foundations

### Formal Concept Definition

A formal concept is a pair C=(A,B) where:
- A ⊆ G (objects/extensions)
- B ⊆ M (attributes/intensions)

Satisfying the maximality condition:
```
A' = B and B' = A
```

Where:
```
A' = {m ∈ M | ∀g ∈ A, (g,m) ∈ I}
B' = {g ∈ G | ∀m ∈ B, (g,m) ∈ I}
```

### Partial Order

Concepts form a lattice with partial order:
```
(A₁,B₁) ≤ (A₂,B₂) ⟺ A₁ ⊆ A₂ ⟺ B₂ ⊆ B₁
```

### Rough Set Approximations

Lower approximation (certain):
```
apr_B(A) = {g ∈ G | [g]_{IND(B)} ⊆ A}
```

Upper approximation (possible):
```
apr̄_B(A) = {g ∈ G | [g]_{IND(B)} ∩ A ≠ ∅}
```

## Example Problems

### 1. Battery Energy Storage System

```python
problem = """
We have a reconfigurable photovoltaic energy storage system composed of 
multiple parallel battery modules and series columns that need optimization. 
The goal is to select the activated battery module configuration for each 
period based on different demands in order to minimize the total energy loss.
"""
```

**Output**: Binary decision variables, energy loss minimization objective, voltage range and current limit constraints

### 2. Vehicle Routing Problem

```python
problem = """
Route 5 delivery vehicles to serve 30 customers with minimum total distance 
while respecting vehicle capacity constraints.
"""
```

**Output**: Routing variables, distance minimization objective, capacity constraints

### 3. Job Shop Scheduling

```python
problem = """
Schedule 20 jobs on 4 machines to minimize total completion time (makespan) 
while respecting precedence constraints and machine availability.
"""
```

**Output**: Assignment variables, makespan objective, temporal and precedence constraints

## API Reference

### RAGFCASystem

#### Methods

- `process_optimization_problem(description: str) -> Dict`: Main processing method
- `extract_attributes_from_text(text: str) -> Set[str]`: Extract attributes from natural language
- `disambiguate_semantics(query_attrs: Set[str]) -> FormalConcept`: Semantic disambiguation
- `augment_query(query_attrs: Set[str]) -> Tuple[Set[str], Set[str]]`: Query augmentation
- `update_knowledge_base(obj: str, attrs: Dict[str, float])`: Incremental learning
- `export_knowledge_base(filepath: str)`: Export KB to JSON
- `import_knowledge_base(filepath: str)`: Import KB from JSON

### FormalBackground

#### Methods

- `add_object(obj: str)`: Add problem instance
- `add_attribute(attr: str)`: Add problem characteristic
- `set_relation(obj: str, attr: str, degree: float)`: Set incidence relation
- `has_relation(obj: str, attr: str, threshold: float) -> bool`: Check relation
- `prime_objects(objects: Set[str]) -> Set[str]`: Compute A'
- `prime_attributes(attributes: Set[str]) -> Set[str]`: Compute B'

### ConceptLattice

#### Methods

- `find_concept_by_attributes(query_attrs: Set[str]) -> FormalConcept`: Find matching concept
- `fuzzy_jaccard_similarity(query: Dict, attrs: Set) -> float`: Fuzzy similarity

### RoughSetProcessor

#### Methods

- `lower_approximation(target: Set[str], attrs: Set[str]) -> Set[str]`: Certain objects
- `upper_approximation(target: Set[str], attrs: Set[str]) -> Set[str]`: Possible objects
- `discover_implicit_constraints(query_attrs: Set[str]) -> Tuple[Set, Set]`: Find constraints

## Performance Considerations

- **Concept Lattice Construction**: O(2^|M|) in worst case, optimized with closure operators
- **Attribute Extraction**: O(|text| × |keywords|)
- **Semantic Matching**: O(|concepts| × |attributes|)
- **Rough Set Computation**: O(|G| × |M|)

For large knowledge bases (>1000 objects), consider:
- Incremental lattice construction
- Attribute indexing
- Concept caching

## Extending the System

### Adding New Problem Domains

```python
# Define new problem type
system.background.set_relation("new_problem_1", "new_attribute_1", 1.0)
system.background.set_relation("new_problem_1", "new_attribute_2", 1.0)

# Rebuild lattice
system.lattice = ConceptLattice(system.background)
system.rough_processor = RoughSetProcessor(system.background)
```

### Custom Attribute Extractors

```python
def custom_extractor(text: str) -> Set[str]:
    # Your custom extraction logic
    attributes = set()
    # ... extract attributes from text
    return attributes

# Override in RAGFCASystem
system.extract_attributes_from_text = custom_extractor
```

### Custom LLM Integration

```python
class CustomRAGFCA(RAGFCASystem):
    def _call_llm_api(self, prompt: str) -> str:
        # Your custom LLM API call
        response = your_llm_api(prompt)
        return response
```

## Files Included

- `rag_fca_system.py`: Main implementation
- `api_integration_example.py`: Examples with Claude API integration
- `knowledge_base.json`: Exported knowledge base
- `README.md`: This documentation
