# Quick Start Guide

## Installation

```bash
pip install -r requirements.txt
```

## Basic Usage

```python
from rag_fca_system import RAGFCASystem

# Initialize the system
system = RAGFCASystem()

# Process an optimization problem
problem = """
We have a reconfigurable photovoltaic energy storage system composed of 
multiple parallel battery modules and series columns that need optimization.
"""

result = system.process_optimization_problem(problem)

# View the formal description
import json
print(json.dumps(result, indent=2))
```

## Output Example

For the photovoltaic battery problem, the system generates:

**Decision Variables:**
- `x_ipt`: Binary variables for module activation (3D: modules × parallel branches × time periods)

**Objective:**
- Minimize total energy loss: ∑_{t=1}^{T} E_loss(t)

**Constraints:**
1. **Parallel Consistency** (implicit): All parallel branches must activate the same number of modules
2. **Voltage Range** (explicit): System voltage must stay within [V_min, V_max]
3. **Current Limit** (explicit): Module current ≤ I_max
4. **Demand Satisfaction** (implicit): Total power must meet demand

**Parameters:**
- N_modules, N_parallel, T (dimensions)
- V_i, V_min, V_max (voltage specs)
- I_max (current limit)
- P_demand(t), E_loss(t) (functions)

## Using Claude API

```python
from api_integration_example import RAGFCASystemWithAPI
import os

# Set API key
os.environ["ANTHROPIC_API_KEY"] = "your-key-here"

# Initialize with API
system = RAGFCASystemWithAPI()

# Process with real LLM
result = system.process_optimization_problem(problem)
```

## Key Features

1. **Automatic attribute extraction** from natural language
2. **Semantic disambiguation** using concept lattices
3. **Implicit constraint discovery** via rough sets
4. **Structured JSON output** ready for optimization solvers

## Files

- `rag_fca_system.py` - Main implementation
- `api_integration_example.py` - API integration examples
- `knowledge_base.json` - Pre-built knowledge base
- `README.md` - Full documentation

## Next Steps

1. Try with your own optimization problems
2. Extend the knowledge base with new problem domains
3. Integrate with your optimization solver
4. Customize attribute extraction for your domain
