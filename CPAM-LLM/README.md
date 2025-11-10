# CPAM-LLM: Constraint Programming Automated Modeling with Large Language Models

## Overview

This repository contains the implementation of CPAM-LLM, an automated constraint programming modeling system that transforms natural language problem descriptions into executable solver code. The system leverages Large Language Models (LLMs) enhanced with Retrieval-Augmented Generation via Formal Concept Analysis (RAG-FCA) to automate the creation of constraint programming models for optimization problems.

## System Architecture

The CPAM-LLM method consists of four core functional modules operating in sequence:

### 1. Natural Language Structuring Module

**Purpose**: Transform unstructured natural language problem descriptions into structured optimization components.

**Key Functions**:
- **Semantic Parsing**: Extract core problem elements (decision variables, constraints, objectives)
- **RAG-FCA Knowledge Augmentation**: Disambiguate natural language using domain knowledge retrieval
  - Queries pre-constructed knowledge base for standardized terminology
  - Retrieves constraint templates and historical cases
  - Resolves ambiguities and implicit constraints
- **Structured Output**: Produces standardized problem element descriptions

### 2. Mathematical Model Generation Module

**Purpose**: Convert structured natural language elements into formalized mathematical models.

**Key Functions**:
- **Entity Abstraction**: Identify and abstract mathematical entities (sets, parameters, variables)
- **Formal Mapping**: Transform decision variables and constraints into mathematical expressions
- **Objective Function Encoding**: Formalize optimization objectives with proper directionality
- **Model Organization**: Structure all components into a rigorous, hierarchical mathematical model

### 3. Solver Code Generation Module

**Purpose**: Translate formal mathematical models into executable solver code (CPLEX CP Optimizer).

**Key Functions**:
- **Code Framework**: Generate code structure with library imports and model initialization
- **Data Integration**: Create data loading and parameterization code
- **Logic Mapping**: Instantiate variables and map constraints to solver API
- **Execution & Interpretation**: Automatically invoke solver and parse results

### 4. Validation and Iterative Optimization Module

**Purpose**: Ensure correctness and robustness through validation and feedback-driven improvement.

**Key Functions**:
- **Double Inspection**: Validate both model construction and code generation
  - Semantic alignment analysis between model and description
  - Static code analysis for syntax and functional compliance
  - Dynamic error interpretation during solver execution
- **Iterative Optimization**: Record and learn from each iteration
  - Maintain data pool of high-quality samples
  - Drive continuous capability evolution through model fine-tuning

## RAG-FCA System Implementation

The `rag_fca_system.py` file implements the Retrieval-Augmented Generation via Formal Concept Analysis mechanism used in the Natural Language Structuring Module.

### Core Components

**Formal Concept Analysis (FCA)**:
- `FormalBackground`: Represents knowledge base K = (G, M, I) where G is optimization problems, M is problem characteristics, I is their relation
- `ConceptLattice`: Builds lattice structure L(K) = (C(K), ≤) for hierarchical knowledge organization
- `FormalConcept`: Represents concepts as (objects, attributes) pairs with partial ordering

**Rough Set Theory**:
- `RoughSetProcessor`: Handles incomplete information and implicit constraint discovery
- Computes lower/upper approximations for uncertain knowledge
- Discovers certain and possible constraints from incomplete queries

**Main System**:
- `RAGFCASystem`: Integrates FCA and rough sets for problem processing
- Semantic feature extraction from natural language
- Knowledge base querying and retrieval
- Attribute augmentation for better understanding
- Supports incremental learning and knowledge base updates

### Knowledge Base Structure

The system maintains a knowledge base of optimization problems with:
- **Objects**: Problem instances (scheduling, assignment, routing, etc.)
- **Attributes**: Problem characteristics (binary variables, time constraints, resource limits, etc.)
- **Relations**: Fuzzy membership degrees indicating attribute relevance to problems

### Processing Flow

1. **Input**: Natural language problem description
2. **Feature Extraction**: Identify problem characteristics using semantic analysis
3. **Knowledge Retrieval**: Query concept lattice for similar problems
4. **Augmentation**: Add implicit constraints and standardized terminology
5. **Output**: Structured formal problem description with decision variables, constraints, and objectives

## Dataset Generation System

The `DATASET.html` file provides a web-based interface for building training datasets with chaotic mapping-based augmentation:
- **Base Dataset Management**: Create and manage problem-model-code sample triplets
- **SFT Data Generation**: Generate supervised fine-tuning data with variations
- **DPO Data Creation**: Build preference pairs for direct preference optimization
- **LLM Integration**: Optional style transformation using language models

## Usage Flow

```
Natural Language Problem
         ↓
[Natural Language Structuring] → RAG-FCA Enhancement
         ↓
Structured Problem Elements
         ↓
[Mathematical Model Generation] → Formal CP Model
         ↓
Mathematical Model
         ↓
[Solver Code Generation] → CPLEX Code
         ↓
Executable Solver Code
         ↓
[Validation & Optimization] → Iterative Refinement
         ↓
Verified Solution
```

## Key Features

- **End-to-End Automation**: Complete pipeline from natural language to executable code
- **Knowledge-Enhanced Understanding**: RAG-FCA mechanism for disambiguation and constraint discovery
- **Dual Validation**: Model-level and code-level verification
- **Iterative Learning**: Continuous improvement through feedback loops
- **Fuzzy Knowledge Handling**: Support for uncertain and incomplete problem descriptions
- **Incremental Knowledge Base**: Expandable domain knowledge repository

## Technical Implementation

**Mathematical Foundation**:
- Formal Concept Analysis (FCA) for knowledge organization
- Rough Set Theory for incomplete information handling
- Fuzzy membership for uncertainty representation

**LLM Integration**:
- Fine-tuned models for constraint programming domain
- Claude API integration for semantic understanding
- Structured output generation in JSON format

**Solver Support**:
- Primary target: CPLEX CP Optimizer
- Extensible to other constraint programming solvers