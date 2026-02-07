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
