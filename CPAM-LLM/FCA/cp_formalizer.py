#!/usr/bin/env python3
"""
Formalization of Constraint Programming Problems using RAG-FCA with Reference Examples

This program transforms natural language descriptions of CP problems into 
formalized descriptions using 5 reference examples as templates via LLM API.
"""

import pandas as pd
import json
import re
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import requests
import time

class FormalConceptAnalysis:
    """
    Implements Formal Concept Analysis for organizing optimization problems
    into concept lattices.
    """
    
    def __init__(self):
        self.objects = set()
        self.attributes = set()
        self.relations = defaultdict(set)
        self.concepts = []
        
    def add_object(self, obj: str, attrs: Set[str]):
        """Add an object with its attributes to the formal context."""
        self.objects.add(obj)
        self.attributes.update(attrs)
        self.relations[obj] = attrs
    
    def compute_closure(self, attr_set: Set[str]) -> Set[str]:
        """Compute B' = {g in G | for all m in B, (g,m) in I}"""
        if not attr_set:
            return self.objects.copy()
        
        result = self.objects.copy()
        for attr in attr_set:
            result = {obj for obj in result if attr in self.relations[obj]}
        return result
    
    def compute_intent(self, obj_set: Set[str]) -> Set[str]:
        """Compute A' = {m in M | for all g in A, (g,m) in I}"""
        if not obj_set:
            return self.attributes.copy()
        
        result = self.attributes.copy()
        for obj in obj_set:
            result = result.intersection(self.relations[obj])
        return result
    
    def extract_concepts(self):
        """Extract formal concepts from the formal context."""
        self.concepts = []
        processed_attrs = set()
        
        for attr in self.attributes:
            attr_set = {attr}
            obj_set = self.compute_closure(attr_set)
            intent = self.compute_intent(obj_set)
            
            concept_key = frozenset(intent)
            if concept_key not in processed_attrs:
                self.concepts.append((obj_set, intent))
                processed_attrs.add(concept_key)
        
        return self.concepts


class LLMClient:
    """Client for interacting with LLM API for text generation."""
    
    def __init__(self, api_base: str, api_key: str = "Your key"):
        self.api_base = api_base
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    
    def generate(self, prompt: str, max_tokens: int = 3500, temperature: float = 0.7) -> str:
        """Generate text using the LLM API."""
        url = f"{self.api_base}/chat/completions"
        
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are an expert in constraint programming and optimization problem formalization. You create precise, structured formal descriptions."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            print(f"  API Error: {str(e)}")
            return ""


class ProblemFormalizer:
    """
    Main class for formalizing CP problem descriptions using reference examples.
    """
    
    def __init__(self, api_base: str, reference_examples: Dict[str, Dict]):
        self.llm = LLMClient(api_base)
        self.fca = FormalConceptAnalysis()
        self.reference_examples = reference_examples
        
        # Knowledge base of problem types
        self.problem_types = {
            "Aircraft Skin Processing": {
                "domain": "manufacturing",
                "reference_idx": 0,
                "attributes": ["job_shop_scheduling", "flexible_machines", "temporal_constraints", 
                              "makespan_minimization", "sequence_dependent", "machine_capacity"]
            },
            "Battery Pack Design": {
                "domain": "energy_systems",
                "reference_idx": 61,
                "attributes": ["module_configuration", "voltage_constraints", "energy_capacity",
                              "thermal_management", "reliability_optimization", "series_parallel_topology"]
            },
            "Charging Station Location": {
                "domain": "facility_location",
                "reference_idx": 77,
                "attributes": ["demand_coverage", "capacity_constraints", "distance_minimization",
                              "fixed_costs", "service_quality", "urban_infrastructure"]
            },
            "DNA Sequence Design": {
                "domain": "molecular_biology",
                "reference_idx": 108,
                "attributes": ["sequence_constraints", "gc_content", "base_pair_complementarity",
                              "word_list_selection", "biochemical_feasibility", "constraint_satisfaction"]
            },
            "VRP": {
                "domain": "logistics",
                "reference_idx": 124,
                "attributes": ["vehicle_routing", "capacity_constraints", "distance_minimization",
                              "time_windows", "multi_depot", "fleet_management"]
            }
        }
    
    def extract_key_attributes(self, natural_language: str, problem_type: str, 
                               constraint_type: str) -> Set[str]:
        """Extract key attributes from natural language description."""
        base_attrs = set(self.problem_types.get(problem_type, {}).get("attributes", []))
        
        # Extract additional attributes from text
        text_lower = natural_language.lower()
        
        if "capacity" in text_lower:
            base_attrs.add("capacity_constraint")
        if "time window" in text_lower or "temporal" in text_lower:
            base_attrs.add("temporal_constraint")
        if "precedence" in text_lower:
            base_attrs.add("precedence_relations")
        if "makespan" in text_lower:
            base_attrs.add("makespan_objective")
        if "distance" in text_lower:
            base_attrs.add("distance_objective")
        if "cost" in text_lower:
            base_attrs.add("cost_minimization")
        if "resource" in text_lower:
            base_attrs.add("resource_constraint")
        if "sequence" in text_lower:
            base_attrs.add("sequence_dependent")
        
        if constraint_type != "base":
            base_attrs.add(f"constraint_type_{constraint_type}")
        
        return base_attrs
    
    def select_reference_example(self, problem_type: str) -> Optional[Dict]:
        """Select the most appropriate reference example based on problem type."""
        type_info = self.problem_types.get(problem_type, {})
        ref_idx = type_info.get("reference_idx")
        
        if ref_idx is not None and ref_idx in self.reference_examples:
            return self.reference_examples[ref_idx]
        
        # Fallback to any available reference
        if self.reference_examples:
            return list(self.reference_examples.values())[0]
        
        return None
    
    def extract_additional_constraint_info(self, natural_language: str, constraint_type: str) -> str:
        """Extract information about additional constraints from natural language."""
        if constraint_type == "base":
            return ""
        
        text_lower = natural_language.lower()
        constraint_info = []
        
        # Common constraint patterns
        if "time window" in text_lower or "must start between" in text_lower or "must be completed between" in text_lower:
            # Extract time window information
            time_pattern = re.search(r'(start|complete|finish|end).*?between.*?(\d+).*?(\d+)', text_lower)
            if time_pattern:
                constraint_info.append(f"Temporal window constraint")
        
        if "must not exceed" in text_lower or "maximum" in text_lower or "at most" in text_lower:
            constraint_info.append("Upper bound constraint")
        
        if "must be at least" in text_lower or "minimum" in text_lower or "at least" in text_lower:
            constraint_info.append("Lower bound constraint")
        
        if "exactly" in text_lower or "must equal" in text_lower:
            constraint_info.append("Equality constraint")
        
        # Job/operation specific constraints
        if "job" in text_lower and any(word in text_lower for word in ["must", "should", "required"]):
            constraint_info.append("Job-specific constraint")
        
        if "operation" in text_lower and any(word in text_lower for word in ["must", "should", "required"]):
            constraint_info.append("Operation-specific constraint")
        
        return " - Additional constraints" if constraint_info else ""
    
    def extract_dataset_description(self, cp_code: str, problem_type: str) -> str:
        """Extract dataset format and structure from CP code."""
        if not cp_code or pd.isna(cp_code):
            return ""
        
        dataset_lines = []
        
        # Find file paths
        file_pattern = re.search(r'filename\s*=.*["\'](.+?\.[\w]+)["\']', cp_code)
        if file_pattern:
            filepath = file_pattern.group(1)
            dataset_lines.append(f'The input file "{filepath}" follows this format:')
        
        # Format-specific descriptions
        if ".fjs" in cp_code:
            dataset_lines.append("First line: Two integers representing the number of jobs and number of machines.")
            dataset_lines.append("Subsequent lines (one per job): Number of operations, followed by operation details including the number of candidate machines, machine-duration pairs (machine ID and processing time), and temporal constraint type ('tight', 'loose', or 'none') after each operation.")
        elif "read_excel" in cp_code or ".xlsx" in cp_code:
            dataset_lines.append("Excel spreadsheet containing structured tabular data with columns for problem parameters.")
        elif ".csv" in cp_code:
            dataset_lines.append("CSV file with comma-separated values representing problem data in tabular format.")
        elif ".dat" in cp_code or ".txt" in cp_code:
            dataset_lines.append("Plain text data file with problem parameters formatted according to standard conventions.")
        
        if dataset_lines:
            return "\n".join(dataset_lines)
        
        return ""
    
    def clean_generated_description(self, text: str) -> str:
        """Clean up the generated description by removing unwanted content."""
        # Remove any solver instructions
        text = re.sub(r'Solver Instruction:.*?(?=\n\n|\Z)', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'Write a Python program.*?(?=\n\n|\Z)', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove separator lines
        text = re.sub(r'={40,}', '', text)
        text = re.sub(r'-{40,}', '', text)
        
        # Clean up extra whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        
        return text
    
    def generate_formal_description_with_llm(self, row: pd.Series, reference: Dict) -> str:
        """Generate formal description using LLM with reference example."""
        
        natural_language = row['CP natural language']
        problem_type = row['Problem types']
        constraint_type = row['Constraint']
        cp_code = row.get('CP code', '')
        
        # Extract dataset description
        dataset_desc = self.extract_dataset_description(cp_code, problem_type)
        
        # Prepare reference example (clean it first)
        clean_reference = self.clean_generated_description(reference['formal_language'])
        
        # Build comprehensive prompt
        prompt = f"""You are creating a formal structured description for a constraint programming problem.

REFERENCE EXAMPLE - Follow this EXACT format and style:
{'='*80}
Problem Type: {reference['problem_type']}
Constraint: {reference['constraint']}

{clean_reference}
{'='*80}

NEW PROBLEM TO FORMALIZE:
Problem Type: {problem_type}
Constraint Category: {constraint_type}

Natural Language Description:
{natural_language}

CRITICAL FORMATTING REQUIREMENTS:
1. Start with "1. Decision Variables"
2. Follow with "2. Parameters and Input Data"
3. Then "3. Constraints"
4. Then "4. Objective Function"
5. If there is dataset information, add "5. Data Specification"
6. DO NOT include any "Solver Instruction" or code generation instructions
7. DO NOT include separator lines (===)

CONTENT REQUIREMENTS:
- Write in the EXACT same style as the reference example
- Use similar level of technical detail and precision
- For "2. Parameters and Input Data": Include ONLY static input data and problem parameters (numbers, dimensions, matrices, etc.)
- For "3. Constraints": Include ALL constraint requirements, including:
  * Base problem constraints (from the problem definition)
  * Additional constraints specific to this variant (if constraint is not "base")
  * Any temporal windows, bounds, or special conditions
- If constraint category is "{constraint_type}" (not "base"), identify the additional constraint from the natural language and add it to section 3
- DO NOT put constraint descriptions in section 2
- Use colons after subsection headers (e.g., "Alternative constraint: description")
- End the description after section 4 or 5 - do not add solver instructions

{f"DATASET INFORMATION:{chr(10)}{dataset_desc}" if dataset_desc else ""}

Generate ONLY the formal description sections (1-5), nothing else:"""

        formal_desc = self.llm.generate(prompt, max_tokens=3500, temperature=0.5)
        
        if not formal_desc:
            return self.generate_template_based_description(row)
        
        # Clean up the response
        formal_desc = self.clean_generated_description(formal_desc)
        
        # Ensure it starts with "1. Decision Variables"
        if not formal_desc.startswith("1."):
            # Try to find and extract the formal description
            match = re.search(r'(1\.\s*Decision Variables.*)', formal_desc, re.DOTALL)
            if match:
                formal_desc = match.group(1)
        
        # Final cleanup
        formal_desc = self.clean_generated_description(formal_desc)
        
        return formal_desc
    
    def generate_template_based_description(self, row: pd.Series) -> str:
        """Fallback template-based generation if API fails."""
        
        problem_type = row['Problem types']
        constraint_type = row['Constraint']
        natural_language = row['CP natural language']
        cp_code = row.get('CP code', '')
        
        # Get reference for structure
        reference = self.select_reference_example(problem_type)
        
        description = f"""1. Decision Variables
Decision variables representing the core decision elements of the {problem_type} problem, controlling the assignment, allocation, or scheduling decisions required for the solution.

2. Parameters and Input Data
Input parameters including problem dimensions, capacity limits, cost/distance matrices, time-related parameters, and other fixed constants that define the specific problem instance to be solved.

3. Constraints
Key constraints ensuring solution validity:
- Structural constraints defining the valid solution space
- Resource capacity constraints limiting available resources
- Logical constraints maintaining solution consistency
- Domain-specific constraints particular to {problem_type}"""

        if constraint_type != "base":
            description += f"\n- Additional constraints for variant {constraint_type}"
        
        description += f"""

4. Objective Function
Optimization objective function to be minimized or maximized while satisfying all problem constraints."""
        
        # Add dataset info if available
        dataset_desc = self.extract_dataset_description(cp_code, problem_type)
        if dataset_desc:
            description += f"""

5. Data Specification
{dataset_desc}"""
        
        return description
    
    def generate_formalized_description(self, row: pd.Series) -> str:
        """Generate comprehensive formalized description for a single problem."""
        
        natural_language = row['CP natural language']
        problem_type = row['Problem types']
        constraint_type = row['Constraint']
        cp_code = row.get('CP code', '')
        
        # Extract attributes for FCA
        attributes = self.extract_key_attributes(natural_language, problem_type, constraint_type)
        problem_id = f"{problem_type}_{constraint_type}"
        self.fca.add_object(problem_id, attributes)
        
        # Check if this is a reference example (already has formal language)
        if 'CP formal language' in row and pd.notna(row['CP formal language']):
            # Clean the reference example too
            return self.clean_generated_description(row['CP formal language'])
        
        # Select appropriate reference example
        reference = self.select_reference_example(problem_type)
        
        # Generate using LLM with reference
        if reference:
            formal_desc = self.generate_formal_description_with_llm(row, reference)
        else:
            formal_desc = self.generate_template_based_description(row)
        
        return formal_desc


def load_reference_examples(df: pd.DataFrame, reference_indices: List[int]) -> Dict[int, Dict]:
    """Load reference examples from the dataframe."""
    references = {}
    
    for idx in reference_indices:
        if idx < len(df) and pd.notna(df.loc[idx, 'CP formal language']):
            references[idx] = {
                'problem_type': df.loc[idx, 'Problem types'],
                'constraint': df.loc[idx, 'Constraint'],
                'natural_language': df.loc[idx, 'CP natural language'],
                'formal_language': df.loc[idx, 'CP formal language']#,
                #'cp_code': df.loc[idx, 'CP code'] if pd.notna(df.loc[idx, 'CP code']) else ''
            }
    
    return references


def main():
    """Main execution function."""
    
    # Configuration
    API_BASE = "https://api.chatanywhere.tech/v1"
    INPUT_FILE = "CP_finetuning.xlsx"
    OUTPUT_FILE = "CP_formalized_output.xlsx"
    SUMMARY_FILE = "formalization_summary.txt"
    
    # Reference example indices (0-indexed)
    REFERENCE_INDICES = [289, 290, 291, 292, 293] #[0, 61, 77, 108, 124]  # Problems 1, 62, 78, 109, 125
    
    print("="*80)
    print("Constraint Programming Problem Formalization using RAG-FCA")
    print("With Reference Examples")
    print("="*80)
    print()
    
    # Load dataset
    print(f"Loading dataset from: {INPUT_FILE}")
    df = pd.read_excel(INPUT_FILE)
    print(f"Loaded {len(df)} problems")
    print()
         # ==========================================
    # TEST mode: only process a subset for testing
    # ==========================================
    TEST_COUNT = 5
    print(f"\n[TEST MODE] Only processing the first {TEST_COUNT} records for testing...")
    df = df.head(TEST_COUNT)
    # ==========================================
    #    
    # Load reference examples
    print("Loading reference examples...")
    reference_examples = load_reference_examples(df, REFERENCE_INDICES)
    print(f"Loaded {len(reference_examples)} reference examples:")
    for idx, ref in reference_examples.items():
        print(f"  - Problem {idx+1}: {ref['problem_type']} - {ref['constraint']}")
    print()
    
    # Initialize formalizer
    print("Initializing Formal Concept Analysis framework...")
    formalizer = ProblemFormalizer(API_BASE, reference_examples)
    print()
    
    # Process problems
    formalized_descriptions = []
    
    print("Processing problems and generating formalized descriptions...")
    print("(Note: API calls may take time, ~2-3 seconds per problem)")
    print()
    
    for idx, row in df.iterrows():
        problem_num = idx + 1
        problem_type = row['Problem types']
        constraint = row['Constraint']
        
        # Check if already has formal language
        if pd.notna(row.get('CP formal language', None)):
            print(f"Problem {problem_num}/{len(df)}: {problem_type} - {constraint} [Reference example]")
            # Clean even the reference examples
            cleaned = formalizer.clean_generated_description(row['CP formal language'])
            formalized_descriptions.append(cleaned)
        else:
            print(f"Problem {problem_num}/{len(df)}: {problem_type} - {constraint} [Generating...]")
            
            try:
                formalized = formalizer.generate_formalized_description(row)
                formalized_descriptions.append(formalized)
                
                # Rate limiting to avoid API throttling
                if idx < len(df) - 1 and idx not in REFERENCE_INDICES:
                    time.sleep(1.5)
                    
            except Exception as e:
                print(f"  Error: {str(e)}")
                formalized_descriptions.append(f"Error in formalization: {str(e)}")
                time.sleep(2)
    
    # Add formalized descriptions to dataframe
    df['CP formal language generated'] = formalized_descriptions
    
    # Build concept lattice
    print()
    print("Building concept lattice from formal context...")
    formalizer.fca.extract_concepts()
    print(f"Extracted {len(formalizer.fca.concepts)} formal concepts")
    print()
    
    # Save results
    print(f"Saving results to: {OUTPUT_FILE}")
    df.to_excel(OUTPUT_FILE, index=False)
    
    # Generate summary report
    summary = generate_summary_report(df, formalizer.fca, reference_examples)
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print()
    print("="*80)
    print("Formalization Complete!")
    print("="*80)
    print(f"Output files:")
    print(f"  - Formalized problems: {OUTPUT_FILE}")
    print(f"  - Summary report: {SUMMARY_FILE}")
    print()
    
    # Display sample
    print("Sample generated formal description (first non-reference problem):")
    print("-"*80)
    for idx in range(len(df)):
        if idx not in REFERENCE_INDICES:
            print(f"Problem {idx+1}: {df.loc[idx, 'Problem types']} - {df.loc[idx, 'Constraint']}")
            print()
            sample_text = formalized_descriptions[idx][:1200] if len(formalized_descriptions[idx]) > 1200 else formalized_descriptions[idx]
            print(sample_text)
            if len(formalized_descriptions[idx]) > 1200:
                print("...")
            print("-"*80)
            break


def generate_summary_report(df: pd.DataFrame, fca: FormalConceptAnalysis, 
                           reference_examples: Dict) -> str:
    """Generate a summary report of the formalization process."""
    
    report = """
===============================================================================
RAG-FCA Formalization Summary Report
With Reference Example-Based Generation
===============================================================================

1. DATASET OVERVIEW
-------------------
"""
    report += f"Total problems processed: {len(df)}\n"
    report += f"Problem types: {df['Problem types'].nunique()}\n"
    report += f"Constraint categories: {df['Constraint'].nunique()}\n\n"
    
    report += "Problem type distribution:\n"
    for ptype, count in df['Problem types'].value_counts().items():
        report += f"  - {ptype}: {count}\n"
    
    report += "\n2. REFERENCE EXAMPLES USED\n"
    report += "-"*40 + "\n"
    report += f"Number of reference examples: {len(reference_examples)}\n\n"
    
    for idx, ref in reference_examples.items():
        report += f"Reference {idx+1}: {ref['problem_type']} - {ref['constraint']}\n"
        report += f"  Formal description length: {len(ref['formal_language'])} characters\n"
    
    report += "\n3. FORMAL CONCEPT ANALYSIS RESULTS\n"
    report += "-"*40 + "\n"
    report += f"Total objects (problem instances): {len(fca.objects)}\n"
    report += f"Total attributes: {len(fca.attributes)}\n"
    report += f"Formal concepts extracted: {len(fca.concepts)}\n\n"
    
    report += "Sample attributes identified:\n"
    for attr in sorted(list(fca.attributes))[:20]:
        report += f"  - {attr}\n"
    
    report += "\n4. CONCEPT LATTICE STRUCTURE\n"
    report += "-"*40 + "\n"
    report += "Top-level concepts (most general):\n"
    
    sorted_concepts = sorted(fca.concepts, key=lambda x: len(x[0]), reverse=True)[:5]
    for i, (objs, attrs) in enumerate(sorted_concepts, 1):
        report += f"\nConcept {i}:\n"
        report += f"  Objects: {len(objs)}\n"
        if objs:
            report += f"  Sample objects: {', '.join(sorted(list(objs))[:3])}\n"
        if attrs:
            report += f"  Attributes: {', '.join(sorted(list(attrs))[:5])}\n"
    
    report += "\n5. GENERATION STATISTICS\n"
    report += "-"*40 + "\n"
    
    has_original = df['CP formal language'].notna().sum()
    newly_generated = len(df) - has_original
    
    report += f"Problems with existing formal language: {has_original}\n"
    report += f"Problems with newly generated descriptions: {newly_generated}\n"
    
    if 'CP formal language generated' in df.columns:
        avg_length = df['CP formal language generated'].str.len().mean()
        report += f"Average description length: {avg_length:.0f} characters\n"
    
    report += "\n6. METHODOLOGY\n"
    report += "-"*40 + "\n"
    report += """
This formalization uses a reference-example-based approach:

1. Reference Selection: For each problem, select the most appropriate reference
   example from the same problem type category.

2. LLM-Based Generation: Use large language model API to generate formal
   descriptions following the structure and style of the reference example.

3. Structured Format: All descriptions follow a consistent format:
   - 1. Decision Variables
   - 2. Parameters and Input Data (ONLY static input parameters)
   - 3. Constraints (ALL constraints including additional ones)
   - 4. Objective Function
   - 5. Data Specification (when applicable)

4. FCA Integration: Build concept lattice from problem attributes to organize
   the knowledge hierarchically.

5. Quality Assurance: Maintain consistency with reference examples while
   adapting to specific problem variations.

6. Cleanup: Remove solver instructions and unnecessary formatting.

===============================================================================
End of Report
===============================================================================
"""
    
    return report


if __name__ == "__main__":
    main()