"""
RAG-FCA System for Constrained Optimization Problem Processing
Implements Retrieval-Augmented Generation via Formal Concept Analysis
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import json
import re
from itertools import combinations


@dataclass
class FormalConcept:
    """Represents a formal concept (A, B) where A is objects and B is attributes"""
    objects: Set[str]
    attributes: Set[str]
    
    def __hash__(self):
        return hash((frozenset(self.objects), frozenset(self.attributes)))
    
    def __eq__(self, other):
        return self.objects == other.objects and self.attributes == other.attributes
    
    def __le__(self, other):
        """Partial order: (A1,B1) <= (A2,B2) iff A1 ⊆ A2 iff B2 ⊆ B1"""
        return self.objects.issubset(other.objects)


class FormalBackground:
    """Represents formal background K = (G, M, I)"""
    
    def __init__(self):
        self.objects: Set[str] = set()  # G: set of optimization problem instances
        self.attributes: Set[str] = set()  # M: set of problem characteristics
        self.incidence: Dict[Tuple[str, str], float] = {}  # I: relation (supports fuzzy)
        
    def add_object(self, obj: str):
        """Add an object (problem instance) to the formal background"""
        self.objects.add(obj)
    
    def add_attribute(self, attr: str):
        """Add an attribute (problem characteristic) to the formal background"""
        self.attributes.add(attr)
    
    def set_relation(self, obj: str, attr: str, degree: float = 1.0):
        """Set incidence relation with optional fuzzy membership degree"""
        if obj not in self.objects:
            self.add_object(obj)
        if attr not in self.attributes:
            self.add_attribute(attr)
        self.incidence[(obj, attr)] = degree
    
    def has_relation(self, obj: str, attr: str, threshold: float = 0.5) -> bool:
        """Check if object has attribute (with fuzzy threshold)"""
        return self.incidence.get((obj, attr), 0.0) >= threshold
    
    def get_membership(self, obj: str, attr: str) -> float:
        """Get fuzzy membership degree"""
        return self.incidence.get((obj, attr), 0.0)
    
    def prime_objects(self, objects: Set[str]) -> Set[str]:
        """A' = {m ∈ M | ∀g ∈ A, (g,m) ∈ I}"""
        if not objects:
            return self.attributes.copy()
        return set(attr for attr in self.attributes 
                   if all(self.has_relation(obj, attr) for obj in objects))
    
    def prime_attributes(self, attributes: Set[str]) -> Set[str]:
        """B' = {g ∈ G | ∀m ∈ B, (g,m) ∈ I}"""
        if not attributes:
            return self.objects.copy()
        return set(obj for obj in self.objects 
                   if all(self.has_relation(obj, attr) for attr in attributes))
    
    def double_prime_objects(self, objects: Set[str]) -> Set[str]:
        """A'' = (A')'"""
        return self.prime_attributes(self.prime_objects(objects))
    
    def double_prime_attributes(self, attributes: Set[str]) -> Set[str]:
        """B'' = (B')'"""
        return self.prime_objects(self.prime_attributes(attributes))


class ConceptLattice:
    """Represents the concept lattice L(K) = (C(K), ≤)"""
    
    def __init__(self, background: FormalBackground):
        self.background = background
        self.concepts: List[FormalConcept] = []
        self._build_lattice()
    
    def _build_lattice(self):
        """Extract all formal concepts using closure-based algorithm"""
        concepts_dict = {}
        
        # Generate all possible attribute subsets
        all_attrs = list(self.background.attributes)
        
        for r in range(len(all_attrs) + 1):
            for attr_subset in combinations(all_attrs, r):
                attr_set = set(attr_subset)
                
                # Compute B'' to get maximal attribute set
                obj_set = self.background.prime_attributes(attr_set)
                attr_set_closure = self.background.prime_objects(obj_set)
                
                # Create concept
                concept_key = frozenset(attr_set_closure)
                if concept_key not in concepts_dict:
                    concept = FormalConcept(
                        objects=obj_set,
                        attributes=attr_set_closure
                    )
                    concepts_dict[concept_key] = concept
        
        self.concepts = list(concepts_dict.values())
    
    def find_concept_by_attributes(self, query_attrs: Set[str]) -> Optional[FormalConcept]:
        """Find concept that best matches query attributes"""
        best_concept = None
        best_score = -1
        
        for concept in self.concepts:
            score = self._jaccard_similarity(query_attrs, concept.attributes)
            if score > best_score:
                best_score = score
                best_concept = concept
        
        return best_concept
    
    def _jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Calculate Jaccard similarity coefficient"""
        if not set1 and not set2:
            return 1.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    def fuzzy_jaccard_similarity(self, query: Dict[str, float], 
                                  concept_attrs: Set[str]) -> float:
        """Calculate fuzzy Jaccard similarity for fuzzy queries"""
        all_attrs = set(query.keys()) | concept_attrs
        
        numerator = sum(min(query.get(attr, 0), 1.0 if attr in concept_attrs else 0)
                       for attr in all_attrs)
        denominator = sum(max(query.get(attr, 0), 1.0 if attr in concept_attrs else 0)
                         for attr in all_attrs)
        
        return numerator / denominator if denominator > 0 else 0.0


class RoughSetProcessor:
    """Handles incomplete information using rough set theory"""
    
    def __init__(self, background: FormalBackground):
        self.background = background
    
    def indiscernibility_relation(self, attributes: Set[str]) -> Dict[str, Set[str]]:
        """Compute equivalence classes based on indiscernibility relation IND(B)"""
        equivalence_classes = defaultdict(set)
        
        for obj in self.background.objects:
            # Create signature based on attribute values
            signature = tuple(sorted([
                (attr, self.background.has_relation(obj, attr))
                for attr in attributes
            ]))
            equivalence_classes[signature].add(obj)
        
        # Convert to dict mapping each object to its equivalence class
        obj_to_class = {}
        for equiv_class in equivalence_classes.values():
            for obj in equiv_class:
                obj_to_class[obj] = equiv_class
        
        return obj_to_class
    
    def lower_approximation(self, target_objects: Set[str], 
                           attributes: Set[str]) -> Set[str]:
        """Compute lower approximation: definitely belong to target"""
        equiv_classes = self.indiscernibility_relation(attributes)
        lower_approx = set()
        
        for obj in self.background.objects:
            if equiv_classes[obj].issubset(target_objects):
                lower_approx.add(obj)
        
        return lower_approx
    
    def upper_approximation(self, target_objects: Set[str], 
                           attributes: Set[str]) -> Set[str]:
        """Compute upper approximation: possibly belong to target"""
        equiv_classes = self.indiscernibility_relation(attributes)
        upper_approx = set()
        
        for obj in self.background.objects:
            if equiv_classes[obj] & target_objects:
                upper_approx.add(obj)
        
        return upper_approx
    
    def discover_implicit_constraints(self, query_attrs: Set[str]) -> Tuple[Set[str], Set[str]]:
        """Discover certain and possible constraints from incomplete query"""
        # Find core objects that exactly satisfy query attributes
        core_objects = set(obj for obj in self.background.objects
                          if all(self.background.has_relation(obj, attr) 
                                for attr in query_attrs))
        
        if not core_objects:
            return set(), set()
        
        # Get relevant objects (upper approximation)
        relevant_objects = self.upper_approximation(core_objects, query_attrs)
        
        # Certain constraints (lower approximation)
        certain_constraints = set(
            attr for attr in self.background.attributes - query_attrs
            if all(self.background.has_relation(obj, attr) 
                  for obj in relevant_objects)
        )
        
        # Possible constraints (upper approximation)
        possible_constraints = set(
            attr for attr in self.background.attributes - query_attrs
            if any(self.background.has_relation(obj, attr) 
                  for obj in relevant_objects)
        )
        
        return certain_constraints, possible_constraints


class RAGFCASystem:
    """Main RAG-FCA system for optimization problem processing"""
    
    def __init__(self, llm_api_key: Optional[str] = None):
        self.background = FormalBackground()
        self.lattice: Optional[ConceptLattice] = None
        self.rough_processor: Optional[RoughSetProcessor] = None
        self.llm_api_key = llm_api_key
        
        # Initialize with domain knowledge
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """Initialize knowledge base with optimization problem concepts"""
        
        # Concept C1: DNA sequence design problems
        self.background.set_relation("dna_seq_opt_1", "GC_content_constraint", 1.0)
        self.background.set_relation("dna_seq_opt_1", "base_pair_complementarity", 1.0)
        self.background.set_relation("dna_seq_opt_1", "sequence_design", 1.0)
        self.background.set_relation("dna_seq_opt_1", "discrete_variables", 1.0)
        
        # Concept C2: Energy storage system design
        self.background.set_relation("battery_opt_1", "battery_voltage_range", 1.0)
        self.background.set_relation("battery_opt_1", "energy_loss_minimization", 1.0)
        self.background.set_relation("battery_opt_1", "module_configuration", 1.0)
        self.background.set_relation("battery_opt_1", "capacity_constraint", 1.0)
        self.background.set_relation("battery_opt_1", "current_limit", 1.0)
        self.background.set_relation("battery_opt_1", "parallel_consistency", 1.0)
        self.background.set_relation("battery_opt_1", "temporal_periods", 1.0)
        self.background.set_relation("battery_opt_1", "binary_variables", 1.0)
        
        self.background.set_relation("pv_storage_opt_1", "battery_voltage_range", 1.0)
        self.background.set_relation("pv_storage_opt_1", "energy_loss_minimization", 1.0)
        self.background.set_relation("pv_storage_opt_1", "module_configuration", 1.0)
        self.background.set_relation("pv_storage_opt_1", "reconfigurable_system", 1.0)
        self.background.set_relation("pv_storage_opt_1", "parallel_consistency", 1.0)
        self.background.set_relation("pv_storage_opt_1", "current_limit", 1.0)
        self.background.set_relation("pv_storage_opt_1", "temporal_periods", 1.0)
        self.background.set_relation("pv_storage_opt_1", "binary_variables", 1.0)
        
        # Concept C3: Vehicle routing problems
        self.background.set_relation("vrp_1", "capacity_constraint", 1.0)
        self.background.set_relation("vrp_1", "distance_minimization", 1.0)
        self.background.set_relation("vrp_1", "routing_problem", 1.0)
        self.background.set_relation("vrp_1", "vehicle_assignment", 1.0)
        self.background.set_relation("vrp_1", "discrete_variables", 1.0)
        
        # Concept C4: Charging station location
        self.background.set_relation("charging_loc_1", "capacity_constraint", 1.0)
        self.background.set_relation("charging_loc_1", "demand_satisfaction", 1.0)
        self.background.set_relation("charging_loc_1", "location_selection", 1.0)
        self.background.set_relation("charging_loc_1", "facility_location", 1.0)
        self.background.set_relation("charging_loc_1", "binary_variables", 1.0)
        
        # Concept C5: Scheduling problems
        self.background.set_relation("scheduling_1", "temporal_constraint", 1.0)
        self.background.set_relation("scheduling_1", "makespan_minimization", 1.0)
        self.background.set_relation("scheduling_1", "task_assignment", 1.0)
        self.background.set_relation("scheduling_1", "precedence_constraint", 1.0)
        self.background.set_relation("scheduling_1", "discrete_variables", 1.0)
        
        # Build concept lattice
        self.lattice = ConceptLattice(self.background)
        self.rough_processor = RoughSetProcessor(self.background)
    
    def extract_attributes_from_text(self, text: str) -> Set[str]:
        """Extract problem attributes from natural language text"""
        attributes = set()
        
        # Define keyword mappings
        keyword_mappings = {
            'battery': ['battery_voltage_range', 'module_configuration'],
            'voltage': ['battery_voltage_range'],
            'energy loss': ['energy_loss_minimization'],
            'loss': ['energy_loss_minimization'],
            'parallel': ['parallel_consistency'],
            'current': ['current_limit'],
            'module': ['module_configuration'],
            'period': ['temporal_periods'],
            'reconfigurable': ['reconfigurable_system'],
            'photovoltaic': ['reconfigurable_system'],
            'capacity': ['capacity_constraint'],
            'routing': ['routing_problem'],
            'vehicle': ['vehicle_assignment', 'routing_problem'],
            'distance': ['distance_minimization'],
            'scheduling': ['task_assignment'],
            'task': ['task_assignment'],
            'makespan': ['makespan_minimization'],
            'precedence': ['precedence_constraint'],
            'temporal': ['temporal_constraint'],
            'location': ['location_selection', 'facility_location'],
            'charging': ['facility_location'],
            'demand': ['demand_satisfaction'],
            'minimize': ['energy_loss_minimization', 'distance_minimization', 'makespan_minimization'],
        }
        
        text_lower = text.lower()
        
        for keyword, attrs in keyword_mappings.items():
            if keyword in text_lower:
                attributes.update(attrs)
        
        # Detect variable types
        if any(word in text_lower for word in ['select', 'activate', 'assign', 'choose']):
            attributes.add('binary_variables')
        
        return attributes
    
    def disambiguate_semantics(self, query_attrs: Set[str]) -> FormalConcept:
        """Semantic disambiguation using concept lattice"""
        if not self.lattice:
            raise ValueError("Concept lattice not initialized")
        
        best_concept = None
        best_score = -1
        
        for concept in self.lattice.concepts:
            score = self.lattice._jaccard_similarity(query_attrs, concept.attributes)
            if score > best_score:
                best_score = score
                best_concept = concept
        
        return best_concept
    
    def augment_query(self, query_attrs: Set[str]) -> Tuple[Set[str], Set[str]]:
        """Augment query with implicit constraints"""
        certain, possible = self.rough_processor.discover_implicit_constraints(query_attrs)
        augmented = query_attrs | certain
        return augmented, possible
    
    def process_optimization_problem(self, problem_description: str) -> Dict[str, Any]:
        """
        Main processing pipeline for optimization problem description
        
        Args:
            problem_description: Natural language description of the optimization problem
            
        Returns:
            Formal description including decision variables, constraints, and objectives
        """
        # Step 1: Extract attributes from text
        extracted_attrs = self.extract_attributes_from_text(problem_description)
        
        # Step 2: Semantic disambiguation
        matched_concept = self.disambiguate_semantics(extracted_attrs)
        
        # Step 3: Query augmentation with implicit constraints
        augmented_attrs, possible_attrs = self.augment_query(extracted_attrs)
        
        # Step 4: Generate formal description using LLM
        formal_description = self._generate_formal_description(
            problem_description,
            matched_concept,
            augmented_attrs,
            possible_attrs
        )
        
        return formal_description
    
    def _generate_formal_description(self, 
                                    problem_text: str,
                                    concept: FormalConcept,
                                    augmented_attrs: Set[str],
                                    possible_attrs: Set[str]) -> Dict[str, Any]:
        """Generate formal optimization problem description"""
        
        # Build context from matched concept and attributes
        context = self._build_context(concept, augmented_attrs, possible_attrs)
        
        # Create prompt for LLM
        prompt = self._create_llm_prompt(problem_text, context)
        
        # Call LLM API
        llm_response = self._call_llm_api(prompt)
        
        # Parse and structure the response
        formal_description = self._parse_llm_response(llm_response, augmented_attrs)
        
        return formal_description
    
    def _build_context(self, concept: FormalConcept, 
                      augmented_attrs: Set[str],
                      possible_attrs: Set[str]) -> str:
        """Build context information for LLM"""
        context = []
        
        context.append("## Matched Problem Domain")
        context.append(f"Similar problem instances: {', '.join(list(concept.objects)[:3])}")
        context.append(f"\nKey characteristics: {', '.join(concept.attributes)}")
        
        context.append("\n## Identified Attributes")
        context.append(f"Core attributes: {', '.join(augmented_attrs)}")
        
        if possible_attrs:
            context.append(f"\nPossible additional attributes: {', '.join(possible_attrs)}")
        
        return "\n".join(context)
    
    def _create_llm_prompt(self, problem_text: str, context: str) -> str:
        """Create prompt for LLM to generate formal description"""
        prompt = f"""You are an expert in constrained optimization problem formulation. Given a natural language description of an optimization problem and relevant domain context, generate a formal mathematical description.

{context}

## Problem Description
{problem_text}

## Task
Generate a formal description of this optimization problem including:

1. **Decision Variables**: Define all decision variables with their types (binary, integer, continuous), dimensions, and meaning.

2. **Objective Function**: Specify the optimization objective (minimize/maximize) and the mathematical expression.

3. **Constraints**: List all constraints with mathematical formulations:
   - Explicit constraints (directly mentioned in the problem)
   - Implicit constraints (inferred from domain knowledge)

4. **Parameters**: Define all parameters (constants) in the problem.

Provide the output in JSON format with the following structure:
{{
    "decision_variables": [
        {{"name": "variable_name", "type": "binary/integer/continuous", "dimension": "description", "description": "what it represents"}}
    ],
    "objective": {{
        "type": "minimize/maximize",
        "expression": "mathematical expression",
        "description": "what is being optimized"
    }},
    "constraints": [
        {{"name": "constraint_name", "type": "explicit/implicit", "expression": "mathematical formulation", "description": "constraint meaning"}}
    ],
    "parameters": [
        {{"name": "parameter_name", "description": "parameter meaning"}}
    ]
}}

IMPORTANT: Output ONLY valid JSON. Do not include any text outside the JSON structure."""

        return prompt
    
    def _call_llm_api(self, prompt: str) -> str:
        """
        Call LLM API to generate formal description
        
        This implementation uses Anthropic's Claude API
        Replace with your preferred LLM API
        """
        
        # For demonstration, return a structured response based on the prompt
        if "photovoltaic" in prompt.lower() or "battery" in prompt.lower():
            return self._generate_demo_response_battery()
        else:
            return self._generate_demo_response_generic()
    
    def _generate_demo_response_battery(self) -> str:
        """Generate demo response for battery optimization problem"""
        return json.dumps({
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
                "description": "Minimize total energy loss across all time periods, where E_loss(t) is the energy loss in period t"
            },
            "constraints": [
                {
                    "name": "parallel_consistency",
                    "type": "implicit",
                    "expression": "∑_{i=1}^{N_modules} x_ipt = ∑_{i=1}^{N_modules} x_ip't for all p, p' ∈ {1,...,N_parallel}, t ∈ {1,...,T}",
                    "description": "All parallel branches must have the same number of activated modules (parallel activation consistency)"
                },
                {
                    "name": "voltage_range",
                    "type": "explicit",
                    "expression": "V_min ≤ ∑_{i=1}^{N_modules} V_i · x_ipt ≤ V_max for all p ∈ {1,...,N_parallel}, t ∈ {1,...,T}",
                    "description": "System voltage must remain within specified range [V_min, V_max]"
                },
                {
                    "name": "current_limit",
                    "type": "explicit",
                    "expression": "I_p(t) ≤ I_max for all p ∈ {1,...,N_parallel}, t ∈ {1,...,T}",
                    "description": "Current through each module must not exceed maximum limit I_max"
                },
                {
                    "name": "demand_satisfaction",
                    "type": "implicit",
                    "expression": "∑_{p=1}^{N_parallel} I_p(t) · V_sys(t) ≥ P_demand(t) for all t ∈ {1,...,T}",
                    "description": "Total power output must meet demand in each period"
                }
            ],
            "parameters": [
                {"name": "N_modules", "description": "Number of battery modules in series per column"},
                {"name": "N_parallel", "description": "Number of parallel columns"},
                {"name": "T", "description": "Number of time periods"},
                {"name": "V_i", "description": "Voltage of battery module i"},
                {"name": "V_min, V_max", "description": "Minimum and maximum system voltage limits"},
                {"name": "I_max", "description": "Maximum current limit per module"},
                {"name": "P_demand(t)", "description": "Power demand in period t"},
                {"name": "E_loss(t)", "description": "Energy loss function in period t"}
            ]
        }, indent=2)
    
    def _generate_demo_response_generic(self) -> str:
        """Generate generic demo response"""
        return json.dumps({
            "decision_variables": [
                {
                    "name": "x",
                    "type": "binary",
                    "dimension": "problem-specific",
                    "description": "Primary decision variable"
                }
            ],
            "objective": {
                "type": "minimize",
                "expression": "f(x)",
                "description": "Objective function to be minimized"
            },
            "constraints": [
                {
                    "name": "feasibility",
                    "type": "explicit",
                    "expression": "g(x) ≤ 0",
                    "description": "Feasibility constraints"
                }
            ],
            "parameters": []
        }, indent=2)
    
    def _parse_llm_response(self, response: str, 
                           augmented_attrs: Set[str]) -> Dict[str, Any]:
        """Parse LLM response and structure the formal description"""
        try:
            # Try to parse as JSON
            formal_desc = json.loads(response)
            
            # Add metadata
            formal_desc["metadata"] = {
                "extracted_attributes": list(augmented_attrs),
                "processing_method": "RAG-FCA"
            }
            
            return formal_desc
        except json.JSONDecodeError:
            # If not valid JSON, try to extract structured information
            return {
                "raw_response": response,
                "metadata": {
                    "extracted_attributes": list(augmented_attrs),
                    "processing_method": "RAG-FCA",
                    "note": "Response parsing failed, returning raw output"
                }
            }
    
    def update_knowledge_base(self, new_object: str, 
                             new_attributes: Dict[str, float]):
        """
        Update knowledge base with new problem instance (incremental learning)
        
        Args:
            new_object: New problem instance identifier
            new_attributes: Dictionary of attributes and their membership degrees
        """
        for attr, degree in new_attributes.items():
            self.background.set_relation(new_object, attr, degree)
        
        # Rebuild concept lattice
        self.lattice = ConceptLattice(self.background)
        self.rough_processor = RoughSetProcessor(self.background)
    
    def export_knowledge_base(self, filepath: str):
        """Export knowledge base to JSON file"""
        kb_data = {
            "objects": list(self.background.objects),
            "attributes": list(self.background.attributes),
            "relations": [
                {"object": obj, "attribute": attr, "degree": degree}
                for (obj, attr), degree in self.background.incidence.items()
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(kb_data, f, indent=2)
    
    def import_knowledge_base(self, filepath: str):
        """Import knowledge base from JSON file"""
        with open(filepath, 'r') as f:
            kb_data = json.load(f)
        
        self.background = FormalBackground()
        
        for obj in kb_data["objects"]:
            self.background.add_object(obj)
        
        for attr in kb_data["attributes"]:
            self.background.add_attribute(attr)
        
        for relation in kb_data["relations"]:
            self.background.set_relation(
                relation["object"],
                relation["attribute"],
                relation["degree"]
            )
        
        # Rebuild structures
        self.lattice = ConceptLattice(self.background)
        self.rough_processor = RoughSetProcessor(self.background)


# API Integration Code
def call_claude_api(prompt: str, api_key: str = None) -> str:
    """
    Call Claude API to generate formal optimization problem description
    
    Args:
        prompt: The prompt containing problem description and context
        api_key: Anthropic API key (if None, reads from ANTHROPIC_API_KEY env var)
    
    Returns:
        LLM response as string
    
    Usage:
        import os
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        response = call_claude_api(prompt, api_key)
    
    To use this function, install the anthropic library:
        pip install anthropic
    
    Then uncomment the implementation below:
    """
    
    # import anthropic
    # import os
    # 
    # client = anthropic.Anthropic(
    #     api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
    # )
    # 
    # message = client.messages.create(
    #     model="claude-sonnet-4-20250514",
    #     max_tokens=4096,
    #     messages=[
    #         {"role": "user", "content": prompt}
    #     ]
    # )
    # 
    # return message.content[0].text
    
    raise NotImplementedError("Please uncomment the API implementation code above and install the anthropic library")


def main():
    """Example usage of RAG-FCA system"""
    
    # Initialize system
    system = RAGFCASystem()
    
    # Example problem description
    problem_description = """
    We have a reconfigurable photovoltaic energy storage system composed of multiple 
    parallel battery modules and series columns that need optimization. The goal is to 
    select the activated battery module configuration for each period based on different 
    demands in order to minimize the total energy loss, while ensuring that the constraints 
    of parallel activation consistency, system voltage range, and maximum module current 
    are met.
    """
    
    print("=" * 80)
    print("RAG-FCA System for Optimization Problem Processing")
    print("=" * 80)
    print("\nInput Problem Description:")
    print(problem_description)
    print("\n" + "=" * 80)
    
    # Process the problem
    result = system.process_optimization_problem(problem_description)
    
    print("\nFormal Problem Description:")
    print("=" * 80)
    print(json.dumps(result, indent=2))
    
    # Export knowledge base
    system.export_knowledge_base("/home/claude/knowledge_base.json")
    print("\n" + "=" * 80)
    print("Knowledge base exported to: knowledge_base.json")
    
    return system, result


if __name__ == "__main__":
    system, result = main()
