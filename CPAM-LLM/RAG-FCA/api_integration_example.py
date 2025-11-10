"""
Example: Integrating RAG-FCA System with Claude API

This file demonstrates how to integrate the RAG-FCA system with Anthropic's Claude API
for real-time LLM-based formal description generation.
"""

import os
from rag_fca_system import RAGFCASystem

# ============================================================================
# API Integration with Anthropic Claude
# ============================================================================

def call_claude_api_integrated(prompt: str, api_key: str = None) -> str:
    """
    Call Claude API to generate formal optimization problem description
    
    Args:
        prompt: The prompt containing problem description and context
        api_key: Anthropic API key (optional, reads from environment if not provided)
    
    Returns:
        LLM response as JSON string
    
    Installation:
        pip install anthropic
    """
    import anthropic
    
    client = anthropic.Anthropic(
        api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
    )
    
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return message.content[0].text


# ============================================================================
# Extended RAG-FCA System with API Integration
# ============================================================================

class RAGFCASystemWithAPI(RAGFCASystem):
    """Extended RAG-FCA system with real Claude API integration"""
    
    def __init__(self, llm_api_key: str = None):
        super().__init__(llm_api_key)
        self.use_api = llm_api_key is not None or os.environ.get("ANTHROPIC_API_KEY") is not None
    
    def _call_llm_api(self, prompt: str) -> str:
        """
        Call LLM API to generate formal description
        
        This version uses the real Claude API if API key is available,
        otherwise falls back to demo responses
        """
        if self.use_api:
            try:
                return call_claude_api_integrated(prompt, self.llm_api_key)
            except Exception as e:
                print(f"API call failed: {e}")
                print("Falling back to demo response...")
        
        # Fallback to demo responses
        if "photovoltaic" in prompt.lower() or "battery" in prompt.lower():
            return self._generate_demo_response_battery()
        else:
            return self._generate_demo_response_generic()


# ============================================================================
# Usage Examples
# ============================================================================

def example_basic_usage():
    """Example 1: Basic usage with demo responses"""
    print("="*80)
    print("Example 1: Basic Usage (Demo Mode)")
    print("="*80)
    
    system = RAGFCASystem()
    
    problem = """
    We need to schedule 10 tasks on 3 machines. Each task has a processing time 
    and must be completed before its deadline. Tasks may have precedence constraints. 
    The goal is to minimize the makespan while respecting all constraints.
    """
    
    result = system.process_optimization_problem(problem)
    
    import json
    print("\nFormal Description:")
    print(json.dumps(result, indent=2))


def example_with_api():
    """Example 2: Usage with real Claude API"""
    print("="*80)
    print("Example 2: With Claude API")
    print("="*80)
    
    # Set your API key here or in environment variable
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if not api_key:
        print("Warning: ANTHROPIC_API_KEY not set. Using demo mode.")
        system = RAGFCASystem()
    else:
        system = RAGFCASystemWithAPI(api_key)
    
    problem = """
    Design an optimal charging station network for electric vehicles in a city.
    We have 20 potential locations and need to select which ones to activate.
    Each station has a capacity limit and installation cost. The goal is to 
    minimize total cost while ensuring that demand from all city areas is satisfied.
    """
    
    result = system.process_optimization_problem(problem)
    
    import json
    print("\nFormal Description:")
    print(json.dumps(result, indent=2))


def example_knowledge_base_update():
    """Example 3: Updating knowledge base with new problems"""
    print("="*80)
    print("Example 3: Knowledge Base Update")
    print("="*80)
    
    system = RAGFCASystem()
    
    # Add a new problem type to the knowledge base
    new_problem_attrs = {
        "portfolio_optimization": 1.0,
        "risk_minimization": 1.0,
        "return_constraint": 1.0,
        "diversification": 1.0,
        "continuous_variables": 1.0
    }
    
    system.update_knowledge_base("portfolio_opt_1", new_problem_attrs)
    
    print("Knowledge base updated with new problem type: portfolio optimization")
    
    # Test with a portfolio optimization problem
    problem = """
    Optimize an investment portfolio with 50 assets to minimize risk while 
    achieving a minimum expected return. Include diversification constraints.
    """
    
    result = system.process_optimization_problem(problem)
    
    import json
    print("\nFormal Description:")
    print(json.dumps(result, indent=2))


def example_batch_processing():
    """Example 4: Batch processing multiple problems"""
    print("="*80)
    print("Example 4: Batch Processing")
    print("="*80)
    
    system = RAGFCASystem()
    
    problems = [
        "Route 5 delivery vehicles to serve 30 customers with minimum total distance.",
        "Schedule 20 jobs on 4 machines to minimize total completion time.",
        "Select optimal battery configuration for each hour to minimize energy loss."
    ]
    
    results = []
    for i, problem in enumerate(problems, 1):
        print(f"\nProcessing problem {i}...")
        result = system.process_optimization_problem(problem)
        results.append(result)
    
    # Save all results
    import json
    with open('batch_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nProcessed {len(problems)} problems. Results saved to batch_results.json")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        example_choice = sys.argv[1]
        
        if example_choice == "1":
            example_basic_usage()
        elif example_choice == "2":
            example_with_api()
        elif example_choice == "3":
            example_knowledge_base_update()
        elif example_choice == "4":
            example_batch_processing()
        else:
            print(f"Unknown example: {example_choice}")
            print("Usage: python api_integration_example.py [1|2|3|4]")
    else:
        # Run all examples
        print("\n" + "="*80)
        print("Running All Examples")
        print("="*80 + "\n")
        
        example_basic_usage()
        print("\n")
        
        example_with_api()
        print("\n")
        
        example_knowledge_base_update()
        print("\n")
        
        example_batch_processing()
