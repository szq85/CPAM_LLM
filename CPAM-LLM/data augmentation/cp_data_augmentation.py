"""
CP Dataset Augmentation using Chaotic Perturbation

This module implements a comprehensive data augmentation system for 
Constraint Programming (CP) datasets using chaotic mapping-based 
perturbations combined with LLM-powered reframing.
"""

import pandas as pd
import numpy as np
import re
import random
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import requests
from copy import deepcopy


@dataclass
class PerturbationParams:
    """Container for perturbation parameters derived from chaotic mapping"""
    alpha: float  # Natural language perturbation rate
    beta: float   # Model perturbation rate  
    gamma: float  # Code perturbation rate


class ChaoticMapper:
    """
    Implements logistic chaotic mapping for generating perturbation parameters.
    
    The logistic map: x_{n+1} = r * x_n * (1 - x_n), where r ∈ [3.5, 4]
    """
    
    def __init__(self, x0: float = 0.4, r: float = 3.9):
        """
        Initialize chaotic mapper.
        
        Args:
            x0: Initial value, should be in (0, 1)
            r: Mapping parameter, should be in [3.5, 4] for chaotic behavior
        """
        assert 0 < x0 < 1, "Initial value must be in (0, 1)"
        assert 3.5 <= r <= 4, "Parameter r must be in [3.5, 4] for chaotic behavior"
        
        self.x0 = x0
        self.r = r
        self.current_x = x0
        
    def next(self) -> float:
        """Generate next value in chaotic sequence"""
        self.current_x = self.r * self.current_x * (1 - self.current_x)
        return self.current_x
    
    def reset(self, x0: Optional[float] = None):
        """Reset the chaotic sequence"""
        if x0 is not None:
            assert 0 < x0 < 1, "Initial value must be in (0, 1)"
            self.x0 = x0
        self.current_x = self.x0
    
    def generate_params(self, 
                       alpha_max: float = 0.3, 
                       beta_max: float = 0.3, 
                       gamma_max: float = 0.3) -> PerturbationParams:
        """
        Generate perturbation parameters from chaotic value.
        
        Args:
            alpha_max: Maximum natural language perturbation rate
            beta_max: Maximum model perturbation rate
            gamma_max: Maximum code perturbation rate
            
        Returns:
            PerturbationParams object with scaled parameters
        """
        x = self.next()
        return PerturbationParams(
            alpha=alpha_max * x,
            beta=beta_max * x,
            gamma=gamma_max * x
        )


class NaturalLanguagePerturber:
    """
    Applies perturbations to natural language descriptions.
    Includes: synonym substitution, word reordering, and lightweight insertions.
    """
    
    # Domain-specific synonym mappings for CP problems
    SYNONYMS = {
        'minimize': ['reduce', 'optimize', 'lower'],
        'maximize': ['increase', 'optimize', 'raise'],
        'constraint': ['restriction', 'limitation', 'condition'],
        'schedule': ['plan', 'organize', 'arrange'],
        'machine': ['equipment', 'device', 'processor'],
        'operation': ['task', 'process', 'activity'],
        'job': ['task', 'work item', 'assignment'],
        'time': ['duration', 'period', 'interval'],
        'optimal': ['best', 'ideal', 'most efficient'],
        'problem': ['challenge', 'issue', 'scenario'],
        'solution': ['resolution', 'answer', 'outcome'],
        'objective': ['goal', 'aim', 'target'],
        'processing': ['execution', 'handling', 'completion'],
        'sequence': ['order', 'series', 'succession'],
        'allocation': ['assignment', 'distribution', 'placement'],
    }
    
    INSERTION_TEMPLATES = [
        "It should be noted that",
        "Specifically,",
        "In this context,",
        "Moreover,",
        "Additionally,",
        "Furthermore,",
        "Notably,",
    ]
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
    
    def perturb(self, text: str, alpha: float) -> str:
        """
        Apply perturbations to natural language text.
        
        Args:
            text: Original text
            alpha: Perturbation rate (0-1)
            
        Returns:
            Perturbed text
        """
        if alpha < 0.05:  # Minimal perturbation
            return text
            
        perturbed = text
        
        # Apply synonym substitution
        if random.random() < alpha:
            perturbed = self._substitute_synonyms(perturbed, alpha)
        
        # Apply lightweight insertions
        if random.random() < alpha * 0.5:
            perturbed = self._insert_phrases(perturbed, alpha)
        
        # Apply minor reordering
        if random.random() < alpha * 0.3:
            perturbed = self._reorder_sentences(perturbed, alpha)
        
        return perturbed
    
    def _substitute_synonyms(self, text: str, alpha: float) -> str:
        """Replace words with synonyms"""
        words = text.split()
        num_substitutions = int(len(words) * alpha * 0.2)  # Substitute up to 20% * alpha
        
        for _ in range(num_substitutions):
            for word_key in self.SYNONYMS:
                pattern = r'\b' + word_key + r'\b'
                if re.search(pattern, text, re.IGNORECASE):
                    synonym = random.choice(self.SYNONYMS[word_key])
                    text = re.sub(pattern, synonym, text, count=1, flags=re.IGNORECASE)
                    break
        
        return text
    
    def _insert_phrases(self, text: str, alpha: float) -> str:
        """Insert transitional phrases"""
        sentences = text.split('. ')
        if len(sentences) < 2:
            return text
        
        insert_idx = random.randint(1, len(sentences) - 1)
        phrase = random.choice(self.INSERTION_TEMPLATES)
        sentences[insert_idx] = phrase + ' ' + sentences[insert_idx]
        
        return '. '.join(sentences)
    
    def _reorder_sentences(self, text: str, alpha: float) -> str:
        """Lightly reorder independent sentences"""
        sentences = text.split('. ')
        if len(sentences) < 3:
            return text
        
        # Only swap adjacent sentences occasionally
        if random.random() < 0.3:
            idx = random.randint(0, len(sentences) - 2)
            sentences[idx], sentences[idx + 1] = sentences[idx + 1], sentences[idx]
        
        return '. '.join(sentences)


class CodePerturber:
    """
    Applies perturbations to CP code while maintaining functionality.
    Includes: variable renaming, comment repositioning, and block reordering.
    """
    
    VARIABLE_PREFIXES = ['var', 'val', 'temp', 'local', 'item', 'elem']
    COMMENT_STYLES = [
        "# {}",
        "# Note: {}",
        "# Important: {}",
    ]
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
    
    def perturb(self, code: str, gamma: float) -> str:
        """
        Apply perturbations to code.
        
        Args:
            code: Original code
            gamma: Perturbation strength (0-1)
            
        Returns:
            Perturbed code
        """
        if gamma < 0.05:
            return code
        
        perturbed = code
        
        # Apply variable renaming
        if random.random() < gamma:
            perturbed = self._rename_variables(perturbed, gamma)
        
        # Apply comment modifications
        if random.random() < gamma * 0.5:
            perturbed = self._modify_comments(perturbed, gamma)
        
        # Apply spacing variations
        if random.random() < gamma * 0.3:
            perturbed = self._adjust_spacing(perturbed)
        
        return perturbed
    
    def _rename_variables(self, code: str, gamma: float) -> str:
        """Rename local variables systematically"""
        # Find common variable patterns
        var_pattern = r'\b([a-z_][a-z0-9_]*)\b'
        variables = set(re.findall(var_pattern, code))
        
        # Filter out keywords and common names
        keywords = {'for', 'in', 'if', 'else', 'while', 'def', 'class', 'return', 
                   'import', 'from', 'as', 'with', 'try', 'except', 'and', 'or'}
        variables = variables - keywords
        
        # Select subset to rename
        num_renames = int(len(variables) * gamma * 0.3)
        vars_to_rename = random.sample(list(variables), min(num_renames, len(variables)))
        
        for old_var in vars_to_rename:
            prefix = random.choice(self.VARIABLE_PREFIXES)
            new_var = f"{prefix}_{old_var}"
            # Use word boundaries to avoid partial replacements
            code = re.sub(r'\b' + old_var + r'\b', new_var, code)
        
        return code
    
    def _modify_comments(self, code: str, gamma: float) -> str:
        """Modify comment styles"""
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if '#' in line and random.random() < gamma * 0.2:
                # Extract comment
                parts = line.split('#', 1)
                if len(parts) == 2:
                    comment_text = parts[1].strip()
                    # Remove existing comment style prefixes
                    comment_text = re.sub(r'^(Note:|Important:)\s*', '', comment_text)
                    # Apply new style
                    new_comment = random.choice(self.COMMENT_STYLES).format(comment_text)
                    lines[i] = parts[0] + new_comment
        
        return '\n'.join(lines)
    
    def _adjust_spacing(self, code: str) -> str:
        """Add or remove blank lines"""
        lines = code.split('\n')
        result = []
        
        for i, line in enumerate(lines):
            result.append(line)
            # Occasionally add blank line after certain patterns
            if random.random() < 0.1:
                if any(pattern in line for pattern in ['def ', 'class ', 'for ', 'if ']):
                    if i < len(lines) - 1 and lines[i + 1].strip():
                        result.append('')
        
        return '\n'.join(result)


class LLMReframer:
    """
    Uses LLM API to reframe perturbed content while maintaining semantics.
    """
    
    def __init__(self, api_key: str, api_base: str = "https://api.chatanywhere.tech/v1"):
        self.api_key = api_key
        self.api_base = api_base
        self.model = "gpt-3.5-turbo"
    
    def reframe_description(self, original: str, perturbed: str) -> str:
        """
        Reframe natural language description using LLM.
        
        Args:
            original: Original description
            perturbed: Perturbed description
            
        Returns:
            Reframed description with rich stylistic variation
        """
        prompt = f"""You are an expert in constraint programming. Please rewrite the following problem description while maintaining its exact technical meaning and all constraints. The rewritten version should:
1. Use different sentence structures and phrasings
2. Maintain all numerical values and technical details
3. Keep the same level of formality
4. Preserve all constraint specifications

Original description:
{perturbed}

Provide only the rewritten description without any preamble or explanation."""

        try:
            response = self._call_llm(prompt)
            return response if response else perturbed
        except Exception as e:
            print(f"LLM reframing failed: {e}")
            return perturbed
    
    def reframe_code(self, original: str, perturbed: str) -> str:
        """
        Reframe code using LLM while maintaining functionality.
        
        Args:
            original: Original code
            perturbed: Perturbed code
            
        Returns:
            Reframed code with stylistic variations
        """
        prompt = f"""You are a Python expert. Please rewrite the following Python code while maintaining its exact functionality. The rewritten version should:
1. Use different variable names and code organization
2. Maintain all logic and functionality
3. Keep the same imports and dependencies
4. Preserve all constraint definitions

Original code:
{perturbed}

Provide only the rewritten code without any explanation or markdown formatting."""

        try:
            response = self._call_llm(prompt)
            return response if response else perturbed
        except Exception as e:
            print(f"LLM code reframing failed: {e}")
            return perturbed
    
    def _call_llm(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Call LLM API with retry logic"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result['choices'][0]['message']['content'].strip()
                else:
                    print(f"API error: {response.status_code}, {response.text}")
            except Exception as e:
                print(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        return None


class CodeValidator:
    """
    Validates generated code for syntax errors and compilation issues.
    """
    
    @staticmethod
    def validate(code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate code syntax.
        
        Args:
            code: Code to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            compile(code, '<string>', 'exec')
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, f"Compilation error: {str(e)}"


class DataAugmentationPipeline:
    """
    Main pipeline for CP dataset augmentation using chaotic perturbation.
    """
    
    def __init__(self, 
                 api_key: str,
                 x0: float = 0.4,
                 r: float = 3.9,
                 alpha_max: float = 0.3,
                 beta_max: float = 0.3,
                 gamma_max: float = 0.3,
                 use_llm: bool = True):
        """
        Initialize augmentation pipeline.
        
        Args:
            api_key: LLM API key
            x0: Initial value for chaotic mapping
            r: Parameter for chaotic mapping
            alpha_max: Maximum natural language perturbation rate
            beta_max: Maximum model perturbation rate
            gamma_max: Maximum code perturbation rate
            use_llm: Whether to use LLM for reframing
        """
        self.chaotic_mapper = ChaoticMapper(x0, r)
        self.nl_perturber = NaturalLanguagePerturber()
        self.code_perturber = CodePerturber()
        self.llm_reframer = LLMReframer(api_key) if use_llm else None
        self.validator = CodeValidator()
        
        self.alpha_max = alpha_max
        self.beta_max = beta_max
        self.gamma_max = gamma_max
        self.use_llm = use_llm
        
        self.augmented_samples = []
        self.validation_stats = {
            'total_generated': 0,
            'passed_validation': 0,
            'failed_syntax': 0,
            'failed_manual': 0
        }
    
    def augment_sample(self, 
                      description: str, 
                      code: str,
                      formal_model: Optional[str] = None,
                      iteration: int = 1) -> Optional[Dict]:
        """
        Augment a single sample using chaotic perturbation.
        
        Args:
            description: Natural language description
            code: CP code implementation
            formal_model: Formal CP model (optional)
            iteration: Iteration number
            
        Returns:
            Augmented sample dictionary or None if validation fails
        """
        # Generate perturbation parameters from chaotic sequence
        params = self.chaotic_mapper.generate_params(
            self.alpha_max, self.beta_max, self.gamma_max
        )
        
        print(f"  Perturbation params - α={params.alpha:.3f}, β={params.beta:.3f}, γ={params.gamma:.3f}")
        
        # Apply structural perturbations
        perturbed_desc = self.nl_perturber.perturb(description, params.alpha)
        perturbed_code = self.code_perturber.perturb(code, params.gamma)
        
        # Apply LLM reframing if enabled
        if self.use_llm and self.llm_reframer:
            print("  Applying LLM reframing...")
            perturbed_desc = self.llm_reframer.reframe_description(description, perturbed_desc)
            perturbed_code = self.llm_reframer.reframe_code(code, perturbed_code)
        
        # Validate generated code
        self.validation_stats['total_generated'] += 1
        is_valid, error_msg = self.validator.validate(perturbed_code)
        
        if not is_valid:
            print(f"  ✗ Validation failed: {error_msg}")
            self.validation_stats['failed_syntax'] += 1
            return None
        
        print("  ✓ Validation passed")
        self.validation_stats['passed_validation'] += 1
        
        return {
            'description': perturbed_desc,
            'code': perturbed_code,
            'formal_model': formal_model,
            'iteration': iteration,
            'params': {
                'alpha': params.alpha,
                'beta': params.beta,
                'gamma': params.gamma
            }
        }
    
    def augment_dataset(self, 
                       df: pd.DataFrame,
                       num_iterations: int = 2,
                       samples_per_iteration: Optional[int] = None) -> pd.DataFrame:
        """
        Augment entire dataset.
        
        Args:
            df: Original dataset DataFrame
            num_iterations: Number of augmentation iterations
            samples_per_iteration: Number of samples to process per iteration (None = all)
            
        Returns:
            Augmented dataset DataFrame
        """
        print(f"\n{'='*80}")
        print(f"Starting dataset augmentation")
        print(f"Original dataset size: {len(df)}")
        print(f"Iterations: {num_iterations}")
        print(f"{'='*80}\n")
        
        augmented_records = []
        
        for iteration in range(1, num_iterations + 1):
            print(f"\n{'='*80}")
            print(f"Iteration {iteration}/{num_iterations}")
            print(f"{'='*80}")
            
            # Sample subset if specified
            if samples_per_iteration:
                sample_df = df.sample(n=min(samples_per_iteration, len(df)))
            else:
                sample_df = df
            
            print(f"Processing {len(sample_df)} samples...")
            
            for idx, row in sample_df.iterrows():
                print(f"\nSample {idx + 1}/{len(sample_df)} (Original index: {row['Number']})")
                print(f"Problem type: {row['Problem types']}")
                
                augmented = self.augment_sample(
                    description=row['CP natural language'],
                    code=row['CP code'],
                    formal_model=row.get('CP formal language'),
                    iteration=iteration
                )
                
                if augmented:
                    record = {
                        'Number': f"{row['Number']}_iter{iteration}",
                        'CP natural language': augmented['description'],
                        'CP code': augmented['code'],
                        'Constraint': row['Constraint'],
                        'Problem types': row['Problem types'],
                        'CP formal language': augmented['formal_model'],
                        'Augmentation iteration': iteration,
                        'Original number': row['Number'],
                        'Alpha': augmented['params']['alpha'],
                        'Beta': augmented['params']['beta'],
                        'Gamma': augmented['params']['gamma']
                    }
                    augmented_records.append(record)
        
        # Combine original and augmented data
        augmented_df = pd.DataFrame(augmented_records)
        combined_df = pd.concat([df, augmented_df], ignore_index=True)
        
        print(f"\n{'='*80}")
        print(f"Augmentation complete!")
        print(f"Original samples: {len(df)}")
        print(f"Augmented samples: {len(augmented_records)}")
        print(f"Total samples: {len(combined_df)}")
        print(f"\nValidation statistics:")
        print(f"  Total generated: {self.validation_stats['total_generated']}")
        print(f"  Passed validation: {self.validation_stats['passed_validation']}")
        print(f"  Failed syntax check: {self.validation_stats['failed_syntax']}")
        print(f"  Success rate: {self.validation_stats['passed_validation']/max(1,self.validation_stats['total_generated'])*100:.1f}%")
        print(f"{'='*80}\n")
        
        return combined_df


def main():
    """Main execution function"""
    
    # Configuration
    CONFIG = {
        'input_file': 'FCA/CP_finetuning.xlsx',
        'output_file': 'augmented_cp_dataset.xlsx',
        'api_key': 'Your key',  # Replace with actual API key
        'x0': 0.4,  # Initial value for chaotic mapping
        'r': 3.9,   # Chaotic parameter
        'alpha_max': 0.3,  # Max natural language perturbation
        'beta_max': 0.3,   # Max model perturbation  
        'gamma_max': 0.3,  # Max code perturbation
        'num_iterations': 1,  # Number of augmentation iterations
        'samples_per_iteration': 2,  # Process 50 samples per iteration (set to None for all)
        'use_llm': True  # Whether to use LLM for reframing
    }
    
    print("\n" + "="*80)
    print("CP DATASET AUGMENTATION USING CHAOTIC PERTURBATION")
    print("="*80)
    
    # Load dataset
    print(f"\nLoading dataset from: {CONFIG['input_file']}")
    df = pd.read_excel(CONFIG['input_file'])
    print(f"Loaded {len(df)} samples")
    print(f"Problem types: {df['Problem types'].unique()}")
    
    # Initialize pipeline
    pipeline = DataAugmentationPipeline(
        api_key=CONFIG['api_key'],
        x0=CONFIG['x0'],
        r=CONFIG['r'],
        alpha_max=CONFIG['alpha_max'],
        beta_max=CONFIG['beta_max'],
        gamma_max=CONFIG['gamma_max'],
        use_llm=CONFIG['use_llm']
    )
    
    # Perform augmentation
    augmented_df = pipeline.augment_dataset(
        df=df,
        num_iterations=CONFIG['num_iterations'],
        samples_per_iteration=CONFIG['samples_per_iteration']
    )

    # Save results
    print(f"Saving augmented dataset to: {CONFIG['output_file']}")
    augmented_df.to_excel(CONFIG['output_file'], index=False)
    
    # Generate summary statistics
    print("\n" + "="*80)
    print("DATASET SUMMARY")
    print("="*80)
    print(f"\nProblem type distribution:")
    print(augmented_df['Problem types'].value_counts())
    
    if 'Augmentation iteration' in augmented_df.columns:
        print(f"\nAugmentation iteration distribution:")
        print(augmented_df['Augmentation iteration'].value_counts().sort_index())
    
    print("\n" + "="*80)
    print("AUGMENTATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
