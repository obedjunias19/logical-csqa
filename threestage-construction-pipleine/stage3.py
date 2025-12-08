import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df2 = pd.read_csv('/kaggle/input/csqa-dataset/csqa_second_stage-5000.csv')

import pandas as pd
import ast
import itertools
import json
from typing import List, Dict, Tuple

import pandas as pd
import ast
import itertools
import json
from typing import List, Dict

def generate_logical_combinations(csv_file_path: str, output_file_path: str = None) -> Dict:
    """
    Generate AND, OR, NEITHER/NOR combinations from CSV data,
    avoiding duplicates and symmetric repeats.
    """
    
    def parse_facts_string(facts_str: str) -> List[str]:
        """Parse stringified list to Python list"""
        try:
            return ast.literal_eval(facts_str)
        except:
            return [item.strip().strip("'\"") for item in facts_str.split(",")]
    
    def generate_and_combinations(correct: List[str], incorrect: List[str]) -> Dict[str, List[str]]:
        valid_and = [f"{c1} AND {c2}" for c1, c2 in itertools.combinations(correct, 2)]
        invalid_and = [f"{c} AND {i}" for c in correct for i in incorrect] + \
                      [f"{i1} AND {i2}" for i1, i2 in itertools.combinations(incorrect, 2)]
        return {"correct": valid_and, "incorrect": invalid_and}
    
    def generate_or_combinations(correct: List[str], incorrect: List[str]) -> Dict[str, List[str]]:
        valid_or = [f"{c1} OR {c2}" for c1, c2 in itertools.combinations(correct, 2)]
        valid_or += [f"{c} OR {i}" for c in correct for i in incorrect]  # only one order
        invalid_or = [f"{i1} OR {i2}" for i1, i2 in itertools.combinations(incorrect, 2)]
        return {"correct": valid_or, "incorrect": invalid_or}
    
    def generate_neither_combinations(correct: List[str], incorrect: List[str]) -> Dict[str, List[str]]:
        valid_neither = [f"NEITHER {i1} NOR {i2}" for i1, i2 in itertools.combinations(incorrect, 2)]
        # Incorrect: pairs with at least one correct
        incorrect_neither = [f"NEITHER {c1} NOR {c2}" for c1, c2 in itertools.combinations(correct, 2)]
        incorrect_neither += [f"NEITHER {c} NOR {i}" for c in correct for i in incorrect]  # only one order
        return {"correct": valid_neither, "incorrect": incorrect_neither}
    
    # Read CSV
    df = pd.read_csv(csv_file_path)
    required_columns = ['question', 'refined_correct_options', 'refined_incorrect_options']
    for col in required_columns:
        if col not in df.columns:
            raise Exception(f"Missing column: {col}")
    
    results = []
    for idx, row in df.iterrows():
        question = row['question']
        correct_facts = parse_facts_string(str(row['refined_correct_options']))
        incorrect_facts = parse_facts_string(str(row['refined_incorrect_options']))
        
        and_comb = generate_and_combinations(correct_facts, incorrect_facts)
        or_comb = generate_or_combinations(correct_facts, incorrect_facts)
        neither_comb = generate_neither_combinations(correct_facts, incorrect_facts)
        
        question_result = {
            "question": question,
            "original_correct_facts": correct_facts,
            "original_incorrect_facts": incorrect_facts,
            "logical_combinations": {
                "AND_combinations": and_comb,
                "OR_combinations": or_comb,
                "NEITHER_combinations": neither_comb
            }
        }
        results.append(question_result)
    
    final_result = {
        "total_questions_processed": len(results),
        "questions": results
    }
    
    if output_file_path:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_file_path}")
    
    return final_result

def symbolic_function():
    """Test the function with sample data"""

    try:
        results = generate_logical_combinations('/kaggle/input/csqa-dataset/csqa_second_stage-5000.csv', 'logical_combinations_output.json')
        
        # Print sample results
        print("Sample Results:")
        print(f"Processed {results['total_questions_processed']} questions")
        
        for i, question_data in enumerate(results['questions'][:1]):  # Show first question
            print(f"\nQuestion {i+1}: {question_data['question']}")
            print("\nAND Combinations:")
            print(f"  Correct: {question_data['logical_combinations']['AND_combinations']['correct']}")
            print(f"  Incorrect: {question_data['logical_combinations']['AND_combinations']['incorrect']}")

            print(f"\nQuestion {i+1}: {question_data['question']}")
            print("\nOR Combinations:")
            print(f"  Correct: {question_data['logical_combinations']['OR_combinations']['correct']}")
            print(f"  Incorrect: {question_data['logical_combinations']['OR_combinations']['incorrect']}")

            print(f"\nQuestion {i+1}: {question_data['question']}")
            print("\nNeither Combinations:")
            print(f"  Correct: {question_data['logical_combinations']['NEITHER_combinations']['correct']}")
            print(f"  Incorrect: {question_data['logical_combinations']['NEITHER_combinations']['incorrect']}")

            
            
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    symbolic_function()