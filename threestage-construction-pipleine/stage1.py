import os
import json
import time
import pandas as pd
import random
from openai import OpenAI, RateLimitError
from tenacity import retry, wait_random_exponential, stop_after_attempt
from datasets import load_dataset
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()

openai_key = user_secrets.get_secret("openai")
if not openai_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
client = OpenAI(api_key=openai_key)


def call_gpt4o_mini(messages):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.5,
            n=1
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except RateLimitError as e:
        print(f"Rate limit error: {e}. Retrying...")
        raise
    except json.JSONDecodeError as e:
        print(f"Received invalid JSON from API: {e}. Retrying...")
        raise e
    except Exception as e:
        print(f"An unexpected error occurred during the API call: {e}")
        raise e

def load_and_filter_csqa_data():
    print("Loading CSQA dataset...")
    ds = load_dataset("tau/commonsense_qa")
    train_df = ds['train'].to_pandas()
    valid_df  = ds['validation'].to_pandas()
    test_df = ds['test'].to_pandas()
    df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    print(df.shape)
    
    print(f"Loaded and filtered {len(df)} text-only questions.")
    return df.iloc[:5000]



def generate_stage1_options(question, existing_correct, existing_incorrect, context=""):
    user_content = f"""
You are a cognitive and logical reasoning expert, skilled in designing assessments that test authentic, multi-step inference. Your specialty is crafting challenging options that force high school students with deep understanding to demonstrate genuine causal understanding rather than surface-level pattern matching. The goal is creating options that require connecting multiple concepts through valid causal chains.

QUESTION: {question}
EXISTING CORRECT: {existing_correct}
EXISTING INCORRECT: {existing_incorrect}

TASK: Expand and improve the existing option pools using the following instructions, making the selection challenging even for high school students.

ADDITIONAL CORRECT OPTIONS Rubric (Generate 4-6 more):
- Use word action phrases or method descriptions
- Should complement existing correct options (different approaches/mechanisms)
- Each must require multi-step causal reasoning to evaluate
- NO explanations of why they work 
- Test genuine understanding, not keyword matching

ADDITIONAL INCORRECT OPTIONS Rubric (Generate 4-6 more):
- Include deceptive and plausible wrong answers
- Should complement existing incorrect options
- Mix of: contextually wrong, mechanism failures, obvious contradictions

DISTRACTOR DIFFICULTY LEVELS:
- 50% Near-miss options: Work in similar contexts but fail due to subtle mechanism differences
- 30% Partial truth options: Contain correct elements but missing critical components  
- 20% Edge case failures: Seem reasonable but fail under specific constraints

DECEPTIVE QUALITY PATTERNS:
- Use domain-appropriate terminology that sounds expert-level
- Include options that work for related but different goals
- Create options where timing/context makes them invalid
- Generate options that have worked historically but don't work now
- Include options where the mechanism is partially correct but incomplete
- You may include options that fail at a later step of reasoning. (Recommended)

AVOID OBVIOUS GIVEAWAYS:
- No clearly wrong domains ("gym" for "finding people outdoors")
- No obviously impossible actions ("flying" for human travel)
- No silly/joke options that immediately stand out
- No options that contradict basic science or common sense
- Avoid shallow wrong answers that can be rejected with a keyword check.

QUALITY REQUIREMENTS:
- Each option must force: Multi Action → Multi Effect → Goal Achievement reasoning
- Include edge cases that test deep vs surface understanding.
- Test causal mechanism understanding, not just correlation.
- Each option must require connecting more than 2 distinct concepts to evaluate.
- You can include constraints that make plausible options actually invalid.
- You may include options requiring understanding real-world context and underlying implicit goals.
- Options in the same bin shouldn't mean the same.
- The answers should be directly factual, instead force the reader to think and reason deeply. (Most IMPORTANT)
- WORD LIMIT: Keep all new options the same length or shorter than the longest exisitng option given to you. (IMPORTANT)
- Try to maintain the same word length for both correct and incorrect new options.

FORBIDDEN PATTERNS:
- Shouldn't contain explanatory phrases
- Shouldn't include location descriptions ("park with scenic view" → just "park")
- Shouldn't use conjunctions within options ("and", "but", "or" within single option)
- Shouldn't provide reasoning ("because", "since", "due to")

OUTPUT (JSON):
{{
  "question": "{question}",
  "expanded_correct_options": [
    "existing correct option 1",
    "existing correct option 2", 
    "new correct option 1",
    "new correct option 2",
    "new correct option 3"
  ],
  "expanded_incorrect_options": [
    "existing wrong option",
    "existing wrong option",
    "plausible wrong option",
    "contextual wrong option", 
    "mechanism failure option",
    "obvious wrong option",
    "plausible wrong option",
    "contextual wrong option",
    "mechanism failure option", 
    "obvious wrong option"
  ]
}}

Create a more comprehensive and challenging set.
"""
    messages = [
        {
            "role": "system", 
            "content": "You are a cognitive and logical reasoning expert, skilled in designing assessments that test authentic, multi-step inference. Your specialty is crafting challenging options that force high school students with deep understanding to demonstrate genuine causal understanding rather than surface-level pattern matching. The goal is creating options that require connecting multiple concepts through valid causal chains. Each option must be evaluable independently. Return only valid JSON."
        },
        {
            "role": "user", 
            "content": user_content
        }
    ]
    return messages




def process_single_stage_pipeline(df, output_file_csv):
    print(f"Starting single-stage pipeline for {len(df)} rows.")
    final_output_list = []
    
    for i in range(len(df)):
        row = df.iloc[i]
        question = row['question']
        labels = row['choices']['label']
        texts = row['choices']['text']
        
        correct_label = row['answerKey'] 
        correct_answer = texts[labels == correct_label][0]
        
        # Get incorrect answers
        incorrect_answers = texts[labels != correct_label]
        
        messages = generate_stage1_options(question, correct_answer, incorrect_answers)
        try:
            result = call_gpt4o_mini(messages)
            
            final_output_list.append(result)
            print(f"  Processed row {i} successfully.")
        except Exception as e:
            print(f"  Failed to process row {i}. Skipping.")
        time.sleep(0.01)

    final_df = pd.DataFrame(final_output_list)
    final_df.to_csv(output_file_csv, index=False, encoding='utf-8')
    print(f"\nProcessing complete. {len(final_output_list)} final results saved to '{output_file_csv}'.")

if __name__ == '__main__':
    filtered_df = load_and_filter_csqa_data()
    if not filtered_df.empty:
        output_csv_file = 'csqa_first_stage-5000.csv'
        process_single_stage_pipeline(filtered_df, output_csv_file)