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

def load_and_filter_csqa_stage1_data():
    print("Loading CSQA dataset...")
    df = pd.read_csv('/kaggle/input/csqa-dataset/csqa_first_stage-5000.csv')
    
    print(f"Loaded and filtered {len(df)} text-only questions.")
    return df

def refine_logical_options(question, generated_correct, generated_incorrect):
   user_content = f"""
You are a reasoning assessment expert who creates sophisticated common sense challenges.

TASK: Refine ALL provided options to create sophisticated reasoning challenges.

QUESTION: {question}
GENERATED CORRECT: {generated_correct}
GENERATED INCORRECT: {generated_incorrect}


RUBRIC for generating or modifying correct answer. This has to be followed STRICTLY.

CORRECT OPTION REFINEMENT:
1. Remove any logically flawed answer. 
    a. Read the question first, understand what is being asked.
    b. Now from the list of generated correct, filter any irrelevant options.
    
2. Remove obvious/direct answers that don't require reasoning
   - "populated areas" → "festivals" (requires: festivals → crowds → people)
   - "kitchen" → "cooking spaces" (requires: cooking → aromas → attraction)

3. Eliminate explanation creep while preserving reasoning requirement
   - "traveling to a conference" → "airport travel"
   - "resentment from unreciprocated efforts" → "resentment"

4. Ensure all correct options require 2-3 step inference
   - Test: Can this be answered by keyword matching alone?
   - If yes, make more abstract or require contextual reasoning

5. Balance difficulty with incorrect options
   - Correct options shouldn't be obviously easier than refined incorrect ones

LOGICAL CONSISTENCY CHECK:
- If question asks about "rooms," all options must be actual rooms/spaces
- If question asks about "equipment," all options must be actual equipment
- If question asks about "activities," all options must be actual activities

REJECT options that:
- Mix categories (activities when rooms needed)
- Use impossible locations (gardens for "rooms in house")
- Reference wrong scale (shelves when rooms needed)

IMPORTANT: Finally give me only 3 correct answers which follow the above metioned instruction and require multi step inference or reasoning. No more than 3.

CORE PRINCIPLE: Create incorrect options that satisfy most criteria but fail on one critical contextual nuance requiring 2-3 steps of reasoning to identify.
CRITICAL REQUIREMENT: Every incorrect option must pass the "80% plausible" test but fail on exactly ONE subtle contextual nuance.

SUBTLE FAILURE PATTERNS (study these, don't copy domains):

Pattern 1 - Social Context Mismatch:
Concept: People present, but wrong social accessibility
Think: Roadblocks have people, but official/temporary, not social gathering

Pattern 2 - Function/Purpose Deviation:
Concept: Right category, but serves different specific function  
Think: Wine cellars have aromas, but alcohol-based, not food-based

Pattern 3 - Temporal/Situational Constraints:
Concept: Works in some conditions, fails in implied scenario
Think: Reception areas have seating, but waiting anxiety, not relaxation

Pattern 4 - Intensity/Appropriateness Mismatch:
Concept: Right activity type, but wrong intensity or social convention
Think: Study halls are quiet, but require focus, not passive rest

TRANSFORMATION PROCESS:
1. Extract core concepts to 2-3 words maximum
2. For incorrect options, identify what makes them 80% correct
3. Ensure they fail on exactly one critical nuance
4. Test: Would this trip both humans and models who don't analyze deeply?

SOME TRANSFORMATION EXAMPLES:
Bad: "fairy tale" for fox seeking (obviously wrong domain)
Good: "urban parks" for fox seeking (parks have nature/animals, but wrong wildness level)

Bad: "chicken coop" for fox seeking (obviously wrong)
Good: "abandoned lots" for fox seeking (empty spaces foxes might explore, but lack food sources)

SOPHISTICATION TARGET:
Each incorrect option should make someone think: "That could work because..." before realizing the subtle flaw.

AVOID:
- Domain copying from examples above
- Obvious impossibilities or silly options  
- Easy rejections based on keywords
- Options that fail at step 1 of reasoning

REFINEMENT INSTRUCTIONS:
- Take each option from the generated lists
- Transform to 2-3 words maximum while preserving reasoning requirement
- For correct options: Remove obviousness, ensure inference needed
- For incorrect options: Add subtle contextual failures

STRICT ENFORCEMENT:
- REJECT any option exceeding 3 words
- NO explanatory text ("while", "they have", "rather than")
- NO reasoning phrases ("because", "but", "often")
- OUTPUT FORMAT: Simple word list only

BEFORE FINALIZING - Validate each option:
1. Count words (must be ≤ 3)
2. Check for explanations (reject if found)
3. Verify reasoning requirement (not obvious)

IMPORTANT: Finally give me only 4 incorrect answers which follows the above metioned instruction and require multi step inference or reasoning. No more than 3.


OUTPUT (JSON):
{{
 "question": "{question}",
 "refined_correct_options": [refined versions of ALL generated_correct items],
 "refined_incorrect_options": [refined versions of ALL generated_incorrect items],
}}


Create contextual reasoning traps, not obvious domain mismatches.
"""
   
   messages = [
       {"role": "system", "content": "Expert at creating sophisticated reasoning challenges with subtle contextual failures. Return only JSON."},
       {"role": "user", "content": user_content}
   ]
   return messages




def process_single_stage_pipeline(df, output_file_csv):
    print(f"Starting single-stage pipeline for {len(df)} rows.")
    final_output_list = []
    
    for i in range(len(df)):
        row = df.iloc[i]
        question = row['question']
        correct_answer = row['expanded_correct_options']
        incorrect_answers = row['expanded_incorrect_options']
        
        messages = refine_logical_options(question, correct_answer, incorrect_answers)
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
    filtered_df = load_and_filter_csqa_stage1_data()
    if not filtered_df.empty:
        output_csv_file = 'csqa_second_stage-5000.csv'
        process_single_stage_pipeline(filtered_df, output_csv_file)