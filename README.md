# CommonsenseQA with Logical Combinations

This repository contains the code and resources for our CommonsenseQA-Logic paper.

## Overview

Or work explores the use of logical operators (AND, OR, NEITHER/NOR) to create more challenging commonsense reasoning tasks, extending the original CommonsenseQA dataset with complex logical combinations.

## Dataset Construction Pipeline

Our dataset is constructed using a three-stage pipeline:

### Stage 1: Option Expansion (`stage1.py`)
- Expands the original CommonsenseQA options
- Generates additional correct and incorrect answer candidates
- Uses GPT-4o-mini to create contextually relevant distractors
- Ensures multi-step reasoning requirements

### Stage 2: Logical Refinement (`stage2.py`)
- Refines expanded options for logical consistency
- Filters options to 2-3 words for clarity
- Removes obvious or trivial answers
- Ensures balanced difficulty across options

### Stage 3: Logical Combination Generation (`stage3.py`)
- Creates AND, OR, and NEITHER/NOR combinations
- Generates correct and incorrect logical combinations
- Produces four question types: AND-only, OR-only, NEITHER-only, and Mixed
- Maintains label rotation for balanced evaluation

## Evaluation

### Decoder-Only Models

We evaluate the following decoder-only models:
- **Llama-3.1-70B-Instruct**
- **Llama-3.1-8B-Instruct**
- **Qwen-2.5-7B-Instruct**

Evaluation scripts are provided in the `n-shot prompts/` directory with multiple prompting strategies:

- **Zero-shot**: Direct question answering without examples
- **One-shot**: Single example before the target question
- **Two-shot**: Two examples before the target question
- **Three-shot**: Three examples before the target question
- **Chain-of-Thought (CoT)**: Step-by-step reasoning with examples

### Encoder-Decoder Models

Fine-tuning code for FLAN-T5 base model is provided in `finetuning/flant5-base-ft.ipynb`.

## Usage

### Dataset Construction

Run the three-stage pipeline sequentially:

```bash
# Stage 1: Option Expansion
python "threestage-construction-pipleine/stage1.py"

# Stage 2: Logical Refinement
python "threestage-construction-pipleine/stage2.py"

# Stage 3: Logical Combination Generation
python "threestage-construction-pipleine/stage3.py"
```

### Model Evaluation

For decoder-only models with different prompting strategies:

```bash
# Zero-shot evaluation
python "n-shot prompts/zero_shot.py"

# Few-shot evaluation
python "n-shot prompts/one_shot.py"
python "n-shot prompts/two_shot.py"
python "n-shot prompts/threeshot.py"

# Chain-of-thought evaluation
python "n-shot prompts/cot.py"
```

### Fine-tuning

For encoder-decoder models, use the Jupyter notebook.

## Acknowledgments

This work builds upon the CommonsenseQA dataset. We thank the original authors and the research community for their contributions.
