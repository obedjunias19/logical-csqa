# LOGICAL-COMMONSENSEQA: A Benchmark for Logical Commonsense Reasoning

This repository contains the code and resources for our LOGICAL-COMMONSENSEQA paper.

## Abstract

Commonsense reasoning often involves evaluating multiple plausible interpretations rather than selecting a single atomic answer, yet most benchmarks rely on single-label evaluation, obscuring whether statements are jointly plausible, mutually exclusive, or jointly implausible. We introduce **LOGICAL-COMMONSENSEQA**, a benchmark that reframes commonsense reasoning as logical composition over pairs of atomic statements using plausibility-level operators (AND, OR, NEITHER/NOR). Evaluating instruction-tuned, reasoning-specialized, and fine-tuned models under zero-shot, few-shot, and chain-of-thought prompting, we find that while models perform reasonably on conjunctive and moderately on disjunctive reasoning, performance degrades sharply on negation-based questions. **LOGICAL-COMMONSENSEQA** exposes fundamental reasoning limitations and provides a controlled framework for advancing compositional commonsense reasoning.

## Logical Operators

LOGICAL-COMMONSENSEQA uses three plausibility-level composition operators:

| Operator | Interpretation | Example |
|----------|---------------|---------|
| `a AND b` | Both statements independently plausible | Q: *Sammy wanted to go to where the people were. Where might he go?* <br> A: local events AND social venues |
| `a OR b` | At least one statement plausible | Q: *Sammy wanted to go to where the people were. Where might he go?* <br> A: local events OR empty parks |
| `NEITHER a NOR b` | Neither statement plausible | Q: *Sammy wanted to go to where the people were. Where might he go?* <br> A: NEITHER quiet retreats NOR empty parks |

## Dataset Statistics

- **Source**: Extended from CommonsenseQA (Talmor et al., 2019)
- **Total instances**: 19,996
- **Distribution**: 4,999 instances each for AND, OR, NEITHER/NOR, and MIXED
- **Splits**: 11,996 training / 6,000 development / 2,000 test

## Project Structure

```
logical-csqa/
├── data/                              # Dataset files
├── finetuning/                        # Fine-tuning notebooks
│   ├── deberta-v3-ft.ipynb
│   ├── entailer-11b-baseline.ipynb
│   └── flant5-base-ft.ipynb
├── n-shot-prompts/                    # Evaluation scripts
│   ├── AND/                           # AND-specific prompts
│   ├── OR/                            # OR-specific prompts
│   ├── NNOR/                          # NEITHER/NOR-specific prompts
│   ├── MIXED/                         # Mixed logical operator prompts
│   ├── ZERO_SHOT/                     # Zero-shot prompts
│   ├── zero_shot.py
│   ├── one_shot.py
│   ├── two_shot.py
│   ├── threeshot.py
│   └── cot.py
└── threestage-construction-pipeline/  # Dataset construction pipeline
    ├── stage1.py
    ├── stage2.py
    └── stage3.py
```

## Data

The `data/` folder contains the following files:

| File | Description |
|------|-------------|
| `train_all_hf.json` | Training set in HuggingFace format |
| `dev_all_hf.json` | Development/validation set in HuggingFace format |
| `test_all_hf.json` | Test set in HuggingFace format |
| `train_logical_combinations_output.json` | Training set with logical combinations |
| `dev_logical_combinations_output.json` | Development set with logical combinations |
| `test_logical_combinations_output.json` | Test set with logical combinations |
| `logical_combinations_output.json` | Full logical combinations output |
| `csqa_first_stage-5000.csv` | Stage 1 pipeline output (expanded options) |
| `csqa_second_stage-5000.csv` | Stage 2 pipeline output (refined options) |

## Dataset Construction Pipeline

The dataset is built using a three-stage pipeline that integrates neural generation with deterministic symbolic composition.

### Stage 1: Generation of Candidate Options (`stage1.py`)
- Samples 5,000 instances from CommonsenseQA
- Uses GPT-4o-mini to over-generate diverse atomic answer candidates
- Produces both plausible and implausible alternatives
- Emphasizes multi-step causal or situational reasoning
- Yields 4-6 plausible and implausible candidates per question

### Stage 2: Refinement and Pruning (`stage2.py`)
- Uses GPT-4o-mini to refine and filter candidates
- Removes logically inconsistent or factually incorrect answers
- Eliminates trivial options resolvable through keyword matching
- Identifies highly plausible options that fail due to non-obvious commonsense violations
- Yields 3 correct and 4 incorrect atomic options per question

### Stage 3: Deterministic Logical Composition (`stage3.py`)
- Uses a symbolic program to deterministically combine pairs of atomic options
- Labels compositions with AND, OR, or NEITHER/NOR relations
- Produces 14,997 instances across the three base relation types
- Adds MIXED condition with randomly assigned operators

## Experiments

### Models Evaluated

**Decoder-Only LLMs** (zero-shot, few-shot, chain-of-thought):
- LLaMA-3.3-70B-Instruct
- LLaMA-3.1-8B-Instruct
- Qwen2.5-7B-Instruct

**Encoder-Decoder Models** (fine-tuned):
- FLAN-T5-base
- Entailer-11B

**Encoder Models** (fine-tuned):
- DeBERTa-v3-base

### Evaluation Settings

| Strategy | Description |
|----------|-------------|
| Zero-shot | Direct question answering without examples |
| Few-shot (1-3) | Prepending labeled examples before target question |
| Chain-of-Thought | Step-by-step reasoning about plausibility |
| Fine-tuning | Supervised training on the training split |

### Key Results (Macro-F1 on Human-Validated Test Set)

| Model | AND | OR | NEITHER/NOR | MIXED |
|-------|-----|-----|-------------|-------|
| LLaMA-3.3-70B (0-shot) | 80.9 | 70.9 | 13.4 | 53.0 |
| LLaMA-3.1-8B (0-shot) | 71.9 | 62.2 | 13.1 | 41.8 |
| Qwen2.5-7B (0-shot) | 79.6 | 68.9 | 12.9 | 53.2 |
| FLAN-T5-base (fine-tuned) | 92.8 | 92.4 | 89.2 | 89.6 |
| DeBERTa-v3-base (fine-tuned) | 87.6 | 87.2 | 84.8 | 82.4 |

**Key Findings**:
- Models perform reasonably well on conjunctive (AND) reasoning
- Moderate performance on disjunctive (OR) reasoning
- Performance collapses on negation-based (NEITHER/NOR) compositions in zero/few-shot settings
- Fine-tuned models achieve 83-93% F1 across all operators, showing the task is learnable with supervision

## Usage

### Dataset Construction

Run the three-stage pipeline sequentially:

```bash
# Stage 1: Generation of Candidate Options
python "threestage-construction-pipeline/stage1.py"

# Stage 2: Refinement and Pruning
python "threestage-construction-pipeline/stage2.py"

# Stage 3: Deterministic Logical Composition
python "threestage-construction-pipeline/stage3.py"
```

### Model Evaluation

For decoder-only models with different prompting strategies:

```bash
# Zero-shot evaluation
python "n-shot-prompts/ZERO_SHOT/zero_shot.py"

# Few-shot evaluation
python "n-shot-prompts/one_shot.py"
python "n-shot-prompts/two_shot.py"
python "n-shot-prompts/threeshot.py"

# Operator-specific evaluation (e.g., AND)
python "n-shot-prompts/AND/one_shot.py"
python "n-shot-prompts/AND/two_shot.py"
python "n-shot-prompts/AND/three_shot.py"

# Chain-of-thought evaluation
python "n-shot-prompts/cot.py"
```

### Fine-tuning

For encoder and encoder-decoder models, use the Jupyter notebooks in `finetuning/`.

<!-- ## Citation -->

<!-- If you use this dataset or code, please cite our paper:

```

```

## Acknowledgments

This work builds upon the CommonsenseQA dataset (Talmor et al., 2019). We thank the original authors and the research community for their contributions.
