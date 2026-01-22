import numpy as np
import pandas as pd

import json
import csv

with open("/kaggle/input/csqa-logicalcombinations/test_all_hf.json", "r") as f:
    data = [json.loads(line) for line in f]

with open('test_dataset.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    
    writer.writerow(['question', 'A', 'B', 'C', 'D', 'correct_label', 'correct_answer_text', 'qa_type'])
    
    for item in data:
        choices = item['choices']
        label = item['label']
        correct_answer = choices[label]
        
        label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        label_letter = label_map[label]
        
        correct_answer = choices[label]
        
        row = [
            item['question'],
            choices[0] if len(choices) > 0 else '',
            choices[1] if len(choices) > 1 else '',
            choices[2] if len(choices) > 2 else '',
            choices[3] if len(choices) > 3 else '',
            label_letter,
            correct_answer,
            item['qa_type']
        ]
        writer.writerow(row)

print("Conversion complete! Check test_dataset.csv")

qa_df = pd.read_csv('/kaggle/working/test_dataset.csv')

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_qa_metrics(results_df):
    """
    
    Calculate comprehensive evaluation metrics for QA results
    
    Args:
        results_df: DataFrame with 'predicted_label' and 'correct_label' columns
    
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    
    valid_predictions = results_df.dropna(subset=['predicted_label'])
    
    if len(valid_predictions) == 0:
        print("Warning: No valid predictions found!")
        return {}
    
    y_true = valid_predictions['correct_label'].tolist()
    y_pred = valid_predictions['predicted_label'].tolist()
    
    total_questions = len(results_df)
    answered_questions = len(valid_predictions)
    unanswered_questions = total_questions - answered_questions
    
    accuracy = accuracy_score(y_true, y_pred)
    
    labels = ['A', 'B', 'C', 'D']
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    answer_rate = answered_questions / total_questions
    
    metrics = {
        'total_questions': total_questions,
        'answered_questions': answered_questions,
        'unanswered_questions': unanswered_questions,
        'answer_rate': answer_rate,
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
    }
    
    for i, label in enumerate(labels):
        metrics[f'{label}_precision'] = precision[i]
        metrics[f'{label}_recall'] = recall[i]
        metrics[f'{label}_f1'] = f1[i]
        metrics[f'{label}_support'] = support[i]
    
    return metrics, cm, y_true, y_pred

def print_qa_metrics(metrics):
    print("QA EVALUATION METRICS")
    
    print(f"Total Questions: {metrics['total_questions']}")
    print(f"Answered Questions: {metrics['answered_questions']}")
    print(f"Answer Rate: {metrics['answer_rate']:.1%}")
    print(f"Accuracy: {metrics['accuracy']:.1%}")
    
    print("OVERALL METRICS")
    print(f"Macro Precision: {metrics['macro_precision']:.3f}")
    print(f"Macro Recall: {metrics['macro_recall']:.3f}")
    print(f"Macro F1: {metrics['macro_f1']:.3f}")
    print(f"Micro Precision: {metrics['micro_precision']:.3f}")
    print(f"Micro Recall: {metrics['micro_recall']:.3f}")
    print(f"Micro F1: {metrics['micro_f1']:.3f}")
    
    print("PER-CLASS METRICS")
    print(f"{'Class':<5} {'Precision':<9} {'Recall':<6} {'F1':<6} {'Support'}")
    
    for label in ['A', 'B', 'C', 'D']:
        precision = metrics[f'{label}_precision']
        recall = metrics[f'{label}_recall']
        f1 = metrics[f'{label}_f1']
        support = int(metrics[f'{label}_support'])
        print(f"{label:<5} {precision:<9.3f} {recall:<6.3f} {f1:<6.3f} {support}")

def plot_confusion_matrix(cm, labels=['A', 'B', 'C', 'D'], save_path=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def analyze_answer_distribution(results_df):
    print("ANSWER DISTRIBUTION ANALYSIS")
    
    true_dist = results_df['correct_label'].value_counts().sort_index()
    print("\nTrue Label Distribution:")
    for label in ['A', 'B', 'C', 'D']:
        count = true_dist.get(label, 0)
        pct = count / len(results_df) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    valid_preds = results_df.dropna(subset=['predicted_label'])
    if len(valid_preds) > 0:
        pred_dist = valid_preds['predicted_label'].value_counts().sort_index()
        print("\nPredicted Label Distribution:")
        for label in ['A', 'B', 'C', 'D']:
            count = pred_dist.get(label, 0)
            pct = count / len(valid_preds) * 100
            print(f"  {label}: {count} ({pct:.1f}%)")
    
    print("\nLabel Rotation Check:")
    if len(set(true_dist.values)) == 1:
        print(" Perfect label rotation detected")
    else:
        print(" Uneven label distribution detected")
        
def comprehensive_qa_evaluation(results_df, save_plots=True):
    
    metrics, cm, y_true, y_pred = calculate_qa_metrics(results_df)
    
    if not metrics:
        return
    
    print_qa_metrics(metrics)
    
    analyze_answer_distribution(results_df)
    
    if save_plots:
        plot_confusion_matrix(cm, save_path='confusion_matrix.png')
    else:
        plot_confusion_matrix(cm)
    
    valid_preds = results_df.dropna(subset=['predicted_label'])
    if len(valid_preds) > 0:
        print("DETAILED CLASSIFICATION REPORT")
        print(classification_report(
            valid_preds['correct_label'], 
            valid_preds['predicted_label'],
            labels=['A', 'B', 'C', 'D']
        ))
    
    return metrics


from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    device_map="auto",
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    torch_dtype="auto"
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

print("Model ready for QLoRA fine-tuning!")

def zero_shot_qa(model, tokenizer, qa_dataframe):
    results = []
    required_cols = ['question', 'A', 'B', 'C', 'D', 'correct_label']
    if not all(col in qa_dataframe.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain the following columns: {required_cols}")
    
    valid_choices = {'A', 'B', 'C', 'D'}
    
    for idx, row in qa_dataframe.iterrows():
        predicted_label = None
        try:

            prompt = f"""Answer the following commonsense question by selecting the correct option.
            Instructions: Please respond with ONLY the single, capital letter (A, B, C, or D) that is the correct answer. Do not include any other text, punctuation, or explanation.

Question: {str(row['question'])}

(A) {str(row['A'])}
(B) {str(row['B'])}
(C) {str(row['C'])}
(D) {str(row['D'])}

Answer:"""

            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            
            output_tokens = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=2,
                do_sample=False
            )
            
            generated_token_ids = output_tokens[0][inputs['input_ids'].shape[1]:]
            generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
            
            # 4. Process the generated text for a strict answer
            # Strip all whitespace and common punctuation, then get the first character
            cleaned_text = generated_text.strip().upper().replace('.', '').replace(',', '').replace(' ', '')
            
            # Validate that the first character is a valid choice
            if cleaned_text and cleaned_text[0] in valid_choices:
                predicted_label = cleaned_text[0]

        except Exception as e:
            print(f"Error processing row {idx}: {e}")
        
        results.append({
            'question': str(row['question']),
            'predicted_label': predicted_label,
            'correct_label': str(row['correct_label']).strip().upper(),
            'full_qa': f"""Question: {str(row['question'])} \n
            A. {str(row['A'])}
            B. {str(row['B'])}
            C. {str(row['C'])}
            D. {str(row['D'])}"""
        })
            
    return results