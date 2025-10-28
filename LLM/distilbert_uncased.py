import sys
import os
import time
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from datasets import Dataset, DatasetDict
from peft import get_peft_model, LoraConfig, TaskType


current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(current_dir)


if parent_dir not in sys.path:
    sys.path.append(parent_dir)


from HelperModule.helper_functions import get_df

SAMPLE_FRAC = 0.05
MODEL_NAME = "distilbert-base-uncased"
MAX_SEQ_LENGTH = 256
OUTPUT_DIR = "./cicids_llm_finetuned"
NUM_EPOCHS = 1
BATCH_SIZE = 16
LEARNING_RATE = 2e-5

df = get_df(sample_frac=SAMPLE_FRAC)
df = df.replace([np.inf, -np.inf], np.nan)
df.dropna()

features = [c for c in df.columns if c not in "Label"]
label_col = "Label"

X = df[features].copy()
y_str = df[label_col].copy()

print("Converting tabular data to text")


def row_to_text(row):
    parts = [
        f"{col.replace(' ', '_')}: {val:.4f}"
        if isinstance(val, (float, np.floating))
        else f"{col.replace(' ', '_')}:{val}"
        for col, val in row.items()
    ]
    return ",".join(parts)


text_data = X.apply(row_to_text, axis=1).tolist()
del X

lbe = LabelEncoder()
y = lbe.fit_transform(y_str)
num_classes = len(lbe.classes_)
data = {"text": text_data, "label": y}
full_dataset = Dataset.from_dict(data)
del text_data

full_dataset = full_dataset.class_encode_column("label")
print("datset features  ", full_dataset.features)
train_test_split_dict = full_dataset.train_test_split(
    test_size=0.2, seed=42, stratify_by_column="label"
)
train_val_split_dict = train_test_split_dict["train"].train_test_split(
    test_size=0.15, seed=42, stratify_by_column="label"
)

dataset_dict = DatasetDict(
    {
        "train": train_val_split_dict["train"],
        "validation": train_val_split_dict["test"],
        "test": train_test_split_dict["test"],
    }
)

print("Loading tokenizer and model")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=num_classes, attn_implementation="flash_attention_2"
)

print("Setting up lora ")
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_lin", "k_lin", "v_lin", "out_lin", "ffn.lin1", "ffn.lin2"],
)

model = get_peft_model(base_model, lora_config)


def tokenizer_function(examples):
    return tokenizer(
        examples["text"], padding="longest", truncation=True, max_length=MAX_SEQ_LENGTH
    )


tokenized_dataset = dataset_dict.map(tokenizer_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset.set_format("torch")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1_weighted": f1}


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    gradient_accumulation_steps=2,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("Starting training")
start_time = time.time()
trainer.train()
training_time = time.time() - start_time

print(f"Finished training in {training_time:.2f}s")



print("\nEvaluating on Test Set...")
test_results = trainer.evaluate(eval_dataset=tokenized_dataset["test"])
print("\nTest Set Evaluation Results:")
print(test_results)

# --- 9. Save the Final Model (PEFT Adapters Only) ---
final_model_path = os.path.join(OUTPUT_DIR, "final_model")
print(f"\nSaving final PEFT adapter model to {final_model_path}")
model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)  # Save tokenizer with the adapter
print("Model saved.")
