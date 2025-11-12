import os 
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset, DatasetDict , Features , ClassLabel, Value
from tqdm import tqdm
import pandas as pd 
import numpy as np 
from sklearn.metrics import accuracy_score, precision_score, recall_score , f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn 


train_val_df = pd.read_parquet("../Data/filtered_data/train.parquet")
test_df = pd.read_parquet("../Data/filtered_data/test.parquet").sample(n = 5000).reset_index(drop = True)


features = train_val_df.drop("Label" , axis =1).columns
label = "Label"

le = LabelEncoder()

train_val_df[label] = le.fit_transform(train_val_df[label])
test_df[label] = le.transform(test_df[label])
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_val_df["Label"]), # y_train_val from your split
    y=train_val_df["Label"]
)

# Convert to a PyTorch tensor
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to("cuda")

benign_df = train_val_df[train_val_df[label] == le.transform(["Benign"])[0]]
train_val_df = train_val_df.sample(n = 12_000).reset_index( drop = True)

train_df , val_df = train_test_split(
    train_val_df,
    test_size= 0.2,
    stratify= train_val_df[label],
    random_state= 42
)
class_names = le.classes_
num_labels = len(class_names)

means = benign_df[features].mean()
std = benign_df[features].std()

def create_feature_string (row , features , means , std):
    string = "Label with:\n"
    for attr in features :
        value = row[attr]
        
        if value > means[attr] + 1.5 * std[attr]:
            string += f" {attr}: extremely high \n"
        elif value > means[attr] + 1.0 * std[attr]:
            string += f" {attr}: high \n"
        elif value > means[attr] + 0.75 * std[attr]:
            string += f" {attr}: slightly high\n"
        elif value < means[attr] - 1.5 * std[attr]:
            string += f" {attr}: extremely low\n"
        elif value < means[attr] - 1.0 * std[attr]:
            string += f" {attr}: low\n"
        elif value < means[attr] - 0.75 * std[attr]:
            string += f" {attr}: slightly low\n"
        else:
            string += f" {attr}: balanced\n"
    return string


def process_dataframe(df):
    texts = []
    labels = []
    for _ , row in tqdm(df.iterrows() , total = len(df) , desc= "Processing rows"):
        texts .append(create_feature_string(row , features , means ,std))
        labels.append(row[label])
        
    return texts , labels

train_texts , train_labels = process_dataframe(train_df)
val_texts , val_labels = process_dataframe(val_df)
test_texts , test_labels = process_dataframe(test_df)

ds_features = Features({
    'text':Value(dtype = "string"),
    'label':ClassLabel(num_classes=num_labels, names = class_names.tolist())
})

train_dataset = Dataset.from_dict ({"text": train_texts, "label": train_labels} , features = ds_features)
val_dataset = Dataset.from_dict ({"text": val_texts, "label": val_labels} , features = ds_features)
test_dataset= Dataset.from_dict ({"text": test_texts , "label": test_labels} , features= ds_features)

sft_datset = DatasetDict({
    "train":train_dataset,
    "validation":val_dataset,
    "test":test_dataset
})

print(sft_datset)

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"] , padding = "max_length" , truncation = True)

cpu_count = os.cpu_count()
tokenized_dataset = sft_datset.map(tokenize_function , batched = True , num_proc=cpu_count - 2)
tokenized_dataset = tokenized_dataset.rename_column("label" , "labels")

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels = num_labels,
    )
def compute_metrics (eval_pred):
    logits , labels = eval_pred
    predictions = np.argmax(logits, axis = -1)

    acc = accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions , average= "macro" , zero_division= 0)
    recall = recall_score(labels , predictions , average= "macro" , zero_division= 0)
    f1 = f1_score(labels , predictions , average= "macro" , zero_division=0)

    return {
        "accuracy": acc,
        "precision":prec,
        "recall": recall,
        "f1": f1
    }
    
    
    
training_arguments = TrainingArguments(
    output_dir="./distilbert_multiclass_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=15,
    per_device_train_batch_size=24,
    per_device_eval_batch_size=32,
    warmup_steps = 500,
    weight_decay= 0.01,
    logging_dir = "./logs",
    logging_steps=50,
    fp16 = True,
    load_best_model_at_end= True,
    metric_for_best_model="f1",
    greater_is_better= True,
    torch_compile= True
)


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False , **kwargs):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (pass class weights)
        loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    
    
trainer = CustomTrainer(
    model = model,
    args= training_arguments,
    train_dataset= tokenized_dataset["train"],
    eval_dataset = tokenized_dataset["validation"],
    compute_metrics= compute_metrics,
)

print("Starting training")
trainer.train()

print("Evaluating best model on validation set...")
val_results = trainer.evaluate(tokenized_dataset["validation"])

test_results = trainer.evaluate(tokenized_dataset["test"])

print("\n--- Final Test Results ---")
print(test_results)

model_name = "DistilBERT_SFT"
results_csv_file = "../model_evaluation_results.csv"

results_data = [
    {
        "model_name": model_name,
        "dataset": "val",  
        "acc": val_results['eval_accuracy'],
        "pre": val_results['eval_precision'],
        "recall": val_results['eval_recall'],
        "f1": val_results['eval_f1']
    },
    {
        "model_name": model_name,
        "dataset": "test", 
        "acc": test_results['eval_accuracy'],
        "pre": test_results['eval_precision'],
        "recall": test_results['eval_recall'],
        "f1": test_results['eval_f1']
    }
]

df_new_results = pd.DataFrame(results_data)

if os.path.exists(results_csv_file):
    df_new_results.to_csv(results_csv_file, mode='a', header=False, index=False)
    print(f"Appended results for '{model_name}' to {results_csv_file}")
else:
    df_new_results.to_csv(results_csv_file, mode='w', header=True, index=False)
    print(f"Created new results file: {results_csv_file}")


import joblib

joblib.dump(le, 'label_encoder.joblib')