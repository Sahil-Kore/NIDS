import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset , DataLoader

import numpy as np

import os 
import sys
import time 
current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(current_dir)


if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
    
from  HelperModule.helper_functions import get_df

class BCosLayer(nn.Module):
    def __init__(self , in_features , out_features):
        super(BCosLayer , self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features , in_features))
        nn.init.kaiming_uniform_(self.weight , a = np.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.scaling = nn.Parameter(torch.ones(out_features))

    
    def forward(self , x ):
        x_norm = F.normalize(x , p=2 , dim = 1 )
        weight_norm = F.normalize(self.weight , p = 2, dim = 1)
        cosine_similarity = F.linear(x_norm , weight_norm)
        output = self.scaling.unsqueeze(0) * (cosine_similarity + self.bias.unsqueeze(0))
        return output


class BCosNetwork(nn.Module):
    def __init__(self , input_size , hidden_size , num_classes):
        super(BCosNetwork , self).__init__()
        self.network = nn.Sequential(
            BCosLayer(input_size , hidden_size),
            nn.ReLU(),
            BCosLayer(hidden_size ,hidden_size//2),
            nn.ReLU(),
            BCosLayer(hidden_size//2 , num_classes)
        )
        
    
    def forward(self , x ):
        return self.network(x)

df = get_df(sample_frac=1.0)
df = df.replace([np.inf , -np.inf] , np.nan)

features = [col for col in df.columns if col not in ["Label"]]
label_col = "Label"

X= df[features].copy()
y_str = df[label_col].copy()

lbe = LabelEncoder()
y = lbe.fit_transform(y_str)
num_classes = len(lbe.classes_)

X_train_val , X_test , y_train_val , y_test = train_test_split(X , y , test_size=0.15 , stratify= y , random_state= 42 )

X_train , X_val ,y_train , y_val = train_test_split(X_train_val , y_train_val, test_size=0.15 , random_state=42 , stratify= y_train_val)

imputer = SimpleImputer(strategy= "median")
X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

BATCH_SIZE = 2048 # Adjust based on GPU memory

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

INPUT_SIZE = X_train.shape[1] # Number of features
HIDDEN_SIZE = 256         # Hyperparameter
LEARNING_RATE = 0.001
EPOCHS = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

model = BCosNetwork(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

model = torch.compile(model)
# --- 6. Training Loop ---
print("\nStarting training...")

start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start_time = time.time()
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # --- Validation Step ---
    model.eval()
    val_loss = 0.0
    all_val_preds = []
    all_val_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_val_preds.extend(predicted.cpu().numpy())
            all_val_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = accuracy_score(all_val_labels, all_val_preds)
    # Use 'weighted' F1 for imbalanced multiclass, 'macro' for unweighted average
    val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted')

    epoch_time = time.time() - epoch_start_time
    print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
          f"Val Acc: {val_accuracy:.4f}, Val F1 (Weighted): {val_f1:.4f}, Time: {epoch_time:.2f}s")

training_time = time.time() - start_time
print(f"\nFinished training. Total time: {training_time:.2f}s")

# --- 7. Final Evaluation on Test Set ---
print("\nEvaluating on Test Set...")
model.eval()
all_test_preds = []
all_test_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_test_preds.extend(predicted.cpu().numpy())
        all_test_labels.extend(labels.cpu().numpy())

test_accuracy = accuracy_score(all_test_labels, all_test_preds)
test_f1_weighted = f1_score(all_test_labels, all_test_preds, average='weighted')
test_f1_macro = f1_score(all_test_labels, all_test_preds, average='macro')

print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test F1 Score (Weighted): {test_f1_weighted:.4f}")
print(f"Test F1 Score (Macro): {test_f1_macro:.4f}")