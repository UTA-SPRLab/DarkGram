import os
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from tqdm import tqdm
import torch.nn.functional as F

# Hyperparameters
EPOCHS = 3
BATCH_SIZE = 16
MAX_LEN = 128
NUM_LABELS = 6  

# Custom dataset class
class MessageDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]

        if pd.isna(text):
            text = ""

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_epoch(model, data_loader, optimizer, device):
    model.train()
    losses = []
    correct_predictions = 0

    for d in tqdm(data_loader, desc="Training"):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        _, preds = torch.max(outputs.logits, dim=1)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return correct_predictions.double() / len(data_loader.dataset), sum(losses) / len(data_loader)

def eval_model(model, data_loader, device, category_names):
    model.eval()
    correct_predictions = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for d in tqdm(data_loader, desc="Evaluating"):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)

            correct_predictions += torch.sum(preds == labels)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Get metrics per category
    report = classification_report(all_labels, all_preds, target_names=category_names, output_dict=True)

    return correct_predictions.double() / len(data_loader.dataset), report

def save_report(report, fold, epoch):
    output_folder = "evolution"
    os.makedirs(output_folder, exist_ok=True)
    file_path = os.path.join(output_folder, f"fold_{fold}_epoch_{epoch}.txt")
    
    with open(file_path, "w") as f:
        f.write(f"Fold {fold} - Epoch {epoch}\n\n")
        for category, metrics in report.items():
            if isinstance(metrics, dict):
                f.write(f"Category: {category}\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall: {metrics['recall']:.4f}\n")
                f.write(f"  F1-score: {metrics['f1-score']:.4f}\n")
                f.write(f"  Support: {metrics['support']}\n\n")
        f.write("\n")

def main(mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=NUM_LABELS).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    all_data = []

    # Load data
    for category in os.listdir('training_data'):
        category_path = os.path.join('training_data', category)
        if os.path.isfile(category_path) and category.endswith(".csv"):
            df = pd.read_csv(category_path)
            df['category'] = category

            # Remove rows where 'message' is NaN
            df = df.dropna(subset=['message'])

            if mode == 'test':
                df = df.head(100)  # Limit to 100 rows per category in test mode
            all_data.append(df)

    # Concatenate all category data
    df_all = pd.concat(all_data)

    X = df_all['message'].values
    y, category_names = df_all['category'].factorize()  

    kfold = KFold(n_splits=3, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        print(f'Fold {fold + 1}')
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        train_dataset = MessageDataset(X_train, y_train, tokenizer, MAX_LEN)
        test_dataset = MessageDataset(X_test, y_test, tokenizer, MAX_LEN)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

        for epoch in range(EPOCHS):
            print(f"Epoch {epoch + 1}/{EPOCHS}")
            train_acc, train_loss = train_epoch(model, train_loader, optimizer, device)
            print(f"Training accuracy: {train_acc}, loss: {train_loss}")

            val_acc, report = eval_model(model, test_loader, device, category_names)
            print(f"Validation accuracy: {val_acc}")
            print("Category-wise performance:")
            for category, metrics in report.items():
                if isinstance(metrics, dict):
                    print(f"Category: {category}")
                    print(f"  Precision: {metrics['precision']:.4f}")
                    print(f"  Recall: {metrics['recall']:.4f}")
                    print(f"  F1-score: {metrics['f1-score']:.4f}")
                    print(f"  Support: {metrics['support']}")

            # Save the report for this fold and epoch
            save_report(report, fold + 1, epoch + 1)

    model.save_pretrained('model')

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ['test', 'full']:
        print("Usage: script.py [test|full]")
        sys.exit(1)

    mode = sys.argv[1]
    main(mode)

