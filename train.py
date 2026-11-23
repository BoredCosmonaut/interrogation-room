import random

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from baseline import load_jsonl,report

MODEL_ID = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
SAVE_DIR = "checkpoints/detector"

EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
SEED = 42
USE_BF16 = False

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_loader(examples,tokenizer,shuffle):
    def collate(batch):
        inputs = tokenizer(
            [e["statement_a"] for e in batch],
            [e["statement_b"] for e in batch],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        inputs['labels'] = torch.tensor([e['label'] for e in batch])
        return inputs
    return DataLoader(examples,batch_size = BATCH_SIZE, shuffle = shuffle,collate_fn = collate)

def predict(model,loader,device):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            logits = model(**batch).logits
            preds.extend(logits.argmax(dim = 1).tolist())
    return preds

def f1_score(examples, preds):
    labels = [e["label"] for e in examples]
    tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    global USE_BF16
    USE_BF16 = device == "cuda" and torch.cuda.is_bf16_supported()
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)}, bf16)" if USE_BF16 else ""))
 
    train =load_jsonl("data/data/train.jsonl")
    val = load_jsonl("data//data/val.jsonl")
    test = load_jsonl("data/data/test.jsonl")
    print(f"train={len(train)}  val={len(val)}  test={len(test)}")
 
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=2, ignore_mismatched_sizes=True
    ).to(device)
    torch.nn.init.normal_(model.classifier.weight, std=0.02)
    torch.nn.init.zeros_(model.classifier.bias)
 
    train_loader = make_loader(train, tokenizer, shuffle=True)
    val_loader = make_loader(val, tokenizer, shuffle=False)
    test_loader = make_loader(test, tokenizer, shuffle=False)
 
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
 
    best_f1 = -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=USE_BF16):
                loss = model(**batch).loss   # HF computes cross-entropy for us when labels are given
 
            if torch.isnan(loss):
                raise RuntimeError(f"Loss became NaN at epoch {epoch}, step {step}")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
 
            total_loss += loss.item()
            print(f"\r  epoch {epoch} step {step}/{len(train_loader)}  loss {total_loss / step:.4f}", end="")
        print()
 
        val_preds = predict(model, val_loader, device)
        val_f1 = f1_score(val, val_preds)
        print(f"  val F1: {val_f1:.3f}")
 
        if val_f1 > best_f1:
            best_f1 = val_f1
            model.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)
            print(f"  saved new best model to {SAVE_DIR}")
 
    print("\nLoading best model for the test set...")
    model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR).to(device)
    test_preds = predict(model, test_loader, device)
    report(test, test_preds)
 
 
if __name__ == "__main__":
    main()
