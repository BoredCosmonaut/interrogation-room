import json
import sys
from collections import defaultdict

import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification

MODEL_ID = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
BATCH_SIZE = 32

def load_jsonl(path):
    with open(path,encoding="utf-8") as f:
        return [json.loads(line) for line in f]
    
def predict(examples,tokenizer,model,device):
    contra_idx = model.config.label2id["contradiction"]
    preds = []

    for i in range(0,len(examples),BATCH_SIZE):
        batch = examples[i:i + BATCH_SIZE]
        inputs = tokenizer(
            [e["statement_a"] for e in batch],
            [e["statement_b"] for  e in batch],
            padding = True,
            trunication = True,
            return_tensors = "pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits

        batch_preds = (logits.argmax(dim = 1) == contra_idx).long().tolist()
        preds.extend(batch_preds)

        print(f"\r  {min(i + BATCH_SIZE, len(examples))}/{len(examples)}", end="")
    print
    return preds

def report(exapmles,preds):
    labels = [e["label"] for e in exapmles]

    tp = sum(1 for p,y in zip(preds,labels) if p == 1 and y ==1)
    fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
    correct = sum(1 for p, y in zip(preds, labels) if p == y)

    accuracy = correct / len(labels)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    print(f"\nAccuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}   (when it says 'lie', how often is it right?)")
    print(f"Recall:    {recall:.3f}   (how many lies does it catch?)")
    print(f"F1:        {f1:.3f}")

    by_type = defaultdict(lambda : [0,0])
    for e,p in zip(exapmles,preds):
        by_type[e["pair_type"]][0] += int(p == e["label"])
        by_type[e["pair_type"]][1] += 1
    print("\nAccuracy by pair type:")
    for t,(c,n) in sorted(by_type.items()):
        print(f"  {t:13s} {c / n:.3f}  ({c}/{n})")
        mistakes = [(e, p) for e, p in zip(exapmles, preds) if p != e["label"]]
    print(f"\nSome mistakes ({len(mistakes)} total):")
    for e, p in mistakes[:8]:
        print(f"  true={e['label']} pred={p}  {e['statement_a']}  ->  {e['statement_b']}")

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/data/test.jsonl"
    examples = load_jsonl(path)
    print(f"Loaded {len(examples)} examples from {path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer =AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(device)
    model.eval()

    preds = predict(examples,tokenizer,model,device)
    report(examples,preds)

if __name__ == "__main__":
    main()