import json
import os
import shutil

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"
BATCH_SIZE = 16

DATA_DIR = "data"
PAIRS_PATH = os.path.join(DATA_DIR, "llm_pairs.jsonl")
PAIRS_RAW = os.path.join(DATA_DIR, "llm_pairs.raw.jsonl")
TEMPLATES_PATH = os.path.join(DATA_DIR, "templates_augmented.json")
TEMPLATES_RAW = os.path.join(DATA_DIR, "templates_augmented.raw.json")

CANONICAL = {
    "person": ("I've never heard of {x}, I don't know who that is.", "Daniel"),
    "place": ("I've never been to {x} in my life.", "the warehouse"),
    "object": ("I've never seen or touched {x}.", "the briefcase"),
    "alibi": ("I was at {x} the whole night and never left.", "my post"),
}

JUDGE_PROMPT = """You are checking labels for a lie-detection dataset.

A suspect in an interrogation said statement A earlier and statement B later, about the same events.

A: "{a}"
B: "{b}"

Answer YES only if ALL of these are true:
- B is said by the suspect about themselves or what they know. Not words of the detective.
- B is a statement, not a question.
- A and B cannot both be true, or B shows the suspect knows something that A denied knowing.

Answer NO if B is a question, hearsay ("I heard", "someone told me"), a guess about other people, an extra detail that fits with A, or anything that could still be true together with A.

Answer with one word: YES or NO."""


def load_model():
    print(f"Loading {MODEL_ID} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    yes_id = tok.encode("YES", add_special_tokens=False)[0]
    no_id = tok.encode("NO", add_special_tokens=False)[0]
    return tok, model, yes_id, no_id


def judge(tok, model, yes_id, no_id, pairs):
    verdicts = []
    for i in range(0, len(pairs), BATCH_SIZE):
        chunk = pairs[i:i + BATCH_SIZE]
        texts = [tok.apply_chat_template(
            [{"role": "user", "content": JUDGE_PROMPT.format(a=a, b=b)}],
            tokenize=False, add_generation_prompt=True) for a, b in chunk]
        inputs = tok(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits[:, -1, :]
        verdicts += (logits[:, yes_id] > logits[:, no_id]).tolist()
        print(f"\r  judged {min(i + BATCH_SIZE, len(pairs))}/{len(pairs)}", end="")
    print()
    return verdicts


def backup(path, raw_path):
    if not os.path.exists(raw_path):
        shutil.copy(path, raw_path)
        print(f"Backed up {path} -> {raw_path}")


def verify_pairs(tok, model, yes_id, no_id):
    if not os.path.exists(PAIRS_PATH):
        print(f"No {PAIRS_PATH}, skipping pairs")
        return
    backup(PAIRS_PATH, PAIRS_RAW)
    with open(PAIRS_RAW, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]

    print(f"\nVerifying {len(rows)} LLM pairs")
    verdicts = judge(tok, model, yes_id, no_id, [(r["a"], r["b"]) for r in rows])
    kept = [r for r, says_lie in zip(rows, verdicts) if says_lie == (r["label"] == "lie")]

    with open(PAIRS_PATH, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    for label in ["lie", "ok"]:
        total = sum(r["label"] == label for r in rows)
        left = sum(r["label"] == label for r in kept)
        print(f"  {label}: kept {left}/{total}")
    dropped = [r for r, says_lie in zip(rows, verdicts) if says_lie != (r["label"] == "lie")]
    print("  Some dropped pairs:")
    for r in dropped[:6]:
        print(f"    [{r['label']}] {r['a']}  ->  {r['b']}")


def verify_templates(tok, model, yes_id, no_id):
    if not os.path.exists(TEMPLATES_PATH):
        print(f"No {TEMPLATES_PATH}, skipping templates")
        return
    backup(TEMPLATES_PATH, TEMPLATES_RAW)
    with open(TEMPLATES_RAW, encoding="utf-8") as f:
        templates = json.load(f)

    for kind, roles in templates.items():
        if kind not in CANONICAL:
            continue
        a_template, filler = CANONICAL[kind]
        a = a_template.format(x=filler)
        for role, want_lie in [("reveals", True), ("safe", False)]:
            lines = roles[role]
            print(f"\nVerifying {kind}/{role} ({len(lines)} lines)")
            verdicts = judge(tok, model, yes_id, no_id, [(a, l.format(x=filler)) for l in lines])
            kept = [l for l, says_lie in zip(lines, verdicts) if says_lie == want_lie]
            dropped = [l for l, says_lie in zip(lines, verdicts) if says_lie != want_lie]
            roles[role] = kept
            print(f"  kept {len(kept)}/{len(lines)}")
            for l in dropped[:4]:
                print(f"    dropped: {l}")

    with open(TEMPLATES_PATH, "w", encoding="utf-8") as f:
        json.dump(templates, f, indent=2, ensure_ascii=False)


def main():
    tok, model, yes_id, no_id = load_model()
    verify_templates(tok, model, yes_id, no_id)
    verify_pairs(tok, model, yes_id, no_id)
    print("\nDone. Originals are kept as *.raw.* files; run this again any time, it always starts from them.")


if __name__ == "__main__":
    main()