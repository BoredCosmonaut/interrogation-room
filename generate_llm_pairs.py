import argparse
import json
import os
import random
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
OUT_PATH = os.path.join("data", "llm_pairs.jsonl")
BATCH_SIZE = 32
PAIRS_PER_CALL = 8

SETTINGS = [
    "a port warehouse", "a nightclub", "a hospital night shift", "a jewelry store", "a train station",
    "a construction site", "a hotel", "a university lab", "a car dealership", "a restaurant kitchen",
    "an office building", "a farm", "a gas station", "a museum", "a fishing boat", "a parking garage",
    "a bakery", "a factory", "a casino", "an apartment building", "a bus depot", "a pharmacy",
    "a football stadium", "a bank branch", "a shopping mall", "a ferry terminal", "a gym", "a school",
]
CRIMES = [
    "a stolen shipment", "a missing cash box", "a break-in", "a forged document", "a hit-and-run",
    "stolen medicine", "a missing laptop", "an assault in the parking lot", "a fire that was set on purpose",
    "bribery", "a stolen car", "a missing necklace", "sabotaged equipment", "a leaked secret",
    "smuggling", "a burglary", "embezzled money", "a stolen painting",
]
ROLES = [
    "night guard", "cleaner", "manager", "delivery driver", "bartender", "nurse", "mechanic",
    "student", "cashier", "cook", "accountant", "dock worker", "taxi driver", "intern", "receptionist",
]
LIE_TYPES = [
    "claims not to know a person, then later reveals knowing them, referring to them only as 'he', 'she' or 'that guy'",
    "claims not to know a person, then later reveals a personal detail about them",
    "gives an alibi for a time, then later mentions doing something elsewhere at that time, without naming the alibi place",
    "claims never to have been somewhere, then later describes it from personal experience",
    "claims never to have touched or seen an object, then later describes handling it",
    "states a time for an event, then later gives a different time for the same event",
    "states an amount or number, then later gives a different one",
    "describes the order of events one way, then later in a conflicting order",
    "claims to have been alone, then later mentions someone who was with them",
    "claims to have no reason to be involved, then later mentions a motive, like money trouble or a grudge",
]
OK_TYPES = [
    "adds a detail that fits with the first statement",
    "talks about something unrelated to the first statement",
    "repeats the first statement in different words",
    "mentions a name or place from the first statement, but only as a question or deflection",
    "repeats hearsay, something they were told, that does not show personal knowledge",
    "gets angry or emotional without adding facts",
    "gives a detail that sounds suspicious but does NOT conflict with the first statement",
    "describes the time of the first statement in a way that is fully consistent with it",
]

GEN_PROMPT = """You are writing training data for a lie detector in a detective game.

Scenario: a {role} is being interrogated about {crime} at {setting}. Invent names, times and details.

Write {n} pairs of statements made by the suspect: "a" is said earlier, "b" later in the same interrogation.

Write exactly {n_lie} pairs with label "lie": b contradicts a or reveals a was false. Use these patterns:
{lie_list}

Write exactly {n_ok} pairs with label "ok": b does NOT contradict a. Use these patterns:
{ok_list}

Rules:
- Natural spoken English, like a real nervous suspect. 5 to 30 words per statement.
- The contradiction must be clear to a careful reader, but not stated openly. Never write "I lied".
- Vary vocabulary. Do not reuse sentences between pairs.
- Output ONLY one JSON object per line: {{"a": "...", "b": "...", "label": "lie"}} or {{"a": "...", "b": "...", "label": "ok"}}
"""

JUDGE_PROMPT = """A suspect in an interrogation said statement A earlier and statement B later. Both are about the same night and the same events.

A: "{a}"
B: "{b}"

Does B contradict A, or reveal that A was false? Answer YES only if both cannot reasonably be true at the same time. Answer with one word: YES or NO."""


def batched_generate(model, tok, prompts, max_new_tokens, temperature):
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                     add_generation_prompt=True) for p in prompts]
    inputs = tok(texts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=temperature > 0, temperature=temperature if temperature > 0 else None,
            top_p=0.95 if temperature > 0 else None, pad_token_id=tok.pad_token_id,
        )
    new = out[:, inputs["input_ids"].shape[1]:]
    return tok.batch_decode(new, skip_special_tokens=True)


def make_prompt():
    n_lie = PAIRS_PER_CALL // 2
    n_ok = PAIRS_PER_CALL - n_lie
    lies = random.sample(LIE_TYPES, n_lie)
    oks = random.sample(OK_TYPES, n_ok)
    return GEN_PROMPT.format(
        role=random.choice(ROLES), crime=random.choice(CRIMES), setting=random.choice(SETTINGS),
        n=PAIRS_PER_CALL, n_lie=n_lie, n_ok=n_ok,
        lie_list="\n".join(f"- {t}" for t in lies),
        ok_list="\n".join(f"- {t}" for t in oks),
    )


def parse_pairs(raw):
    pairs = []
    for line in raw.splitlines():
        m = re.search(r"\{.*\}", line)
        if not m:
            continue
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        a, b, label = str(obj.get("a", "")).strip(), str(obj.get("b", "")).strip(), obj.get("label")
        if label not in ("lie", "ok") or not a or not b or a == b:
            continue
        if not (3 <= len(a.split()) <= 40 and 3 <= len(b.split()) <= 40):
            continue
        pairs.append({"a": a, "b": b, "label": label})
    return pairs


def judge(model, tok, pairs):
    prompts = [JUDGE_PROMPT.format(a=p["a"], b=p["b"]) for p in pairs]
    answers = []
    for i in range(0, len(prompts), BATCH_SIZE * 2):
        answers += batched_generate(model, tok, prompts[i:i + BATCH_SIZE * 2], max_new_tokens=3, temperature=0)
    return ["YES" in a.upper() for a in answers]


def count_existing():
    if not os.path.exists(OUT_PATH):
        return 0, 0
    with open(OUT_PATH, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    return len(rows), max((r["scenario"] for r in rows), default=-1) + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=5000, help="how many verified pairs to end up with")
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)
    have, next_scenario = count_existing()
    print(f"{have} pairs already in {OUT_PATH}, target {args.target}")
    if have >= args.target:
        return

    print(f"Loading {MODEL_ID} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    stats = {"generated": 0, "kept": 0}
    while have < args.target:
        raws = batched_generate(model, tok, [make_prompt() for _ in range(BATCH_SIZE)],
                                max_new_tokens=1200, temperature=0.9)

        batch = []
        for raw in raws:
            for p in parse_pairs(raw):
                p["scenario"] = next_scenario
                batch.append(p)
            next_scenario += 1
        if not batch:
            continue

        verdicts = judge(model, tok, batch)
        kept = [p for p, says_lie in zip(batch, verdicts) if says_lie == (p["label"] == "lie")]

        with open(OUT_PATH, "a", encoding="utf-8") as f:
            for p in kept:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

        stats["generated"] += len(batch)
        stats["kept"] += len(kept)
        have += len(kept)
        n_lie = sum(p["label"] == "lie" for p in kept)
        print(f"  +{len(kept)}/{len(batch)} kept ({n_lie} lies)  total {have}/{args.target}  "
              f"judge agreement so far {stats['kept'] / stats['generated']:.0%}")

    print(f"\nDone. {have} pairs in {OUT_PATH}")
    with open(OUT_PATH, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    for r in random.sample(rows, min(8, len(rows))):
        print(f"  [{r['label']}] {r['a']}  ->  {r['b']}")


if __name__ == "__main__":
    main()