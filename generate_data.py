

import json
import os
import random

random.seed(42)

FILLERS = {
    "train": {
        "person": ["Alex", "Sarah", "Michael", "Emma", "David", "Olivia", "James",
                   "Sophia", "Daniel", "Grace", "Ryan", "Chloe", "Nathan", "Lucy"],
        "place": ["the warehouse", "the parking lot", "the loading dock", "the bar",
                  "the train station", "the old factory", "the gas station",
                  "the back alley", "the harbor", "the motel"],
        "object": ["the red van", "the briefcase", "the black jacket", "the phone",
                   "the toolbox", "the envelope", "the keys", "the laptop"],
    },
    "val": {
        "person": ["Thomas", "Hannah", "Lucas", "Mia", "Ethan"],
        "place": ["the pier", "the garage", "the diner", "the bridge"],
        "object": ["the backpack", "the white truck", "the wallet", "the camera"],
    },
    "test": {
        "person": ["Victor", "Julia", "Oscar", "Nora", "Felix"],
        "place": ["the casino", "the rooftop", "the storage unit", "the bus depot"],
        "object": ["the suitcase", "the green sedan", "the receipt", "the radio"],
    },
}


TEMPLATES = {
    "person": {
        "denials": [
            "I've never heard of {x}.",
            "I don't know anyone named {x}.",
            "I have no idea who {x} is.",
            "I've never met {x} in my life.",
            "That name, {x}, means nothing to me.",
        ],
        "reveals": [
            "{x} wouldn't do something like that.",
            "{x} lives far from here.",
            "{x} always drinks too much on weekends.",
            "I lent {x} some money last year.",
            "{x} and I used to play cards together.",
            "{x} has a bad temper, everyone knows that.",
            "{x} called me last night.",
            "I saw {x} at the bar on Friday.",
        ],
        "safe": [
            "Who is {x}?",
            "Why do you keep asking me about {x}?",
            "You mentioned {x} earlier, but I still don't know them.",
            "Someone named {x} was in the news, I think.",
            "If {x} said something about me, they're lying.",
            "I've never spoken to {x}.",
        ],
    },
    "place": {
        "denials": [
            "I wasn't anywhere near {x} that night.",
            "I've never been to {x}.",
            "I don't even know where {x} is.",
            "I didn't go to {x}.",
        ],
        "reveals": [
            "The lights at {x} were off when I got there.",
            "It was freezing at {x} that night.",
            "I parked right in front of {x}.",
            "The back door of {x} was unlocked.",
            "I only stayed at {x} for ten minutes.",
            "There was nobody else at {x} when I arrived.",
        ],
        "safe": [
            "Where exactly is {x}?",
            "I've heard {x} is a rough area.",
            "I was home all night, not at {x}.",
            "Why would I go to {x}?",
            "Plenty of people go to {x}, it doesn't mean anything.",
        ],
    },
    "object": {
        "denials": [
            "I never saw {x}.",
            "I don't know anything about {x}.",
            "I've never touched {x}.",
            "{x}? I have no idea what you're talking about.",
        ],
        "reveals": [
            "{x} was already scratched when I found it.",
            "I put {x} back where it was.",
            "{x} was heavier than it looked.",
            "I only held {x} for a second.",
            "Someone else must have moved {x} after I left it.",
        ],
        "safe": [
            "What's so important about {x}?",
            "You'll have to show me {x}, I've never seen it.",
            "If my fingerprints are on {x}, someone planted them.",
            "Is {x} even connected to this case?",
        ],
    },
}


AUGMENTED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates_augmented.json")
K_PER_DENIAL = 6


def load_templates():
    if os.path.exists(AUGMENTED_PATH):
        with open(AUGMENTED_PATH, encoding="utf-8") as f:
            print(f"Using augmented templates from {AUGMENTED_PATH}")
            return json.load(f)
    print("Using built-in templates (run paraphrase.py to augment)")
    return TEMPLATES


def split_templates(templates):
    rng = random.Random(123)
    splits = {"train": {}, "val": {}, "test": {}}
    for kind, roles in templates.items():
        for s in splits.values():
            s[kind] = {}
        for role, items in roles.items():
            items = items[:]
            rng.shuffle(items)
            n = len(items)
            n_test = max(1, round(0.15 * n))
            n_val = max(1, round(0.15 * n))
            splits["test"][kind][role] = items[:n_test]
            splits["val"][kind][role] = items[n_test:n_test + n_val]
            splits["train"][kind][role] = items[n_test + n_val:]
    return splits


def pick(items, k):
    return random.sample(items, min(k, len(items)))


def make_pairs(split_name, split_templates_):
    """Build all examples for one split."""
    fillers = FILLERS[split_name]
    examples = []

    for kind, templates in split_templates_.items():
        values = fillers[kind]

        for x in values:
            for denial in templates["denials"]:
                a = denial.format(x=x)

                for reveal in pick(templates["reveals"], K_PER_DENIAL):
                    examples.append(make_example(a, reveal.format(x=x), 1, kind, "reveal"))

                for safe in pick(templates["safe"], K_PER_DENIAL):
                    examples.append(make_example(a, safe.format(x=x), 0, kind, "safe"))

                other = random.choice([v for v in values if v != x])
                reveal = random.choice(templates["reveals"])
                examples.append(make_example(a, reveal.format(x=other), 0, kind, "other_entity"))

                others = [d for d in templates["denials"] if d != denial]
                if others:
                    examples.append(make_example(a, random.choice(others).format(x=x), 0, kind, "consistent"))

    return examples


def make_example(a, b, label, kind, pair_type):
    a = a[0].upper() + a[1:]
    b = b[0].upper() + b[1:]
    return {
        "statement_a": a,
        "statement_b": b,
        "label": label,
        "kind": kind,           
        "pair_type": pair_type,  
    }


def balance(examples):
    pos = [e for e in examples if e["label"] == 1]
    neg = [e for e in examples if e["label"] == 0]
    n = min(len(pos), len(neg))
    out = random.sample(pos, n) + random.sample(neg, n)
    random.shuffle(out)
    return out


def save_jsonl(examples, path):
    with open(path, "w", encoding="utf-8") as f:
        for e in examples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    splits = split_templates(load_templates())

    for split in ["train", "val", "test"]:
        examples = balance(make_pairs(split, splits[split]))
        save_jsonl(examples, os.path.join(out_dir, f"{split}.jsonl"))

        n_pos = sum(e["label"] for e in examples)
        print(f"{split:5s}: {len(examples):6d} examples ({n_pos} contradictions)")

    print("\nSample:")
    for e in random.sample(examples, 5):
        print(f"  [{e['label']}] {e['statement_a']}  ->  {e['statement_b']}   ({e['pair_type']})")


if __name__ == "__main__":
    main()