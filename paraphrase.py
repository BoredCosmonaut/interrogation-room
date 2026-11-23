
import json
import os
import re
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
from generate_data import TEMPLATES, AUGMENTED_PATH  

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"   
N_PER_TEMPLATE = 15

THING = {"person": "a person", "place": "a place", "object": "an object"}

ROLE_INSTRUCTIONS = {
    "denials": (
        "The suspect DENIES knowing, having been at, or having seen {thing} called {{x}}. "
        "Vary the wording, tone and length: nervous, angry, short, long, casual, formal."
    ),
    "reveals": (
        "The suspect accidentally REVEALS that they DO know {thing} called {{x}}: "
        "a personal detail, a memory, a description, something only someone who knows {{x}} could say. "
        "Do NOT just reword the example: invent NEW, DIFFERENT details each time. "
        "The suspect must not openly admit anything, it should sound like a slip."
    ),
    "safe": (
        "The suspect mentions {thing} called {{x}} but reveals NO knowledge of it: "
        "questions, deflections, hearsay, accusations against the detective, hypotheticals. "
        "Nothing said should prove they know {{x}}."
    ),
}

PROMPT = """You are writing dialogue lines for a detective game. A suspect is being interrogated.

{instruction}

Example line: "{example}"

Write {n} new lines like this. Rules:
- Each line must contain the placeholder {{x}} exactly once, written literally as {{x}}.
- One line per row. No numbering, no quotes, no explanations.
- Natural spoken English, 4 to 20 words.
"""


def clean(line):
    line = line.strip().strip('"').strip()
    line = re.sub(r"^\s*(\d+[\.\)]|[-*•])\s*", "", line)  
    return line.strip().strip('"').strip()


def is_valid(line):
    if line.count("{x}") != 1:
        return False
    if re.search(r"[{}]", line.replace("{x}", "")):  
        return False
    return 4 <= len(line.split()) <= 25


def generate(model, tok, prompt):
    messages = [{"role": "user", "content": prompt}]
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=700, do_sample=True, temperature=0.9, top_p=0.95
        )
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    print(f"Loading {MODEL_ID} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    augmented = {}
    for kind, roles in TEMPLATES.items():
        augmented[kind] = {}
        for role, templates in roles.items():
            pool = set(templates)  
            instruction = ROLE_INSTRUCTIONS[role].format(thing=THING[kind])

            for i, example in enumerate(templates, 1):
                prompt = PROMPT.format(instruction=instruction, example=example, n=N_PER_TEMPLATE)
                raw = generate(model, tok, prompt)
                lines = [clean(l) for l in raw.splitlines()]
                kept = [l for l in lines if is_valid(l)]
                pool.update(kept)
                print(f"  {kind}/{role} {i}/{len(templates)}: kept {len(kept)}/{len([l for l in lines if l])}")

            augmented[kind][role] = sorted(pool)
            print(f"{kind}/{role}: {len(templates)} -> {len(pool)} templates")

    with open(AUGMENTED_PATH, "w", encoding="utf-8") as f:
        json.dump(augmented, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {AUGMENTED_PATH}")

    for kind in augmented:
        for role in ["reveals", "safe"]:
            print(f"\n{kind}/{role} samples:")
            for line in sorted(augmented[kind][role], key=hash)[:4]:
                print("   ", line)


if __name__ == "__main__":
    main()