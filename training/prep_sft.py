"""Prepare stage-B instruction data (CPU, offline) from konkani/konkani-instruct-100k (shared HF cache, pinned revision).

Keeps rows whose response is mostly Kannada script (the model should answer in Kannada script), plus up to
CAP_KN_TO_LATIN rows with a Kannada-script instruction and a Latin-script response (translation/transliteration).
Each row -> Gemma chat template (system folded in by the template) ; loss only on the assistant part. Rows longer than
SEQ are dropped; rows are shuffled (seeded) and packed into SEQ-token blocks (pad tail with EOS, label -100).
Train is capped at MAX_TRAIN_BLOCKS (~2 time-limited job slots). Writes data/sft/{train,valid}_{ids,labels}.npy + manifest_v1.json."""
import collections
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
REPO, REV = "konkani/konkani-instruct-100k", "dd18239a9954351753c1039f4c35923663e27edb"
TOK = "unsloth/gemma-3-4b-it"
SEQ, SEED, CAP_KN_TO_LATIN, MAX_TRAIN_BLOCKS, MAX_VALID_BLOCKS = 1024, 20261020, 3000, 5120, 120


def kn_share(t):
    c = collections.Counter()
    for ch in t:
        o = ord(ch)
        if 0x0C80 <= o <= 0x0CFF: c["kn"] += 1
        elif 0x0900 <= o <= 0x097F: c["deva"] += 1
        elif ch.isascii() and ch.isalpha(): c["latin"] += 1
    n = sum(c.values())
    return (c["kn"] / n if n else 0.0), (c["latin"] / n if n else 0.0)


def select(rows, rng):
    kn_out, kn_to_latin = [], []
    for r in rows:
        resp_kn, resp_lat = kn_share(r["response"])
        ins_kn, _ = kn_share(r["instruction"])
        if resp_kn >= 0.5:
            kn_out.append(r)
        elif ins_kn >= 0.6 and resp_lat >= 0.6:
            kn_to_latin.append(r)
    rng.shuffle(kn_to_latin)
    return kn_out, kn_to_latin[:CAP_KN_TO_LATIN]


def encode(tok, r):
    msgs = ([{"role": "system", "content": r["system"]}] if r.get("system") else []) + [
        {"role": "user", "content": r["instruction"]}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": r["response"]}], tokenize=False)
    assert full.startswith(prompt), "chat template prefix mismatch"
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"]
    return f_ids, [-100] * len(p_ids) + f_ids[len(p_ids):]


def pack(examples, eos, max_blocks):
    ids, labels, blocks_i, blocks_l = [], [], [], []
    for x, y in examples:
        if len(ids) + len(x) > SEQ:
            pad = SEQ - len(ids)
            blocks_i.append(ids + [eos] * pad); blocks_l.append(labels + [-100] * pad)
            ids, labels = [], []
            if len(blocks_i) >= max_blocks:
                break
        ids += x; labels += y
    return np.asarray(blocks_i, dtype=np.uint32), np.asarray(blocks_l, dtype=np.int32)


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    tok = AutoTokenizer.from_pretrained(TOK); rng = random.Random(SEED)
    out = ROOT / "data/sft"; out.mkdir(parents=True, exist_ok=True)
    man = {"created": time.strftime("%F %T"), "dataset": REPO, "revision": REV, "tokenizer_path": TOK, "seq": SEQ,
           "seed": SEED, "splits": {}}
    for split, fname, cap in (("train", "konkani_Train.jsonl", MAX_TRAIN_BLOCKS), ("valid", "konkani_validate.jsonl", MAX_VALID_BLOCKS)):
        src = hf_hub_download(REPO, fname, repo_type="dataset", revision=REV)
        rows = [json.loads(l) for l in open(src)]
        kn_out, kn_to_latin = select(rows, rng)
        enc, dropped = [], 0
        for r in kn_out + kn_to_latin:
            x, y = encode(tok, r)
            if len(x) > SEQ:
                dropped += 1; continue
            enc.append((x, y))
        rng.shuffle(enc)
        ids, labels = pack(enc, tok.eos_token_id, cap)
        ip, lp = out / f"{split}_ids.npy", out / f"{split}_labels.npy"
        np.save(ip, ids); np.save(lp, labels)
        trained = int((labels != -100).sum())
        man["splits"][split] = {"source": src, "source_sha256": sha256(src), "rows": len(rows), "kn_out": len(kn_out),
                                "kn_to_latin": len(kn_to_latin), "dropped_too_long": dropped, "encoded": len(enc),
                                "blocks": int(len(ids)), "trained_tokens": trained, "ids": str(ip), "ids_sha256": sha256(ip),
                                "labels": str(lp), "labels_sha256": sha256(lp)}
        print(split, json.dumps({k: v for k, v in man["splits"][split].items() if "sha" not in k and k not in ("source", "ids", "labels")}), flush=True)
    (out / "manifest_v1.json").write_text(json.dumps(man, indent=1) + "\n")
    x, y = enc[0]
    print("example (decoded, assistant part):", tok.decode([t for t, l in zip(x, y) if l != -100])[:300])


if __name__ == "__main__":
    main()
