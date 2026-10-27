"""Stage-B v2 data (CPU, offline): v1 selection from konkani-instruct-100k (Kannada-script answers + capped
Kannada->Latin rows), but (1) drop rows whose answer opens with English boilerplate ("Introduction:", "The Konkani
Answer", "Here is", "### 1."), (2) strip markdown headers/bold so answers read as plain conversation, (3) mix in
Mangalorean corpus blocks (full loss) at CORPUS_SHARE of all blocks, to keep the dialect active (replay).
Writes data/sft_v2/{train,valid}_{ids,labels}.npy + manifest_v2.json. Usage: prep_sft_v2.py"""
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prep_sft as v1  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TOTAL_TRAIN_BLOCKS, CORPUS_SHARE, VALID_BLOCKS, SEED = 4736, 0.4, 120, 20261021
BOILER = re.compile(r"(introduction\s*:|the konkani answer|here is|here's|translation\s*:|^\s*#+\s*\d)", re.I | re.M)


def clean(text):
    t = re.sub(r"^\s*#+\s*", "", text, flags=re.M)
    t = t.replace("**", "").replace("__", "")
    t = re.sub(r"^\s*[*•]\s+", "- ", t, flags=re.M)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    tok = AutoTokenizer.from_pretrained(v1.TOK); rng = random.Random(SEED)
    out = ROOT / "data/sft_v2"; out.mkdir(parents=True, exist_ok=True)
    man = {"created": time.strftime("%F %T"), "dataset": v1.REPO, "revision": v1.REV, "seed": SEED, "seq": v1.SEQ,
           "corpus_share": CORPUS_SHARE, "splits": {}}
    corpus = np.load(ROOT / "data/tokens/train_gemma3_seq1024.npy", mmap_mode="r")
    for split, fname in (("train", "konkani_Train.jsonl"), ("valid", "konkani_validate.jsonl")):
        src = hf_hub_download(v1.REPO, fname, repo_type="dataset", revision=v1.REV)
        rows = [json.loads(l) for l in open(src)]
        kn_out, kn_to_latin = v1.select(rows, rng)
        dropped_boiler, enc, too_long = 0, [], 0
        for r in kn_out + kn_to_latin:
            if BOILER.search(r["response"][:300]):
                dropped_boiler += 1; continue
            x, y = v1.encode(tok, {**r, "response": clean(r["response"])})
            if len(x) > v1.SEQ:
                too_long += 1; continue
            enc.append((x, y))
        rng.shuffle(enc)
        if split == "train":
            n_corpus = int(TOTAL_TRAIN_BLOCKS * CORPUS_SHARE); n_sft = TOTAL_TRAIN_BLOCKS - n_corpus
        else:
            n_corpus, n_sft = 0, VALID_BLOCKS
        ids, labels = v1.pack(enc, tok.eos_token_id, n_sft)
        assert len(ids) == n_sft, f"{split}: only {len(ids)} instruction blocks, need {n_sft}"
        if n_corpus:
            pick = np.random.default_rng(SEED).permutation(len(corpus))[:n_corpus]
            c = np.asarray(corpus[pick], dtype=np.uint32)
            ids = np.concatenate([ids, c]); labels = np.concatenate([labels, c.astype(np.int32)])
            perm = np.random.default_rng(SEED + 1).permutation(len(ids)); ids, labels = ids[perm], labels[perm]
        ip, lp = out / f"{split}_ids.npy", out / f"{split}_labels.npy"
        np.save(ip, ids); np.save(lp, labels)
        man["splits"][split] = {"rows": len(rows), "kn_out": len(kn_out), "kn_to_latin": len(kn_to_latin),
                                "dropped_boilerplate": dropped_boiler, "dropped_too_long": too_long, "encoded": len(enc),
                                "instruction_blocks": n_sft, "corpus_blocks": n_corpus, "blocks": int(len(ids)),
                                "trained_tokens": int((labels != -100).sum()), "ids": str(ip), "ids_sha256": sha256(ip),
                                "labels": str(lp), "labels_sha256": sha256(lp)}
        print(split, json.dumps({k: v for k, v in man["splits"][split].items() if "sha" not in k and k not in ("ids", "labels")}), flush=True)
    (out / "manifest_v2.json").write_text(json.dumps(man, indent=1) + "\n")
    x, y = enc[0]
    print("example answer:", tok.decode([t for t, l in zip(x, y) if l != -100])[:300])


if __name__ == "__main__":
    main()
