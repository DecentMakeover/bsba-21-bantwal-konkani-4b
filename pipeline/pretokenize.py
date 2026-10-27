"""Pre-tokenize the exported corpus (CPU only, offline) into packed blocks for training.
Docs are tokenized in file order, joined with EOS, and cut into SEQ-token blocks (last partial block dropped).
Writes data/tokens/{train,valid}_gemma3_seq1024.npy (uint32) and data/tokens/manifest_v1.json with SHA256 of
inputs/outputs, so training configs can freeze them. Usage: pretokenize.py TOKENIZER_PATH"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
SEQ = 1024


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(tok_path):
    tok = AutoTokenizer.from_pretrained(tok_path)
    out_dir = ROOT / "data/tokens"; out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"created": time.strftime("%F %T"), "tokenizer_path": tok_path, "seq": SEQ, "eos_token_id": tok.eos_token_id,
                "splits": {}}
    for split in ("train", "valid"):
        src = ROOT / f"data/corpus/{split}.jsonl"
        ids, docs = [], 0
        for line in src.open():
            ids.extend(tok(json.loads(line)["text"], add_special_tokens=False)["input_ids"])
            ids.append(tok.eos_token_id); docs += 1
        n = len(ids) // SEQ
        arr = np.asarray(ids[: n * SEQ], dtype=np.uint32).reshape(n, SEQ)
        dst = out_dir / f"{split}_gemma3_seq{SEQ}.npy"
        np.save(dst, arr)
        manifest["splits"][split] = {"source": str(src), "source_sha256": sha256(src), "docs": docs, "tokens": len(ids),
                                     "blocks": n, "path": str(dst), "sha256": sha256(dst)}
        print(f"{split}: {docs} docs, {len(ids):,} tokens, {n:,} blocks", flush=True)
    (out_dir / "manifest_v1.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
