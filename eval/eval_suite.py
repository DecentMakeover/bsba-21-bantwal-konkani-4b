"""Automatic scorecard for Konkani models (one job). For each model in the frozen protocol:
  - perplexity on in-domain validation blocks and on held-out-channel blocks (channels absent from training)
  - Konkani-Bench (200 human-checked pairs): Konkani(Kannada script)->English and English->Konkani, corpus chrF
    (char 6-gram, beta 2, whitespace ignored) against the references, greedy, batched
  - everyday requests (English prompt, reply asked in Mangalorean Konkani): share of replies mostly in Kannada
    script ("stays Konkani") and Mangalorean word share (Kannada-script words found in the title lexicon)
Model = base + merged Konkani LLM adapter + optional extra merged adapters + optional LoRA adapter.
Writes <output_dir>/scorecard.json and outputs_<model>.json. Usage: eval_suite.py PROTOCOL.json [--device cpu] [--dry]"""
import argparse
import collections
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

KN = re.compile(r"[ಀ-೿]+")


def chrf(hyps, refs, n=6, beta=2.0):
    match, hyp_tot, ref_tot = [0] * n, [0] * n, [0] * n
    for h, r in zip(hyps, refs):
        h, r = re.sub(r"\s+", "", h), re.sub(r"\s+", "", r)
        for k in range(1, n + 1):
            hc = collections.Counter(h[i:i + k] for i in range(len(h) - k + 1))
            rc = collections.Counter(r[i:i + k] for i in range(len(r) - k + 1))
            match[k - 1] += sum((hc & rc).values()); hyp_tot[k - 1] += sum(hc.values()); ref_tot[k - 1] += sum(rc.values())
    ks = [k for k in range(n) if hyp_tot[k] and ref_tot[k]]
    if not ks:
        return 0.0
    p = sum(match[k] / hyp_tot[k] for k in ks) / len(ks); r = sum(match[k] / ref_tot[k] for k in ks) / len(ks)
    return 0.0 if p + r == 0 else 100 * (1 + beta ** 2) * p * r / (beta ** 2 * p + r)


def kn_letter_share(t):
    kn = sum(0x0C80 <= ord(c) <= 0x0CFF for c in t); lat = sum(c.isascii() and c.isalpha() for c in t)
    return kn / (kn + lat) if kn + lat else 0.0


def load(P, spec, device):
    m = AutoModelForImageTextToText.from_pretrained(P["base_path"], dtype=torch.bfloat16, device_map=device)
    for path in [P["konkani_adapter_path"]] + spec.get("merge", []):
        m = PeftModel.from_pretrained(m, path).merge_and_unload()
    if spec.get("adapter"):
        m = PeftModel.from_pretrained(m, spec["adapter"])
    return m.eval()


@torch.no_grad()
def ppl(model, blocks, n, mb, device):
    tot = 0.0; cnt = 0
    for i in range(0, n, mb):
        x = torch.from_numpy(np.asarray(blocks[i:min(i + mb, n)]).astype(np.int64)).to(device)
        loss = model(input_ids=x, attention_mask=torch.ones_like(x), token_type_ids=torch.zeros_like(x), labels=x).loss
        tot += loss.item() * x.shape[0]; cnt += x.shape[0]
    return math.exp(tot / cnt)


@torch.no_grad()
def generate(model, tok, prompts, max_new, bs, device):
    tok.padding_side = "left"
    outs = []
    for i in range(0, len(prompts), bs):
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
                 for p in prompts[i:i + bs]]
        x = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        y = model.generate(**x, max_new_tokens=max_new, do_sample=False, repetition_penalty=1.1, pad_token_id=tok.pad_token_id)
        outs += [tok.decode(r[x["input_ids"].shape[1]:], skip_special_tokens=True).strip() for r in y]
    return outs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("protocol"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry", action="store_true"); a = ap.parse_args()
    P = json.loads(Path(a.protocol).read_text())
    out = Path(P["output_dir"] + ("-dryrun" if a.dry else "")); out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(P["base_path"])
    valid = np.load(P["valid_blocks"], mmap_mode="r"); held = np.load(P["heldout_blocks"], mmap_mode="r")
    bench = [json.loads(l) for l in open(P["kbench_path"])]
    lex = {l.split("\t")[0] for l in open(P["lexicon_path"])}
    nb, nbench, every, max_tr, max_ev = P["ppl_blocks"], len(bench), P["everyday_prompts"], P["max_new_translate"], P["max_new_everyday"]
    if a.dry:
        nb, nbench, every, max_tr, max_ev = 2, 4, every[:2], 8, 8
    bench = bench[:nbench]
    card = {"protocol": P["name"], "created_utc": time.strftime("%FT%TZ", time.gmtime()), "models": {}}
    for spec in P["models"]:
        t0 = time.time(); model = load(P, spec, a.device)
        r = {"ppl_valid": round(ppl(model, valid, min(nb, len(valid)), 4, a.device), 2),
             "ppl_heldout_channels": round(ppl(model, held, min(nb, len(held)), 4, a.device), 2)}
        kn2en = generate(model, tok, [P["kn2en_template"].format(text=b["kannada"]) for b in bench], max_tr, P["batch"], a.device)
        en2kn = generate(model, tok, [P["en2kn_template"].format(text=b["english"]) for b in bench], max_tr, P["batch"], a.device)
        r["kbench_kn2en_chrf"] = round(chrf(kn2en, [b["english"] for b in bench]), 2)
        r["kbench_en2kn_chrf"] = round(chrf(en2kn, [b["kannada"] for b in bench]), 2)
        ev = generate(model, tok, [P["everyday_template"].format(task=t) for t in every], max_ev, P["batch"], a.device)
        r["everyday_stays_konkani"] = round(sum(kn_letter_share(o) >= 0.5 for o in ev) / len(ev), 3)
        words = [w for o in ev for w in KN.findall(o)]
        r["everyday_mangalorean_word_share"] = round(sum(w in lex for w in words) / len(words), 3) if words else 0.0
        en2kn_words = [w for o in en2kn for w in KN.findall(o)]
        r["kbench_en2kn_mangalorean_word_share"] = round(sum(w in lex for w in en2kn_words) / len(en2kn_words), 3) if en2kn_words else 0.0
        r["seconds"] = round(time.time() - t0)
        card["models"][spec["name"]] = r
        (out / f"outputs_{spec['name']}.json").write_text(json.dumps(
            {"everyday": [{"task": t, "output": o} for t, o in zip(every, ev)],
             "kbench": [{"kannada": b["kannada"], "english": b["english"], "kn2en": x, "en2kn": y}
                        for b, x, y in zip(bench, kn2en, en2kn)]}, ensure_ascii=False, indent=1) + "\n")
        print(json.dumps({"model": spec["name"], **r}), flush=True)
        (out / "scorecard.json").write_text(json.dumps(card, indent=1) + "\n")
        del model
        if a.device == "cuda":
            torch.cuda.empty_cache()
    assert len(json.loads((out / "scorecard.json").read_text())["models"]) == len(P["models"])
    print(json.dumps({"event": "done", "models": len(P["models"])}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
