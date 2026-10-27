"""Resumable LoRA training for one time-limited job slot, with optional per-token labels (instruction tuning) and an extra eval.

v2 of train_lora.py (v1 kept unchanged): adds `merge_adapter_paths` (merged into the base before the trainable LoRA, e.g. stage A). Otherwise like train_cpt.py, plus: `train_labels`/`valid_labels` (-100 = no loss, e.g. prompt tokens),
`init_adapter_path` (start from an earlier LoRA instead of a fresh one), `frozen_data` ({path: sha256}, all verified at
start) and `extra_eval` ({name, blocks, n}: unlabeled blocks, e.g. the Mangalorean corpus, to watch for forgetting).
Saves checkpoint atomically, evaluates, writes final/ when total_steps is reached, exits 0 only after re-validating.
Usage: train_lora.py PROTOCOL.json [--device cpu] [--dry]"""
import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForImageTextToText, get_cosine_schedule_with_warmup

T0 = time.time()


def log(out, rec):
    rec = {"utc": time.strftime("%FT%TZ", time.gmtime()), **rec}
    print(json.dumps(rec), flush=True)
    with (out / "metrics.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")


def verify_data(P):
    for path, digest in P["frozen_data"].items():
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        assert h.hexdigest() == digest, f"{path} does not match its frozen SHA256"


def batch(ids, labels, device):
    x = torch.from_numpy(np.asarray(ids).astype(np.int64)).to(device)
    y = x if labels is None else torch.from_numpy(np.asarray(labels).astype(np.int64)).to(device)
    return {"input_ids": x, "attention_mask": torch.ones_like(x), "token_type_ids": torch.zeros_like(x), "labels": y}


@torch.no_grad()
def eval_loss(model, ids, labels, n, mb, device):
    model.eval(); total, count = 0.0, 0
    for i in range(0, n, mb):
        j = min(i + mb, n)
        loss = model(**batch(ids[i:j], None if labels is None else labels[i:j], device)).loss.item()
        total += loss * (j - i); count += j - i
    model.train()
    return total / count


def trainable(model):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def save_checkpoint(model, opt, sched, step, names, out):
    tmp, ck, prev = out / "checkpoint.tmp", out / "checkpoint", out / "checkpoint.prev"
    shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    model.save_pretrained(tmp / "adapter")
    torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "torch_rng": torch.get_rng_state()},
               tmp / "train_state.pt")
    (tmp / "state.json").write_text(json.dumps({"step": step, "param_names": names}))
    for f in tmp.rglob("*"):
        if f.is_file():
            with open(f, "rb") as fh:
                os.fsync(fh.fileno())
    if ck.exists():
        shutil.rmtree(prev, ignore_errors=True); ck.rename(prev)
    tmp.rename(ck)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("protocol"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry", action="store_true", help="tiny CPU check: micro 1, accum 1, 4 steps, 2 per run")
    a = ap.parse_args()
    P = json.loads(Path(a.protocol).read_text())
    out = Path(P["output_dir"] + ("-dryrun" if a.dry else "")); out.mkdir(parents=True, exist_ok=True)
    mb, accum, total = P["micro_batch"], P["grad_accum"], P["total_steps"]
    budget, reserve, n_eval = P["slot_budget_s"], P["reserve_s"], P["eval_blocks"]
    extra = P.get("extra_eval"); n_extra = extra["n"] if extra else 0
    per_run = None
    if a.dry:
        mb, accum, total, per_run, n_eval, n_extra, reserve = 1, 1, 4, 2, 2, 2, 0
    ck = out / "checkpoint"
    start_step = json.loads((ck / "state.json").read_text())["step"] if ck.exists() else 0
    if start_step >= total and (out / "final").exists():
        log(out, {"event": "already_complete", "step": start_step}); return 0

    verify_data(P)
    train = np.load(P["train_ids"], mmap_mode="r"); valid = np.load(P["valid_ids"], mmap_mode="r")
    train_y = np.load(P["train_labels"], mmap_mode="r") if P.get("train_labels") else None
    valid_y = np.load(P["valid_labels"], mmap_mode="r") if P.get("valid_labels") else None
    need = total * mb * accum
    assert len(train) >= need, f"need {need} blocks, have {len(train)}"
    order = \
        np.random.default_rng(P["seed"]).permutation(len(train))[:need]

    model = AutoModelForImageTextToText.from_pretrained(P["base_path"], dtype=torch.bfloat16, device_map=a.device)
    for path in [P["konkani_adapter_path"]] + P.get("merge_adapter_paths", []):
        model = PeftModel.from_pretrained(model, path).merge_and_unload()  # stage A merged = frozen base for v2
    if ck.exists():
        model = PeftModel.from_pretrained(model, ck / "adapter", is_trainable=True)
    elif P.get("init_adapter_path"):
        model = PeftModel.from_pretrained(model, P["init_adapter_path"], is_trainable=True)
    else:
        L = P["lora"]
        model = get_peft_model(model, LoraConfig(r=L["r"], lora_alpha=L["alpha"], lora_dropout=L["dropout"],
                                                 target_modules=L["target_modules"], task_type="CAUSAL_LM"))
    named = trainable(model); names = [n for n, _ in named]
    opt = torch.optim.AdamW([p for _, p in named], lr=P["lr"], weight_decay=P["weight_decay"])
    sched = get_cosine_schedule_with_warmup(opt, P["warmup_steps"], total)
    if ck.exists():
        saved = json.loads((ck / "state.json").read_text())
        assert saved["param_names"] == names, "trainable parameter order changed; refusing to resume"
        st = torch.load(ck / "train_state.pt", weights_only=False)
        opt.load_state_dict(st["optimizer"]); sched.load_state_dict(st["scheduler"]); torch.set_rng_state(st["torch_rng"])
    log(out, {"event": "start", "step": start_step, "total": total, "resumed": ck.exists(),
              "init": "checkpoint" if ck.exists() else ("init_adapter" if P.get("init_adapter_path") else "fresh"),
              "load_s": round(time.time() - T0), "trainable_m": round(sum(p.numel() for _, p in named) / 1e6, 1)})
    if start_step == 0 and not a.dry:
        rec = {"event": "eval_before", "step": 0,
               "valid_loss": round(eval_loss(model, valid, valid_y, min(n_eval, len(valid)), mb, a.device), 4)}
        if extra:
            xb = np.load(extra["blocks"], mmap_mode="r")
            rec[f"{extra['name']}_ppl"] = round(math.exp(eval_loss(model, xb, None, min(n_extra, len(xb)), mb, a.device)), 2)
        log(out, rec)

    model.train(); step, run, run_n, t_train, step_s = start_step, 0.0, 0, time.time(), None
    while step < total and (per_run is None or step - start_step < per_run):
        if step_s and time.time() - T0 + step_s + reserve > budget:
            break
        t_step = time.time()
        for k in range(accum):
            j = (step * accum + k) * mb
            idx = order[j:j + mb]
            loss = model(**batch(train[idx], None if train_y is None else train_y[idx], a.device)).loss / accum
            loss.backward(); run += loss.item()
        torch.nn.utils.clip_grad_norm_([p for _, p in named], P["max_grad_norm"])
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1; run_n += 1
        dt = time.time() - t_step; step_s = dt if step_s is None else 0.8 * step_s + 0.2 * dt
        if step % P["log_every"] == 0 or step == start_step + 1:
            done = step - start_step
            log(out, {"event": "train", "step": step, "loss": round(run / run_n, 4), "lr": sched.get_last_lr()[0],
                      "tok_s": round(done * mb * accum * P["seq"] / (time.time() - t_train)),
                      "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1) if a.device == "cuda" else None})
            run, run_n = 0.0, 0

    save_checkpoint(model, opt, sched, step, names, out)
    rec = {"event": "slot_end", "step": step, "total": total, "complete": step >= total,
           "valid_loss": round(eval_loss(model, valid, valid_y, min(n_eval, len(valid)), mb, a.device), 4)}
    if extra:
        xb = np.load(extra["blocks"], mmap_mode="r")
        rec[f"{extra['name']}_ppl"] = round(math.exp(eval_loss(model, xb, None, min(n_extra, len(xb)), mb, a.device)), 2)
    if step >= total:
        shutil.rmtree(out / "final", ignore_errors=True); model.save_pretrained(out / "final")
    saved = json.loads((out / "checkpoint/state.json").read_text())
    assert saved["step"] == step and (out / "checkpoint/adapter/adapter_model.safetensors").stat().st_size > 0
    rec["elapsed_s"] = round(time.time() - T0)
    log(out, rec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
