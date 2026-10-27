"""Resumable LoRA continued pretraining for one time-limited job slot (frozen protocol JSON).

Each run: load base + merged Konkani adapter, resume our LoRA + optimizer + scheduler from output_dir/checkpoint if
present, train on a fixed seeded block order until total_steps or until the slot budget (minus a reserve for eval +
save) would be exceeded, save the checkpoint atomically (previous kept as checkpoint.prev), evaluate validation loss,
append to metrics.jsonl, and save output_dir/final when training is complete. Exits 0 only after re-validating the
saved checkpoint. Usage: train_cpt.py PROTOCOL.json [--device cpu] [--dry]"""
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


def batch(arr, device):
    x = torch.from_numpy(arr.astype(np.int64)).to(device)
    return {"input_ids": x, "attention_mask": torch.ones_like(x), "token_type_ids": torch.zeros_like(x), "labels": x}


@torch.no_grad()
def eval_loss(model, valid, n, mb, device):
    model.eval(); losses = []
    for i in range(0, n, mb):
        losses.append(model(**batch(valid[i:min(i + mb, n)], device)).loss.item())
    model.train()
    return sum(losses) / len(losses)


def verify_tokens(P):
    """The frozen token manifest pins the block files; refuse to train on anything else."""
    m = json.loads(Path(P["token_manifest"]).read_text())
    for split, key in (("train", "train_blocks"), ("valid", "valid_blocks")):
        assert m["splits"][split]["path"] == P[key], f"{split} path differs from token manifest"
        h = hashlib.sha256()
        with open(P[key], "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        assert h.hexdigest() == m["splits"][split]["sha256"], f"{split} blocks do not match the token manifest"


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
    per_run = None
    if a.dry:
        mb, accum, total, per_run, n_eval, reserve = 1, 1, 4, 2, 2, 0
    ck = out / "checkpoint"
    start_step = json.loads((ck / "state.json").read_text())["step"] if ck.exists() else 0
    if start_step >= total and (out / "final").exists():
        log(out, {"event": "already_complete", "step": start_step}); return 0

    verify_tokens(P)
    train = np.load(P["train_blocks"], mmap_mode="r"); valid = np.load(P["valid_blocks"], mmap_mode="r")
    need = total * mb * accum
    assert len(train) >= need, f"need {need} blocks, have {len(train)}"
    order = np.random.default_rng(P["seed"]).permutation(len(train))[:need]

    dtype = torch.bfloat16
    model = AutoModelForImageTextToText.from_pretrained(P["base_path"], dtype=dtype, device_map=a.device)
    model = PeftModel.from_pretrained(model, P["konkani_adapter_path"]).merge_and_unload()
    if ck.exists():
        model = PeftModel.from_pretrained(model, ck / "adapter", is_trainable=True)
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
    log(out, {"event": "start", "step": start_step, "total": total, "resumed": ck.exists(), "load_s": round(time.time() - T0),
              "trainable_m": round(sum(p.numel() for _, p in named) / 1e6, 1)})

    model.train(); step, run, run_n, t_train, step_s = start_step, 0.0, 0, time.time(), None
    while step < total and (per_run is None or step - start_step < per_run):
        if step_s and time.time() - T0 + step_s + reserve > budget:
            break
        t_step = time.time()
        for k in range(accum):
            j = (step * accum + k) * mb
            loss = model(**batch(train[order[j:j + mb]], a.device)).loss / accum
            loss.backward(); run += loss.item()
        torch.nn.utils.clip_grad_norm_([p for _, p in named], P["max_grad_norm"])
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1; run_n += 1
        dt = time.time() - t_step; step_s = dt if step_s is None else 0.8 * step_s + 0.2 * dt
        if step % P["log_every"] == 0 or step == start_step + 1:
            done = step - start_step
            log(out, {"event": "train", "step": step, "loss": round(run / run_n, 4),
                      "lr": sched.get_last_lr()[0], "tok_s": round(done * mb * accum * P["seq"] / (time.time() - t_train)),
                      "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1) if a.device == "cuda" else None})
            run, run_n = 0.0, 0

    save_checkpoint(model, opt, sched, step, names, out)
    vl = eval_loss(model, valid, min(n_eval, len(valid)), mb, a.device)
    complete = step >= total
    if complete:
        shutil.rmtree(out / "final", ignore_errors=True); model.save_pretrained(out / "final")
    saved = json.loads((out / "checkpoint/state.json").read_text())
    assert saved["step"] == step and (out / "checkpoint/adapter/adapter_model.safetensors").stat().st_size > 0
    log(out, {"event": "slot_end", "step": step, "total": total, "complete": complete, "valid_loss": round(vl, 4),
              "valid_ppl": round(math.exp(vl), 2), "eval_blocks": min(n_eval, len(valid)), "elapsed_s": round(time.time() - T0)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
