"""Generation check: the same prompts through baseline (Gemma-3-4B-it + konkani adapter, merged), stage A (cpt LoRA)
and stage B (sft LoRA), greedy decoding. The base is loaded once; LoRAs are switched with set_adapter /
disable_adapter. Prompts and settings come from a frozen protocol JSON. Writes notes/generation_check_<name>.json.
Usage: generate_check.py PROTOCOL.json [--device cpu] [--dry]"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer


@torch.no_grad()
def generate(model, tok, prompt, kind, max_new, device):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True) \
        if kind == "chat" else prompt
    x = tok(text, return_tensors="pt", add_special_tokens=(kind != "chat")).to(device)
    t = time.time()
    out = model.generate(**x, max_new_tokens=max_new, do_sample=False, repetition_penalty=1.1)
    new = out[0][x["input_ids"].shape[1]:]
    return tok.decode(new, skip_special_tokens=True).strip(), round(time.time() - t, 1), int(new.shape[0])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("protocol"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry", action="store_true"); a = ap.parse_args()
    P = json.loads(Path(a.protocol).read_text())
    prompts = P["prompts"][:2] if a.dry else P["prompts"]
    tok = AutoTokenizer.from_pretrained(P["base_path"])
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(P["base_path"], dtype=torch.bfloat16, device_map=a.device)
    model = PeftModel.from_pretrained(model, P["konkani_adapter_path"]).merge_and_unload()
    model = PeftModel.from_pretrained(model, P["adapters"]["stage_a"], adapter_name="stage_a")
    model.load_adapter(P["adapters"]["stage_b"], adapter_name="stage_b")
    model.eval()
    print(json.dumps({"event": "loaded", "s": round(time.time() - t0)}), flush=True)
    results = []
    for p in prompts:
        max_new = 4 if a.dry else P["max_new_tokens"][p["kind"]]
        row = {**p, "outputs": {}}
        with model.disable_adapter():
            row["outputs"]["baseline"] = generate(model, tok, p["prompt"], p["kind"], max_new, a.device)
        for name in ("stage_a", "stage_b"):
            model.set_adapter(name)
            row["outputs"][name] = generate(model, tok, p["prompt"], p["kind"], max_new, a.device)
        results.append(row)
        print(json.dumps({"event": "prompt_done", "id": p["id"], "secs": {k: v[1] for k, v in row["outputs"].items()}}), flush=True)
    out = Path(P["output"].replace(".json", "-dryrun.json") if a.dry else P["output"])
    out.write_text(json.dumps({"protocol": P["name"], "created_utc": time.strftime("%FT%TZ", time.gmtime()),
                               "models": {"baseline": "Gemma-3-4B-it + konkani/konkani-gemma-3-4b-it",
                                          "stage_a": P["adapters"]["stage_a"], "stage_b": P["adapters"]["stage_b"]},
                               "results": results}, ensure_ascii=False, indent=1) + "\n")
    assert len(json.loads(out.read_text())["results"]) == len(prompts)
    print(json.dumps({"event": "done", "prompts": len(prompts), "output": str(out), "elapsed_s": round(time.time() - t0)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
