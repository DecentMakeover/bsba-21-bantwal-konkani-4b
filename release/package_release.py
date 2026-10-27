"""Release packaging (one time-limited job slot): merge base + Konkani LLM adapter + stage A + stage B v2 into one model,
verify the merge (perplexity before/after merge and after reload must match), save it, convert to GGUF (f16) with
llama.cpp, quantize (Q8_0, Q4_K_M), smoke-test the Q4 file on CPU, and write release/manifest.json.
Usage: package_release.py PROTOCOL.json"""
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

T0 = time.time()


def log(out, rec):
    rec = {"utc": time.strftime("%FT%TZ", time.gmtime()), "s": round(time.time() - T0), **rec}
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    with (out / "package_log.jsonl").open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


@torch.no_grad()
def ppl(model, blocks, device):
    model.eval(); tot = 0.0
    for i in range(0, len(blocks), 4):
        x = torch.from_numpy(np.asarray(blocks[i:i + 4]).astype(np.int64)).to(device)
        tot += model(input_ids=x, attention_mask=torch.ones_like(x), token_type_ids=torch.zeros_like(x), labels=x).loss.item() * x.shape[0]
    return math.exp(tot / len(blocks))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def kn_share(t):
    kn = sum(0x0C80 <= ord(c) <= 0x0CFF for c in t); lat = sum(c.isascii() and c.isalpha() for c in t)
    return kn / (kn + lat) if kn + lat else 0.0


def main():
    P = json.loads(Path(sys.argv[1]).read_text())
    out = Path(P["output_dir"]); merged = out / "merged"; gg = out / "gguf"
    out.mkdir(parents=True, exist_ok=True); gg.mkdir(exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    blocks = np.load(P["check_blocks"], mmap_mode="r")[: P["check_n"]]
    tok = AutoTokenizer.from_pretrained(P["base_path"])

    # 1. merge + verify
    m = AutoModelForImageTextToText.from_pretrained(P["base_path"], dtype=torch.bfloat16, device_map=dev)
    for path in [P["konkani_adapter_path"]] + P["merge_adapter_paths"]:
        m = PeftModel.from_pretrained(m, path).merge_and_unload()
    m = PeftModel.from_pretrained(m, P["final_adapter_path"])
    p_adapter = ppl(m, blocks, dev)
    m = m.merge_and_unload()
    p_merged = ppl(m, blocks, dev)
    log(out, {"event": "merged", "ppl_with_adapter": round(p_adapter, 3), "ppl_merged": round(p_merged, 3)})
    assert abs(p_merged - p_adapter) / p_adapter < 0.01, "merge changed the model's perplexity by >1%"
    shutil.rmtree(merged, ignore_errors=True)
    m.save_pretrained(merged, safe_serialization=True, max_shard_size="5GB")
    tok.save_pretrained(merged)
    AutoProcessor.from_pretrained(P["base_path"]).save_pretrained(merged)
    del m; torch.cuda.empty_cache() if dev == "cuda" else None
    m = AutoModelForImageTextToText.from_pretrained(merged, dtype=torch.bfloat16, device_map=dev)
    p_reload = ppl(m, blocks, dev)
    log(out, {"event": "reloaded", "ppl_reloaded": round(p_reload, 3)})
    assert abs(p_reload - p_merged) / p_merged < 0.01, "saved model does not reproduce the merged model"
    del m; torch.cuda.empty_cache() if dev == "cuda" else None

    # 2. GGUF convert + quantize
    lc = Path(P["llama_cpp_dir"]); name = P["release_name"]
    f16 = gg / f"{name}-f16.gguf"
    env = {**os.environ, "PYTHONPATH": str(lc / "gguf-py")}
    r = subprocess.run([sys.executable, str(lc / "convert_hf_to_gguf.py"), str(merged), "--outtype", "f16", "--outfile", str(f16)],
                       capture_output=True, text=True, env=env)
    log(out, {"event": "convert", "rc": r.returncode, "tail": (r.stderr or r.stdout)[-400:]})
    assert r.returncode == 0 and f16.exists(), "GGUF conversion failed"
    for q in P["quant_types"]:
        dst = gg / f"{name}-{q}.gguf"
        r = subprocess.run([str(lc / "build/bin/llama-quantize"), str(f16), str(dst), q], capture_output=True, text=True)
        log(out, {"event": "quantize", "type": q, "rc": r.returncode, "gb": round(dst.stat().st_size / 1e9, 2) if dst.exists() else None})
        assert r.returncode == 0 and dst.exists(), f"quantize {q} failed"

    # 3. smoke test of the smallest file on CPU
    q4 = gg / f"{name}-{P['quant_types'][-1]}.gguf"; tests = []
    for prompt in P["test_prompts"]:
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        text = text.removeprefix(tok.bos_token or "")
        r = subprocess.run([str(lc / "build/bin/llama-completion"), "-m", str(q4), "-p", text, "-n", "160", "--temp", "0",
                            "-no-cnv", "--no-display-prompt", "-t", "8"], capture_output=True, text=True, timeout=600)
        ans = r.stdout.strip()
        tests.append({"prompt": prompt, "answer": ans, "kn_share": round(kn_share(ans), 3), "rc": r.returncode})
        log(out, {"event": "gguf_test", "rc": r.returncode, "kn_share": round(kn_share(ans), 3), "answer": ans[:200]})
    (out / "gguf_smoke_test.json").write_text(json.dumps(tests, ensure_ascii=False, indent=1) + "\n")
    assert sum(t["rc"] == 0 and t["kn_share"] >= 0.5 for t in tests) >= 2, "GGUF smoke test: answers not mostly Kannada script"

    # 4. manifest
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.suffix in (".safetensors", ".gguf", ".json", ".model", ".jinja"))
    man = {"release_name": name, "created_utc": time.strftime("%FT%TZ", time.gmtime()), "source": P["description"],
           "check": {"ppl_with_adapter": p_adapter, "ppl_merged": p_merged, "ppl_reloaded": p_reload, "blocks": P["check_n"]},
           "files": [{"path": str(p.relative_to(out)), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in files
                     if p.name not in ("manifest.json",)]}
    (out / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    log(out, {"event": "done", "files": len(man["files"])})
    return 0


if __name__ == "__main__":
    sys.exit(main())
