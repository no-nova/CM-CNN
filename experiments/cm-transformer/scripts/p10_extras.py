"""Phase 10 extras: prompt-level generation statistics, calibration tail metrics,
and the cross-architecture energy/error correlation table.

Tasks
-----
genprompt : 64 prompts (short <=32 tokens, long >=128) x 4 categories.  Greedy
            agreement, first-divergence step, and temperature-0.7 sampling with
            three seeds, reporting per-step KL(dense || sparse) and top-1
            probability agreement.  Prompt is the sampling unit for the bootstrap.
calibtail : runtime active-channel overlap across calibration corpora and seeds,
            and the calibration-size sweep with paired bootstrap intervals.
corr_all  : Spearman/Pearson between ||a_b||_2 and ||delta y_b||_2 on whisper-base
            (GELU FFN) and SmolVLM2 (ViT + SwiGLU LM) so the table covers five
            architectures.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))

import p1_v2 as P1  # noqa: E402
import p6_sota as P6  # noqa: E402
import p8_reviewer_defense as P8  # noqa: E402
import p9_round2_closure as P9  # noqa: E402

RES = EXP / "results"
LANE = 32


def log(*a):
    print(*a, flush=True)


def boot(v, n=1000, seed=1234):
    v = np.asarray(v, float)
    if v.size == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, v.size, (n, v.size))].mean(1)
    return {"mean": round(float(v.mean()), 4), "lo": round(float(np.percentile(m, 2.5)), 4),
            "hi": round(float(np.percentile(m, 97.5)), 4), "median": round(float(np.median(v)), 4),
            "n": int(v.size)}


# --------------------------------------------------------------------------- #
def _fwd(runner, toks, pos, state, cfg):
    if hasattr(runner, "tb"):
        kw = {"mlp_override": cfg.get("override"), "skip_scale": cfg.get("scale"),
              "skip_theta": cfg.get("theta"), "skip_rate": cfg.get("rate", 0.0)}
        return runner.tb.forward(toks, state=state, pos0=pos, **kw)["logits"]
    return runner.logits(toks)


def trace(runner, ids, n_new, cfg, temp=0.0, seed=0):
    """Greedy/sampled trace plus the per-step distributions it drew from."""
    from model_ref import new_state
    st = new_state()
    lg = _fwd(runner, ids, 0, st, cfg)
    pos = ids.numel()
    cur = lg[-1:].argmax(-1) if temp == 0 else torch.multinomial(
        torch.softmax(lg[-1:].float() / temp, -1), 1, generator=None)
    toks, kl, p1 = [], [], []
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(n_new):
        toks.append(int(cur))
        nxt = _fwd(runner, cur, pos, st, cfg)[-1:]
        pos += 1
        if temp == 0:
            cur = nxt.argmax(-1)
        else:
            pr = torch.softmax(nxt.float() / temp, -1)
            cur = torch.multinomial(pr.cpu(), 1, generator=gen).to(nxt.device)
    return toks


def step_stats(runner, ids, n_new, cfg_d, cfg_s, temp):
    """Same trajectory for both models; per-step KL and top-1 prob of the drawn token."""
    from model_ref import new_state
    sd, ss = new_state(), new_state()
    lg_d = _fwd(runner, ids, 0, sd, cfg_d)
    lg_s = _fwd(runner, ids, 0, ss, cfg_s)
    pos = ids.numel()
    cur = lg_d[-1:].argmax(-1)
    kls, tops, agree = [], [], []
    for _ in range(n_new):
        pd = torch.softmax(lg_d[-1:].float(), -1)
        ps = torch.softmax(lg_s[-1:].float(), -1)
        kls.append(float(torch.sum(pd * (pd.clamp_min(1e-9).log() - ps.clamp_min(1e-9).log()))))
        tops.append(float(pd[0, int(cur)]))
        agree.append(int(pd.argmax(-1)[0] == ps.argmax(-1)[0]))
        n_d = _fwd(runner, cur, pos, sd, cfg_d)[-1:]
        n_s = _fwd(runner, cur, pos, ss, cfg_s)[-1:]
        pos += 1
        lg_d, lg_s = n_d, n_s
        cur = n_d.argmax(-1)
    return kls, tops, agree


def task_genprompt(device, model="qwen3.5-0.8b", n_prompts=64, n_new=32, seeds=(0, 1, 2)):
    runner = P6.make_runner(model, device, torch.float32)
    acts = P9.capture(runner, P9.calib_texts())
    perm = P9.perms_for(runner, acts, kinds=("ours",))["ours"]
    cfg = P9.make_cfg(runner, acts, perm, 0.20)
    prompts = P9.prompt_set()[:n_prompts]
    log(f"[gen] {model} prompts={len(prompts)} gen={n_new} seeds={list(seeds)}")

    agr, fds, fds_list, kl_steps, top1s = [], [], [], [], []
    for k, p in enumerate(prompts):
        ids = P9.encode(runner, [p["prompt"]], 256)[0].to(device)
        a = trace(runner, ids, n_new, {})
        b = trace(runner, ids, n_new, cfg)
        same = [int(x == y) for x, y in zip(a, b)]
        agr.append(sum(same) / len(same))
        fd = next((i for i, s in enumerate(same) if s == 0), n_new)
        fds.append(fd)
        kls, tops, ag = step_stats(runner, ids, n_new, {}, cfg, 0.0)
        kl_steps.append(float(np.mean(kls)))
        top1s.append(float(np.mean(ag)))
        fds_list.append(fd)
        if (k + 1) % 8 == 0:
            log(f"[gen] {model} greedy {k + 1}/{len(prompts)}")
    out = {"model": model, "n_prompts": len(prompts), "gen_len": n_new, "tau": 0.20,
           "greedy": {"token_agreement": boot(agr), "first_divergence": boot(fds),
                      "first_divergence_median": float(np.median(fds)),
                      "fds_hist": {str(i): int(fds.count(i)) for i in sorted(set(fds))},
                      "step_kl": boot(kl_steps), "step_top1_agree": boot(top1s)},
           "sampling": {}}
    for seed in seeds:
        kl_s, top_s = [], []
        for p in prompts[:16]:
            ids = P9.encode(runner, [p["prompt"]], 256)[0].to(device)
            kls, tops, _ = step_stats_sampled(runner, ids, n_new, cfg, temp=0.7, seed=seed)
            kl_s.append(float(np.mean(kls)))
            top_s.append(float(np.mean(tops)))
        out["sampling"][str(seed)] = {"kl": boot(kl_s), "top1_prob": boot(top_s)}
        log(f"[gen] {model} sampling seed={seed} kl {out['sampling'][str(seed)]['kl']['mean']:.4f}")
    (RES / "p10_genprompt.json").write_text(json.dumps({model: out}, indent=1))
    log("wrote results/p10_genprompt.json")
    return out


@torch.no_grad()
def step_stats_sampled(runner, ids, n_new, cfg, temp, seed):
    """One trajectory drawn from the dense model at temperature T, scored under both."""
    from model_ref import new_state
    sd, ss = new_state(), new_state()
    lg_d = _fwd(runner, ids, 0, sd, {})
    lg_s = _fwd(runner, ids, 0, ss, cfg)
    pos = ids.numel()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    cur = torch.multinomial(torch.softmax(lg_d[-1:].float() / temp, -1).cpu(), 1,
                            generator=gen).to(lg_d.device)
    kls, tops = [], []
    for _ in range(n_new):
        pd = torch.softmax(lg_d[-1:].float(), -1)
        ps = torch.softmax(lg_s[-1:].float(), -1)
        kls.append(float(torch.sum(pd * (pd.clamp_min(1e-9).log() - ps.clamp_min(1e-9).log()))))
        tok = int(torch.multinomial(torch.softmax(lg_d[-1:].float() / temp, -1).cpu(), 1,
                                    generator=gen))
        tops.append(float(pd[0, tok]))
        n_d = _fwd(runner, cur, pos, sd, {})[-1:]
        n_s = _fwd(runner, cur, pos, ss, cfg)[-1:]
        pos += 1
        lg_d, lg_s = n_d, n_s
        cur = tok_ten = torch.tensor([tok], device=lg_d.device)
    return kls, tops, None


# --------------------------------------------------------------------------- #
def task_calibtail(device, model="qwen3.5-0.8b"):
    """Runtime overlap of kept blocks across calibration corpora, and the size sweep."""
    from model_ref import block_energy
    runner = P6.make_runner(model, device, torch.float32)
    ids = P9.encode(runner, [str(P8.corpus_text("wiki") * 8)],
                    P9.WINDOWS * P9.SEQ_LEN + 1)[0].to(device)
    out = {"model": model, "active_overlap": {}, "size_sweep": {}, "matrix": {}}
    eval_text = str(P8.corpus_text("wiki")) * 2
    eval_acts = {i: v for i, v in P9.capture(runner, [eval_text]).items()}
    base = P9.ppl_of(P9.win_nll(runner, ids))
    domains = ("wiki", "code", "zh", "math")
    keep_sets, res_ppl = {}, {}
    for cal in domains:
        cal_ids = P9.encode(runner, [str(P8.corpus_text(cal) * 4)], 1024)[0]
        acts = P9.capture(runner, [str(P8.corpus_text(cal) * 4)])
        perm = P9.perms_for(runner, acts, kinds=("ours",))["ours"]
        cfg = P9.make_cfg(runner, acts, perm, 0.20)
        keep_sets[cal] = {}
        for l in range(runner.L):
            a = eval_acts[l].to(device).index_select(1, perm[l].to(device))
            a = a * cfg["scale"][l].to(device)
            e = block_energy(a)
            ebar = e.mean(0)          # canonical active set: blocks active on average
            keep_sets[cal][l] = set(torch.nonzero(ebar >= cfg["theta"][l])
                                    .flatten().tolist())
        res_ppl[cal] = P9.ppl_of(P9.win_nll(runner, ids, **cfg))
        log(f"[calibtail] {model} cal={cal} ppl {res_ppl[cal]:.4f}")
    for a in domains:
        for b in domains:
            if a >= b:
                continue
            js = [len(keep_sets[a][l] & keep_sets[b][l]) /
                  max(len(keep_sets[a][l] | keep_sets[b][l]), 1) for l in range(runner.L)]
            out["active_overlap"][f"{a}_vs_{b}"] = round(float(np.mean(js)), 4)
    dp = {k: v - base for k, v in res_ppl.items()}
    out["matrix"] = {"dense_ppl": round(base, 4),
                     "delta_ppl": {k: round(v, 4) for k, v in dp.items()},
                     "mean": round(float(np.mean(list(dp.values()))), 4),
                     "std": round(float(np.std(list(dp.values()))), 4)}
    log(f"[calibtail] {model} overlap {out['active_overlap']}")
    log(f"[calibtail] {model} dPPL mean {out['matrix']['mean']:+.4f} sd {out['matrix']['std']:.4f}")

    base_nll = np.array(P9.win_nll(runner, ids)) / P9.SEQ_LEN
    for n_tok in (128, 512, 2048, 8192):
        chunk = str(P8.corpus_text("wiki"))[: n_tok * 4]
        acts = P9.capture(runner, [chunk])
        perm = P9.perms_for(runner, acts, kinds=("ours",))["ours"]
        cfg = P9.make_cfg(runner, acts, perm, 0.20)
        nll = P9.win_nll(runner, ids, **cfg)
        per = np.array(nll) / P9.SEQ_LEN
        bper = np.asarray(base_nll, dtype=float)
        rng = np.random.default_rng(P9.SEED)
        idx = rng.integers(0, per.size, (2000, per.size))
        d = np.exp(per[idx].mean(1)) - np.exp(bper[idx].mean(1))
        ci = {"mean": round(float(d.mean()), 4),
              "lo": round(float(np.percentile(d, 2.5)), 4),
              "hi": round(float(np.percentile(d, 97.5)), 4),
              "n": int(per.size)}
        out["size_sweep"][str(n_tok)] = {"delta_ppl": round(P9.ppl_of(nll) - base, 4),
                                         "delta_ppl_ci": ci}
        log(f"[calibtail] {model} N={n_tok} dPPL {out['size_sweep'][str(n_tok)]['delta_ppl']:+.4f}")
    name = f"p10_calibtail{'_06b' if '0.6' in model else ''}.json"
    (RES / name).write_text(json.dumps(out, indent=1))
    log("wrote", name)


# --------------------------------------------------------------------------- #
def task_corr_all(device):
    """Whisper (GELU encoder FFN) and SmolVLM2 (ViT + LM) energy/error correlation."""
    out = {"lane": LANE, "models": {}}
    try:
        out["models"].update(_corr_whisper())
    except Exception as exc:                     # pragma: no cover - optional
        out["models"]["whisper-base"] = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        out["models"].update(_corr_smolvlm(device))
    except Exception as exc:                     # pragma: no cover - optional
        out["models"]["smolvlm2-256m"] = {"error": f"{type(exc).__name__}: {exc}"}
    (RES / "p10_corr_all.json").write_text(json.dumps(out, indent=1))
    log("wrote results/p10_corr_all.json")


def _block_stats(a, Wd, lane=LANE):
    a = a.float().detach()
    Wd = Wd.float().detach().to(a.device)
    assert Wd.shape[1] == a.shape[-1], (Wd.shape, a.shape)   # [d_model, d_ff]
    nb = a.shape[-1] // lane
    A = a.reshape(-1, nb, lane)
    Wb = Wd.t().reshape(nb, lane, -1)
    e = (A ** 2).sum(-1)
    d = torch.einsum("nbl,blo->nbo", A, Wb).pow(2).sum(-1).sqrt()
    wn = Wb.pow(2).sum((-1, -2)).sqrt()
    r = float(np.corrcoef(e.detach().reshape(-1).cpu().numpy(),
                          d.detach().reshape(-1).cpu().numpy())[0, 1])
    rx = e.detach().reshape(-1).cpu().numpy().argsort().argsort().astype(float)
    ry = d.detach().reshape(-1).cpu().numpy().argsort().argsort().astype(float)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    y = A.reshape(-1, a.shape[-1]) @ Wd.t()
    keep = e >= torch.quantile(e.reshape(-1), 0.2)
    yh = (A * keep[..., None]).reshape(-1, a.shape[-1]) @ Wd.t()
    rel = float((y - yh).norm() / y.norm().clamp_min(1e-9))
    return {"cv_w": round(float(wn.std() / wn.mean().clamp_min(1e-9)), 4),
            "cv_a": round(float(e.mean(0).std() / e.mean(0).mean().clamp_min(1e-9)), 4),
            "pearson": round(r, 4), "spearman": round(rho, 4),
            "rel_l2_tau020": round(rel, 4)}


def _corr_whisper():
    """GELU encoder FFN: load whisper directly and capture fc2 inputs per layer."""
    from transformers import WhisperForConditionalGeneration
    import bench_whisper_real as W
    import bench_whisper as WB
    root = Path(EXP).parents[1] / "data" / "p6_whisper_audio"
    wavs = sorted(root.rglob("*.flac"))[:6]
    if not wavs:
        return {}
    m = WhisperForConditionalGeneration.from_pretrained(str(WB.WHISPER_DIR),
                                                         dtype=torch.float32).eval()
    from transformers import WhisperFeatureExtractor
    feats_proc = WhisperFeatureExtractor.from_pretrained(str(WB.WHISPER_DIR))
    enc = m.model.encoder
    n = len(enc.layers)
    caps = {i: [] for i in range(n)}
    hooks = [blk.fc2.register_forward_pre_hook(
        lambda mod, args, i=i: caps[i].append(args[0].detach().reshape(-1, 2048)[-256:].float()))
        for i, blk in enumerate(enc.layers)]
    for w in wavs:
        audio = W.load_wav(w)
        f = feats_proc(audio, sampling_rate=16000, return_tensors="pt").input_features
        with torch.no_grad():
            enc(f)
    for h in hooks:
        h.remove()
    per = [_block_stats(torch.cat(caps[i], 0), enc.layers[i].fc2.weight) for i in range(n)]
    return {"whisper-base": {"arch": "GELU ff1/ff2, 512->2048->512", "per_layer": per,
                             "mean": {k: round(float(np.mean([p[k] for p in per])), 4)
                                      for k in per[0]}}}


def _corr_smolvlm(device):
    """SmolVLM2: ViT GELU FFN and language-tower SwiGLU, captured with hooks."""
    import numpy as np
    import bench_smolvlm as S
    model, proc, _ = S.load_model(device)
    cfg = model.config
    p = cfg.vision_config.patch_size
    img = S.synth_shapes(512)
    arr = np.array(img.convert("RGB"))
    batch = proc(text="<image>\nDescribe the image.", images=[arr], return_tensors="pt")
    pv = batch["pixel_values"]
    if pv.dim() == 5:
        pv = pv[:, 0]
    pam = batch["pixel_attention_mask"].bool()
    pm = (pam[:, 0, ::p, ::p] if pam.dim() == 4 else pam[:, ::p, ::p]).contiguous()
    vt = model.model.vision_model
    vcap = {}

    def hook(i):
        def f(mod, args):
            vcap.setdefault(i, []).append(args[0].detach().float().reshape(-1, 3072))
        return f

    hs = [blk.mlp.fc2.register_forward_pre_hook(hook(i)) for i, blk in enumerate(vt.encoder.layers)]
    with torch.no_grad():
        vt(pixel_values=pv.to(device), patch_attention_mask=pm.to(device))
    for h in hs:
        h.remove()
    vis = [_block_stats(torch.cat(vcap[i], 0).cpu(), vt.encoder.layers[i].mlp.fc2.weight.cpu())
           for i in range(len(vt.encoder.layers))]

    tm = model.model.text_model
    tcap = {}

    def thook(i):
        def f(mod, args):
            tcap.setdefault(i, []).append(args[0].detach().float().reshape(-1, 1536))
        return f

    hs = [blk.mlp.down_proj.register_forward_pre_hook(thook(i))
          for i, blk in enumerate(tm.layers)]
    text = proc(text="The capital of France is", return_tensors="pt")
    with torch.no_grad():
        model(**{k: v.to(device) for k, v in text.items()})
    for h in hs:
        h.remove()
    txt = [_block_stats(torch.cat(tcap[i], 0).cpu(), tm.layers[i].mlp.down_proj.weight.cpu())
           for i in range(len(tm.layers))]

    def agg(per):
        return {k: round(float(np.mean([q[k] for q in per])), 4) for k in per[0]}

    return {"smolvlm2-256m": {"arch": "ViT GELU fc1/fc2 768->3072; LM SwiGLU 576->1536",
                              "towers": {"vision": {"per_layer": vis, "mean": agg(vis)},
                                         "text": {"per_layer": txt, "mean": agg(txt)}},
                              "mean": agg(vis)}}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="genprompt")
    ap.add_argument("--model", default="qwen3.5-0.8b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prompts", type=int, default=64)
    ap.add_argument("--gen", type=int, default=32)
    a = ap.parse_args()
    if a.task == "genprompt":
        task_genprompt(a.device, a.model, a.prompts, a.gen)
    elif a.task == "calibtail":
        task_calibtail(a.device, a.model)
    elif a.task == "corr":
        task_corr_all(a.device)
