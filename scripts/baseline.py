import argparse
import json
import os
import sys
import time
from math import gcd
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "origin" / "FLowHigh_code"))

from cfm_superresolution import ConditionalFlowMatcherWrapper, FLowHigh, MelVoco
from postprocessing import PostProcessing

N_MELS = 256
SR = 48000


def build(args):
    m = MelVoco(
        n_mels=N_MELS,
        sampling_rate=SR,
        f_max=24000,
        n_fft=2048,
        win_length=2048,
        hop_length=480,
        vocoder="bigvgan",
        vocoder_config=str(ROOT / "origin/FLowHigh_code/vocoder/BIGVGAN/config/bigvgan_48khz_256band_config.json"),
        vocoder_path=str(ROOT / "origin/checkpoints/bigvgan/g_48_00850000"),
    )
    g = FLowHigh(dim_in=N_MELS, audio_enc_dec=m, depth=2, dim_head=64, heads=16, architecture="transformer")
    w = ConditionalFlowMatcherWrapper(
        flowhigh=g,
        cfm_method="independent_cfm_adaptive",
        torchdiffeq_ode_method=args.ode,
        sigma=1e-4,
    )
    ck = torch.load(ROOT / "origin/checkpoints/adaptive/FLowHigh_indep_adaptive_400k.pt", map_location="cpu", weights_only=False)
    w.load_state_dict(ck["model"])
    return m.to(args.device).eval(), w.to(args.device).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--ode", choices=["euler", "midpoint"], default="euler")
    p.add_argument("--std", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-post", action="store_true")
    args = p.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    x, sr = sf.read(args.input, always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        raise ValueError("empty input")
    if sr != SR:
        g = gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    peak = float(np.max(np.abs(x)))
    if peak > 0:
        x = x / peak

    torch.manual_seed(args.seed)
    m, w = build(args)
    cond_wave = torch.from_numpy(x).unsqueeze(0).to(args.device)
    t0 = time.time()
    with torch.inference_mode():
        y = w.sample(
            cond=cond_wave,
            time_steps=args.steps,
            cfm_method="independent_cfm_adaptive",
            std_1=1.0,
            std_2=args.std,
        ).squeeze(1)
        if not args.no_post:
            pp = PostProcessing(0)
            y = pp.post_processing(y, cond_wave, cond_wave.size(-1))
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    rtf = (time.time() - t0) / (x.size / SR)
    y = y.squeeze(0).clamp(-1, 1).float().cpu().numpy()
    sf.write(out, y, SR, subtype="PCM_16")
    np.save(out.with_suffix(".npy"), y.astype(np.float32))
    meta = {
        "input": str(args.input),
        "input_sr": int(sr),
        "output": str(out),
        "steps": args.steps,
        "ode": args.ode,
        "std": args.std,
        "seed": args.seed,
        "post": not args.no_post,
        "rtf": rtf,
        "audio_sec": x.size / SR,
        "out_samples": int(y.size),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
