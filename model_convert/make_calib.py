import argparse
import json
import os
import sys
import tarfile
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "origin" / "FLowHigh_code"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UPSTREAM))

N_MELS = 256
SR = 48000
FRAMES = 500
CLIP = FRAMES * 480


def load_audio(path):
    x, sr = sf.read(path, always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = np.asarray(x, dtype=np.float32)
    if sr != SR:
        g = gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def chunks(path, limit=8):
    x = load_audio(path)
    peak = float(np.max(np.abs(x))) if x.size else 0
    if peak > 0:
        x = x / peak
    out = []
    for start in range(0, max(1, len(x) - CLIP + 1), CLIP):
        c = x[start:start + CLIP]
        if len(c) < CLIP:
            c = np.pad(c, (0, CLIP - len(c)), mode="reflect")
        out.append(c.astype(np.float32))
        if len(out) >= limit:
            break
    if not out:
        c = np.pad(x, (0, max(0, CLIP - len(x))), mode="reflect")[:CLIP]
        out.append(c.astype(np.float32))
    return out


def build():
    from cfm_superresolution import ConditionalFlowMatcherWrapper, FLowHigh, MelVoco

    m = MelVoco(
        n_mels=N_MELS,
        sampling_rate=SR,
        f_max=24000,
        n_fft=2048,
        win_length=2048,
        hop_length=480,
        vocoder="bigvgan",
        vocoder_config=str(UPSTREAM / "vocoder/BIGVGAN/config/bigvgan_48khz_256band_config.json"),
        vocoder_path=str(ROOT / "origin/checkpoints/bigvgan/g_48_00850000"),
    )
    g = FLowHigh(dim_in=N_MELS, audio_enc_dec=m, depth=2, dim_head=64, heads=16, architecture="transformer")
    w = ConditionalFlowMatcherWrapper(
        flowhigh=g, cfm_method="independent_cfm_adaptive", torchdiffeq_ode_method="euler", sigma=1e-4
    )
    ck = torch.load(ROOT / "origin/checkpoints/adaptive/FLowHigh_indep_adaptive_400k.pt", map_location="cpu", weights_only=False)
    w.load_state_dict(ck["model"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return m.to(dev).eval(), w.to(dev).eval(), dev


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="build/calib")
    p.add_argument("--clips", type=int, default=50)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    sources = []
    for pattern, limit in [
        ("../LavaSR.AXERA/.tmp/open_speech/8k/*.wav", 4),
        ("../LavaSR.AXERA/.tmp/test_wavs/libri_8000.wav", 4),
        ("../LavaSR.AXERA/.tmp/test_wavs/libri_long60s.wav", 4),
        ("../LavaSR.AXERA/.tmp/test_wavs/libri_24000.wav", 4),
        ("../LavaSR.AXERA/samples/listen/8k/*.wav", 2),
        ("../LavaSR.AXERA/samples/listen/16k/*.wav", 2),
    ]:
        for f in sorted(Path(ROOT).glob(pattern)):
            sources.extend(chunks(f, limit=limit))
            if len(sources) >= args.clips:
                break
        if len(sources) >= args.clips:
            break
    if not sources:
        raise SystemExit("no source audio")
    sources = sources[: args.clips]
    print(f"{len(sources)} clips")

    m, w, dev = build()
    rng = np.random.default_rng(args.seed)
    flow_x, flow_c, flow_emb = [], [], []
    mels = []
    final_mels = []

    with torch.inference_mode():
        for ci, x in enumerate(sources):
            c = torch.from_numpy(x).unsqueeze(0).to(dev)
            cond = m.encode(c)
            if cond.shape[1] != FRAMES:
                raise RuntimeError(f"bad frames {cond.shape}")
            cond_np = cond.cpu().numpy().astype(np.float32)
            eps = rng.standard_normal(cond_np.shape).astype(np.float32)
            y = cond + torch.from_numpy(eps).to(dev)
            for i, t in enumerate([0.0, 0.25, 0.5, 0.75]):
                t_t = torch.tensor([t], device=dev)
                flow_x.append(y.cpu().numpy().astype(np.float32))
                flow_c.append(cond_np)
                flow_emb.append(w.flowhigh.sinu_pos_emb(t_t).cpu().numpy().astype(np.float32))
                v = w.flowhigh.forward(y, times=t_t, cond=cond, cond_drop_prob=0.0)
                y = y + 0.25 * v
            flow_x.append(y.cpu().numpy().astype(np.float32))
            flow_c.append(cond_np)
            flow_emb.append(w.flowhigh.sinu_pos_emb(torch.tensor([1.0], device=dev)).cpu().numpy().astype(np.float32))
            mels.append(y.transpose(1, 2).cpu().numpy().astype(np.float32))
            final_mels.append(cond_np)

            y2 = cond + torch.from_numpy((1e-4 * rng.standard_normal(cond_np.shape)).astype(np.float32)).to(dev)
            flow_x.append(y2.cpu().numpy().astype(np.float32))
            flow_c.append(cond_np)
            flow_emb.append(w.flowhigh.sinu_pos_emb(torch.tensor([0.0], device=dev)).cpu().numpy().astype(np.float32))

    out = Path(args.out_dir)
    for name, arrs in [("flow_x", flow_x), ("flow_cond", flow_c), ("flow_time_emb", flow_emb), ("bigvgan_mel", mels)]:
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        for i, a in enumerate(arrs):
            np.save(d / f"{i:04d}.npy", a)
        with tarfile.open(out / f"{name}.tar.gz", "w:gz") as tar:
            for f in sorted(d.glob("*.npy")):
                tar.add(f, arcname=f.name)
        print(name, len(arrs))
    for name, arrs in [("flow_x_nct", flow_x), ("flow_cond_nct", flow_c)]:
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        for i, a in enumerate(arrs):
            np.save(d / f"{i:04d}.npy", a.transpose(0, 2, 1))
        with tarfile.open(out / f"{name}.tar.gz", "w:gz") as tar:
            for f in sorted(d.glob("*.npy")):
                tar.add(f, arcname=f.name)
        print(name, len(arrs))

    gt_mels = []
    gt_dir = out / "gt_mel"
    gt_dir.mkdir(parents=True, exist_ok=True)
    gt_paths = [
        Path("/data/shared/huyuan/8860_export/fastenhancer.axera/samples/p232_013_original.wav"),
        Path("../LavaSR.AXERA/.tmp/test_wavs/libri_48000.wav"),
    ]
    for gt_path in gt_paths:
        if not gt_path.exists():
            continue
        gx = load_audio(gt_path)
        peak = float(np.max(np.abs(gx))) if gx.size else 0
        if peak > 0:
            gx = gx / peak
        for k in range(max(1, len(gx) // CLIP + 1)):
            seg = gx[k * CLIP:(k + 1) * CLIP]
            if len(seg) == 0:
                break
            if len(seg) < CLIP:
                seg = np.pad(seg, (0, CLIP - len(seg)), mode="reflect")
            with torch.inference_mode():
                gm = m.encode(torch.from_numpy(seg).unsqueeze(0).to(dev))
            arr = gm.transpose(1, 2).cpu().numpy().astype(np.float32)
            np.save(gt_dir / f"{len(list(gt_dir.glob('*.npy'))):04d}.npy", arr)
            gt_mels.append(arr)
    print("gt_mel", len(gt_mels))
    if gt_mels:
        base = len(mels)
        for i, arr in enumerate(gt_mels):
            np.save(out / "bigvgan_mel" / f"{base + i:04d}.npy", arr)
        with tarfile.open(out / "bigvgan_mel.tar.gz", "w:gz") as tar:
            for f in sorted((out / "bigvgan_mel").glob("*.npy")):
                tar.add(f, arcname=f.name)
        print("bigvgan_mel", base + len(gt_mels))
    t100 = out / "bigvgan_mel_T100"
    t100.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted((out / "bigvgan_mel").glob("*.npy")):
        a = np.load(f)
        for k in range(0, a.shape[-1] - 99, 100):
            np.save(t100 / f"{n:04d}.npy", a[..., k:k + 100].copy())
            n += 1
    with tarfile.open(out / "bigvgan_mel_T100.tar.gz", "w:gz") as tar:
        for f in sorted(t100.glob("*.npy")):
            tar.add(f, arcname=f.name)
    print("bigvgan_mel_T100", n)

    (out / "meta.json").write_text(json.dumps({
        "frames": FRAMES,
        "clips": len(sources),
        "flow_count": len(flow_x),
        "sources": [str(p) for p in sources[:0]],
    }, indent=2))


if __name__ == "__main__":
    main()
