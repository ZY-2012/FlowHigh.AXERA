import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from flowhigh_sdk import audio


def decode_mel(voc, mel, chunk, overlap):
    hop = chunk - overlap
    wav = None
    for s in range(0, mel.shape[-1], hop):
        m = mel[..., s:s + chunk]
        if m.shape[-1] < chunk:
            m = np.pad(m, ((0, 0), (0, 0), (0, chunk - m.shape[-1])), mode="edge")
        c = voc.run(None, {"mel": np.ascontiguousarray(m)})[0][0, 0]
        if wav is None:
            wav = c
        else:
            n = min(overlap * 480, len(wav), len(c))
            if n > 0:
                fade = np.linspace(0.0, 1.0, n, dtype=np.float32)
                wav[-n:] = wav[-n:] * (1.0 - fade) + c[:n] * fade
            wav = np.concatenate([wav, c[n:]])
    return wav.astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--models", default="models")
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--std", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chunk", type=int, default=100)
    p.add_argument("--overlap", type=int, default=20)
    p.add_argument("--flow-overlap", type=int, default=50)
    p.add_argument("--time-emb", default="assets/time_emb_4.npy")
    args = p.parse_args()

    x, sr = sf.read(args.input, always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = np.asarray(x, dtype=np.float32)
    target = int(round(audio.SR / sr * x.size))
    x = audio.resample(x, sr)
    peak = float(np.max(np.abs(x)))
    if peak > 0:
        x = x / peak
    cond_wave = x.astype(np.float32)
    mel = audio.log_mel(cond_wave)[0].T[None]
    frames = 500
    if mel.shape[-1] < frames:
        mel = np.pad(mel, ((0, 0), (0, 0), (0, frames - mel.shape[-1])), mode="edge")
    total = mel.shape[-1]
    if total > frames:
        n_chunks = (total - frames + (frames - args.flow_overlap) - 1) // (frames - args.flow_overlap) + 1
        padded = frames + (n_chunks - 1) * (frames - args.flow_overlap)
        if padded > total:
            mel = np.pad(mel, ((0, 0), (0, 0), (0, padded - total)), mode="edge")
    padded_total = mel.shape[-1]

    import axengine

    flow = axengine.InferenceSession(str(Path(args.models) / "flowhigh.axmodel"))
    voc = axengine.InferenceSession(str(Path(args.models) / "bigvgan.axmodel"))
    emb = np.load(args.time_emb).astype(np.float32)
    rng = np.random.RandomState(args.seed)
    hop = frames - args.flow_overlap
    starts = list(range(0, max(1, padded_total - frames + 1), hop))

    t0 = time.time()
    flow_ms = []
    out = None
    for s in starts:
        c = mel[..., s:s + frames]
        y = c + args.std * rng.randn(*c.shape).astype(np.float32)
        for i in range(args.steps):
            t1 = time.time()
            v = flow.run(None, {
                "x": np.ascontiguousarray(y),
                "cond": np.ascontiguousarray(c),
                "time_emb": np.ascontiguousarray(emb[i]),
            })[0]
            flow_ms.append((time.time() - t1) * 1000)
            y = y + (1.0 / args.steps) * v
        wav = decode_mel(voc, y, args.chunk, args.overlap)
        if out is None:
            out = wav
        else:
            n = min(args.flow_overlap * 480, len(out), len(wav))
            if n > 0:
                fade = np.linspace(0.0, 1.0, n, dtype=np.float32)
                out[-n:] = out[-n:] * (1.0 - fade) + wav[:n] * fade
            out = np.concatenate([out, wav[n:]])
    out = out[:int(total * 480)]
    out = audio.postprocess(out, cond_wave[:len(out)], len(out))[:target]
    sf.write(args.output, out, audio.SR, subtype="PCM_16")
    dt = time.time() - t0
    print(f"chunks={len(starts)} flow_ms={flow_ms} total_ms={dt*1000:.1f} rtf={dt/(x.size/audio.SR):.4f}")


if __name__ == "__main__":
    main()
