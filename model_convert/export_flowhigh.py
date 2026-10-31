import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "origin" / "FLowHigh_code"
VOC_DIR = UPSTREAM / "vocoder" / "BIGVGAN"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(VOC_DIR))

N_MELS = 256


def neutralise_cuda():
    original_load = torch.load

    def load_cpu(*args, **kwargs):
        kwargs["map_location"] = "cpu"
        return original_load(*args, **kwargs)

    torch.load = load_cpu
    torch.nn.Module.cuda = lambda self, *a, **k: self
    torch.Tensor.cuda = lambda self, *a, **k: self


def build_vocoder():
    from bigvgan.env import AttrDict
    from bigvgan.models import BigVGAN

    cfg_path = UPSTREAM / "vocoder/BIGVGAN/config/bigvgan_48khz_256band_config.json"
    ckpt_path = ROOT / "origin/checkpoints/bigvgan/g_48_00850000"
    cfg = AttrDict(json.load(open(cfg_path)))
    voc = BigVGAN(cfg)
    voc.load_state_dict(torch.load(ckpt_path)["generator"])
    voc.remove_weight_norm()
    voc.eval()
    return voc, cfg


def build_flow():
    from cfm_superresolution import ConditionalFlowMatcherWrapper, FLowHigh, MelVoco

    enc = MelVoco(
        n_mels=N_MELS,
        sampling_rate=48000,
        f_max=24000,
        n_fft=2048,
        win_length=2048,
        hop_length=480,
        vocoder="bigvgan",
        vocoder_config=str(UPSTREAM / "vocoder/BIGVGAN/config/bigvgan_48khz_256band_config.json"),
        vocoder_path=str(ROOT / "origin/checkpoints/bigvgan/g_48_00850000"),
    )
    gen = FLowHigh(
        dim_in=N_MELS,
        audio_enc_dec=enc,
        depth=2,
        dim_head=64,
        heads=16,
        architecture="transformer",
    )
    wrapper = ConditionalFlowMatcherWrapper(
        flowhigh=gen,
        cfm_method="independent_cfm_adaptive",
        torchdiffeq_ode_method="euler",
        sigma=1e-4,
    )
    ckpt = torch.load(ROOT / "origin/checkpoints/adaptive/FLowHigh_indep_adaptive_400k.pt", weights_only=False)
    wrapper.load_state_dict(ckpt["model"])
    gen.eval()
    return gen


class FlowField(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, time_emb, cond):
        y = self.net.forward(x.transpose(1, 2), time_emb=time_emb,
                              cond=cond.transpose(1, 2), cond_drop_prob=0.0)
        return y.transpose(1, 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=500)
    p.add_argument("--out-dir", type=Path, default=ROOT / "models")
    p.add_argument("--skip-vocoder", action="store_true")
    p.add_argument("--skip-flow", action="store_true")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    neutralise_cuda()

    import onnxruntime as ort

    if not args.skip_vocoder:
        from model_convert.bigvgan_static import make_static

        voc, cfg = build_vocoder()
        make_static(voc, torch.randn(1, cfg.num_mels, 50))
        mel = torch.randn(1, cfg.num_mels, args.frames)
        out = args.out_dir / f"bigvgan_T{args.frames}.onnx"
        torch.onnx.export(
            voc,
            (mel,),
            str(out),
            input_names=["mel"],
            output_names=["wav"],
            opset_version=17,
            do_constant_folding=False,
            dynamo=False,
        )
        ref = voc(mel).detach().numpy()
        got = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"]).run(
            None, {"mel": mel.numpy()}
        )[0]
        err = float(np.abs(ref - got).max())
        print(json.dumps({"file": str(out), "mb": out.stat().st_size / 1e6, "max_abs_err": err}))

    if not args.skip_flow:
        gen = build_flow()
        field = FlowField(gen).eval()
        x = torch.randn(1, N_MELS, args.frames)
        cond = torch.randn(1, N_MELS, args.frames)
        time_emb = gen.sinu_pos_emb(torch.tensor([0.25]))
        out = args.out_dir / f"flowfield_T{args.frames}.onnx"
        torch.onnx.export(
            field,
            (x, time_emb, cond),
            str(out),
            input_names=["x", "time_emb", "cond"],
            output_names=["velocity"],
            opset_version=17,
            do_constant_folding=False,
            dynamo=False,
        )
        ref = field(x, time_emb, cond).detach().numpy()
        got = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"]).run(
            None, {"x": x.numpy(), "time_emb": time_emb.detach().numpy(), "cond": cond.numpy()}
        )[0]
        err = float(np.abs(ref - got).max())
        print(json.dumps({"file": str(out), "mb": out.stat().st_size / 1e6, "max_abs_err": err}))


if __name__ == "__main__":
    main()
