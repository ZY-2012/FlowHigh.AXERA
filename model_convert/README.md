# FlowHigh AX650 conversion

This directory converts the upstream FLowHigh checkpoint into two AX650 NPU models:

- `flowhigh.axmodel`: flow velocity, inputs `x[1,256,500]`, `cond[1,256,500]`, `time_emb[1,1024]`, output `velocity[1,256,500]`.
- `bigvgan.axmodel`: BigVGAN 48 kHz vocoder, input `mel[1,256,100]`, output `wav[1,1,48000]`.

Both use **U16 activations / S8 weights**.

## Why the flow model is NCT

The PyTorch flow model uses `[B, T, C]`. AX650/Pulsar2 lowers 3D tensors as channel-first, so the NPU graph must expose `[B, C, T]` and transpose internally. The exported wrapper does this. Feeding `[B, T, C]` to the NPU produces wrong attention, which looks like a failed quantization.

## Patches to upstream

`origin/FLowHigh_code` must be patched with `upstream.patch`:

- explicit rotary positions and outer product
- MatMul attention instead of `Einsum`
- explicit L2 normalization and softmax
- tanh-approximate GELU
- optional `time_emb` argument to skip the internal time MLP

Apply from the project root:

```bash
git -C origin/FLowHigh_code apply ../../model_convert/upstream.patch
```

## Export

```bash
python model_convert/export_flowhigh.py --frames 500 --skip-vocoder --out-dir models
python model_convert/export_flowhigh.py --frames 100 --skip-flow --out-dir models
```

`bigvgan_static.py` materialises the alias-free resampler kernels so the ONNX tracer sees fixed depthwise kernels.

## Calibration

```bash
python model_convert/make_calib.py --out-dir build/calib --clips 50
```

This writes 300 flow samples and 55 BigVGAN mel samples, including VCTK/Open Speech/Samplelib and two 48 kHz references.

## Pulsar2

```bash
PULSAR2_IMAGE=docker-registry.aitsw.axera-tech.com/pulsar2:8.0 \
  bash model_convert/compile.sh flowhigh_nct bigvgan_T100
```

Outputs:

```text
build/compile/flowhigh_nct/model.axmodel
build/compile/bigvgan_T100/model.axmodel
```

Copy them to `models/flowhigh.axmodel` and `models/bigvgan.axmodel`.

The BigVGAN graph is compiled for 1 second (`T=100`). The runtime decodes 1-second chunks with 0.2 seconds of overlap and crossfade.
