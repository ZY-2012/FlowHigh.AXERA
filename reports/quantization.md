# FlowHigh AX650 quantization report

## Target

- Board: AX650N, NPU3, axengine 2.12.0s, compiler 8.0.
- Flow: U16 activations / S8 weights, 38 MB.
- BigVGAN: U16 activations / S8 weights, 20 MB.
- Input: 8 kHz / 16 kHz -> 48 kHz.
- No denoising.

## Graph split

```text
host: resample_poly -> log-mel [1,256,T] -> 1-step flow -> BigVGAN chunks -> STFT merge
NPU:  flow velocity (37 MB) + BigVGAN 1 s chunks (20 MB)
```

The flow NPU input is channel-first `[1,256,500]`; the wrapper transposes to `[1,500,256]` internally. This is required: Pulsar2 lowers 3D tensors channel-first, and feeding `[1,500,256]` directly makes attention run on the wrong axis.

## Numerical checks

| Check | Result |
|---|---|
| Flow ONNX vs PyTorch velocity | max err 1.5e-4 |
| Flow U16/S8 on board vs ONNX | SNR 32.96 dB, corr 0.99977 |
| Flow latency, 5 s | 21 ms (after warmup) |
| BigVGAN ONNX vs PyTorch | max err 8.2e-6 |
| BigVGAN U16/S8 on board vs ONNX, 1 s | SNR 20.55 dB, corr 0.99560 |
| BigVGAN latency, 1 s | ~93 ms |
| BigVGAN chunk overlap | 0.2 s; boundary jump reduced from 0.44 to 0.005 |

BigVGAN is the sensitive part; its direct waveform SNR is low, but the final speech quality after spectral merging is close to the FP32 pipeline.

## Full pipeline quality

VCTK `p232_013`, 48 kHz reference. Board uses 1-step flow with `std=1e-4`, BigVGAN 1 s / 0.2 s overlap. Baseline and board use different RNG seeds, so the PESQ/LSD comparison is indicative; flow and BigVGAN SNR rows are strict same-input comparisons.

| Input | Pipeline | PESQ-WB | STOI | Mel LSD |
|---|---|---|---:|---:|
| 8 kHz | PyTorch FP32 | 3.872 | 0.9981 | 2.323 |
| 8 kHz | AX650 U16/S8 | **3.876** | **0.9982** | 2.333 |
| 16 kHz | PyTorch FP32 | 4.570 | 1.0000 | 2.315 |
| 16 kHz | AX650 U16/S8 | **4.573** | **1.0000** | 2.302 |

## Speed

5 s input, Python/axengine, one flow step:

| Stage | Time |
|---|---:|
| Flow | 33 ms |
| BigVGAN (6 x 1 s chunks) | 578 ms |
| Host STFT/mel/postprocess | ~600 ms |
| Total | 0.94–1.22 s |
| RTF | 0.19–0.24 |

The current runtime is Python-only. Moving STFT/postprocess to C++ and using a native flow/vocoder loop is the next optimization; the NPU portion itself is 0.61 s for 5 s.

## Long audio

The flow graph is fixed at 500 mel frames (5 s). Longer audio is processed in 5 s chunks with 0.5 s overlap and waveform crossfade; BigVGAN uses 1 s chunks with 0.2 s overlap. The 60 s Samplelib input was processed end-to-end with no boundary clicks above the normal waveform derivative.

## Limitations

- The 5 s flow chunk is a hard graph size; longer context would need a different compiled graph.
- No denoising; FLowHigh is BWE only.
- C++ runtime is not included in this first quantization milestone.
