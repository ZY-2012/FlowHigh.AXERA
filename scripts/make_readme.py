from pathlib import Path

GH = "https://cdn.jsdelivr.net/gh/ZY-2012/FlowHigh.AXERA@master/samples/listen"
HF = "https://huggingface.co/HY-2012/FlowHigh.AXERA/resolve/main/samples/listen"
ROOT = Path(__file__).resolve().parents[1]

en = ["OSR_us_000_0010", "OSR_us_000_0013", "OSR_us_000_0057", "OSR_us_000_0059"]
zh = ["OSR_cn_000_0072", "OSR_cn_000_0073", "OSR_cn_000_0074", "OSR_cn_000_0075"]
sp = ["speech_2", "speech_3"]


def cell(kind, inp, out, spec_in, spec_out, hf):
    if hf:
        a = f'<audio controls style="width:220px" src="{HF}/{inp}"></audio>'
        b = f'<audio controls style="width:220px" src="{HF}/{out}"></audio>'
    else:
        a = f'[▶]({GH}/{inp})'
        b = f'[▶]({GH}/{out})'
    return f'| {kind} | {a}<br><img src="samples/listen/spectrograms/{spec_in}" width="360"> | {b}<br><img src="samples/listen/spectrograms/{spec_out}" width="360"> |'


def table(title, name, kind, inp, out, spec_in, spec_out, hf):
    return f"**{name}**\n\n| 版本 | 输入 | FlowHigh 输出 |\n|---|---|---|\n{cell(kind, inp, out, spec_in, spec_out, hf)}\n"


def build(hf=False):
    lines = []
    if hf:
        lines += [
            "---",
            "library_name: axera",
            "pipeline_tag: audio-to-audio",
            "base_model: kiriyamaX/FLowHigh_indep_adaptive_400k",
            "language:",
            "- en",
            "- zh",
            "tags:",
            "- axera",
            "- ax650",
            "- npu3",
            "- flowhigh",
            "- speech-super-resolution",
            "- audio-super-resolution",
            "- bandwidth-extension",
            "- real-time",
            "- python",
            "---",
            "",
        ]
    lines.append("# FlowHigh.AXERA")
    lines.append("")
    lines.append("**8/16 kHz 语音 → 48 kHz 超分 · AX650 NPU3 · Python RTF 0.24（20–60 s 音频）· 2 个 axmodel 共 58 MB · 不做降噪**")
    lines.append("")
    if hf:
        badges = "![RTF](https://img.shields.io/badge/RTF-0.24%20Python-brightgreen) ![Size](https://img.shields.io/badge/axmodel-58%20MB-green) [![GitHub](https://img.shields.io/badge/GitHub-ZY--2012%2FFlowHigh.AXERA-black)](https://github.com/ZY-2012/FlowHigh.AXERA)"
    else:
        badges = "![RTF](https://img.shields.io/badge/RTF-0.24%20Python-brightgreen) ![Size](https://img.shields.io/badge/axmodel-58%20MB-green) [![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE) [![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-FlowHigh.AXERA-yellow)](https://huggingface.co/HY-2012/FlowHigh.AXERA)"
    lines.append(badges)
    lines.append("")
    if hf:
        lines += [
            "- GitHub: https://github.com/ZY-2012/FlowHigh.AXERA",
            "- 上游: https://github.com/jjunak-yun/FLowHigh_code",
            "",
            "## 目录",
            "",
            "- 快速开始",
            "- 结果",
            "- 推理速度",
            "- 数据来源",
            "- License",
            "",
        ]
    else:
        lines += [
            "- HuggingFace: https://huggingface.co/HY-2012/FlowHigh.AXERA",
            "- GitHub: https://github.com/ZY-2012/FlowHigh.AXERA",
            "- 上游: https://github.com/jjunak-yun/FLowHigh_code",
            "",
            "## 目录",
            "",
            "- 快速开始",
            "- 结果",
            "- 推理速度",
            "- 模型转换",
            "- 数据来源",
            "- License",
            "",
        ]
    lines += [
        "## 快速开始（AX650 板端）",
        "",
        "从 HuggingFace 下载 `models/`、`assets/`、`python/`、`samples/`，在包根目录运行：",
        "",
        "```bash",
        "bash run_ax650.sh samples/listen/8k/OSR_us_000_0010_input_8k.wav output.wav",
        "```",
        "",
        "输入支持 8 kHz / 16 kHz，输出固定 48 kHz。默认 1-step flow、`std=1e-4`；不做降噪。",
        "",
        "模型为 `FLowHigh_indep_adaptive_400k`（481,490,826 bytes，Flow 35.4M + BigVGAN 14.0M 参数）。AX650 侧拆成 Flow 38 MB + BigVGAN 20 MB 两个 U16/S8 axmodel；38 MB 是 S8 权重后的正常大小。",
        "",
        "## 结果",
        "",
        "### 8 kHz 输入（Open Speech / Samplelib）",
        "",
        "#### English",
        "",
    ]
    for n in en:
        lines += table("8k", n, "8k", f"8k/{n}_input_8k.wav", f"48k_from8k/{n}_output_48k.wav",
                       f"{n}_input_8k_spec.png", f"{n}_output_48k_spec.png", hf).splitlines()
        lines.append("")
    lines += ["#### Chinese", ""]
    for n in zh:
        lines += table("8k", n, "8k", f"8k/{n}_input_8k.wav", f"48k_from8k/{n}_output_48k.wav",
                       f"{n}_input_8k_spec.png", f"{n}_output_48k_spec.png", hf).splitlines()
        lines.append("")
    lines += ["#### Samplelib（1 分钟，频谱图为前 10 s）", ""]
    lines += table("8k", "samplelib_speech_1m", "8k", "8k/samplelib_speech_1m_input_8k.wav",
                   "48k_from8k/samplelib_speech_1m_output_48k.wav",
                   "samplelib_speech_1m_input_8k_spec.png", "samplelib_speech_1m_output_48k_spec.png", hf).splitlines()
    lines += ["", "### 16 kHz 参照（人声）", "",
              "同一批人声降采样到 16 kHz 后输入；8k 行为 8 kHz 输入，16k 行为 16 kHz 输入。", ""]
    for n in sp:
        lines += table("16k", n, None, None, None, None, None, hf).splitlines()[:2]
        # replace with full rows for 8k and 16k
        lines = lines[:-2]
        lines += [f"**{n}**", "", "| 版本 | 输入 | FlowHigh 输出 |", "|---|---|---|"]
        lines.append(cell("8k", f"8k/{n}_input_8k.wav", f"48k_from8k/{n}_output_48k.wav",
                          f"{n}_input_8k_spec.png", f"{n}_output_8k_48k_spec.png", hf))
        lines.append(cell("16k", f"16k/{n}_input_16k.wav", f"48k_from16k/{n}_output_48k.wav",
                          f"{n}_input_16k_spec.png", f"{n}_output_16k_48k_spec.png", hf))
        lines.append("")
    lines += [
        "## 推理速度：RTF",
        "",
        "| 链路 | Python RTF |",
        "|---|---:|",
        "| 8/16 kHz → 48 kHz | **0.24** |",
        "",
        "20–60 s 音频，1-step flow，BigVGAN 1 s 分块、0.2 s overlap。5 s 短音频因分块/启动开销会略高；C++ 运行时尚未提供。",
        "",
    ]
    if not hf:
        lines += [
            "## 模型转换",
            "",
            "```bash",
            "python model_convert/export_flowhigh.py --frames 500 --skip-vocoder --out-dir models",
            "python model_convert/export_flowhigh.py --frames 100 --skip-flow --out-dir models",
            "python model_convert/make_calib.py --out-dir build/calib --clips 50",
            "python model_convert/prepare_compile.py",
            "PULSAR2_IMAGE=docker-registry.aitsw.axera-tech.com/pulsar2:8.0 \\",
            "  bash model_convert/compile.sh flowhigh_nct bigvgan_T100",
            "```",
            "",
            "Flow NPU 输入为 `[1,256,500]`（channel-first），图内转置到 `[1,500,256]`；BigVGAN 按 1 s 编译，运行时 overlap 交叉淡化。详见 `model_convert/README.md`。",
            "",
        ]
    lines += [
        "## 数据来源",
        "",
        "- Open Speech Repository（English / Chinese 8k）：http://voiptroubleshooter.com/open_speech/index.html",
        "- Samplelib speech 1m：https://samplelib.com/mp3/sample-speech-1m.mp3",
        "- 16 kHz 参照：公开人声样例经 48k->16k 降采样。",
        "",
        "## License",
        "",
        "MIT；FLowHigh 与 BigVGAN 均为 MIT。模型 checkpoint 来自上游/公开镜像，仓库不重复分发训练权重。",
        "",
    ]
    return "\n".join(lines)


(ROOT / "README.md").write_text(build(False))
(ROOT / "release" / "README.md").write_text(build(True))
print("written")
