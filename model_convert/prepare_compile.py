import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model_convert" / "configs"
OUT.mkdir(parents=True, exist_ok=True)


def cfg(name, input_onnx, input_shapes, calibs, size):
    return {
        "model_type": "ONNX",
        "target_hardware": "AX650",
        "npu_mode": "NPU3",
        "input": f"/workspace/models/{input_onnx}",
        "output_dir": f"/workspace/build/compile/{name}",
        "output_name": "model.axmodel",
        "work_dir": f"/workspace/build/compile/{name}_work",
        "input_shapes": input_shapes,
        "onnx_opt": {
            "disable_onnx_optimization": False,
            "enable_onnxsim": False,
            "model_check": True,
        },
        "quant": {
            "calibration_method": "MinMax",
            "precision_analysis": False,
            "highest_mix_precision": False,
            "layer_configs": [{
                "start_tensor_names": ["DEFAULT"],
                "end_tensor_names": ["DEFAULT"],
                "data_type": "U16",
                "weight_data_type": "S8",
                "output_data_type": "U16",
            }],
            "input_configs": [{
                "tensor_name": tensor,
                "calibration_dataset": f"/workspace/build/calib/{calib}",
                "calibration_format": "Numpy",
                "calibration_size": size,
                "calibration_mean": [],
                "calibration_std": [],
            } for tensor, calib in calibs],
        },
        "input_processors": [],
    }


configs = {
    "flowhigh_nct": cfg(
        "flowhigh_nct",
        "flowfield_T500.onnx",
        "x:1x256x500,cond:1x256x500,time_emb:1x1024",
        [("x", "flow_x_nct.tar.gz"), ("cond", "flow_cond_nct.tar.gz"), ("time_emb", "flow_time_emb.tar.gz")],
        300,
    ),
    "bigvgan_T100": cfg(
        "bigvgan_T100",
        "bigvgan_T100.onnx",
        "mel:1x256x100",
        [("mel", "bigvgan_mel_T100.tar.gz")],
        275,
    ),
}

for name, c in configs.items():
    p = OUT / f"{name}.json"
    p.write_text(json.dumps(c, indent=2) + "\n")
    print(p)
