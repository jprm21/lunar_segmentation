"""Exporta un modelo PyTorch a ONNX y verifica consistencia básica de inferencia.

Uso:
    python scripts/export_onnx_verify.py --input model.pth --output modelo_n/modelo1.onnx
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.unet_mobilenet import UNetMobileNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exporta checkpoint .pth a ONNX y compara salida PyTorch vs ONNX Runtime"
    )
    parser.add_argument("--input", required=True, type=Path, help="Ruta al checkpoint .pth")
    parser.add_argument("--output", required=True, type=Path, help="Ruta de salida del archivo .onnx")
    parser.add_argument("--image-size", default=384, type=int, help="Tamaño H=W del input dummy")
    parser.add_argument("--num-classes", default=5, type=int, help="Número de clases del modelo")
    parser.add_argument("--opset", default=11, type=int, help="Versión opset ONNX")
    parser.add_argument("--seed", default=1234, type=int, help="Semilla para input dummy reproducible")
    return parser.parse_args()


def load_model(weights_path: Path, num_classes: int) -> torch.nn.Module:
    if not weights_path.exists():
        raise FileNotFoundError(f"No existe el checkpoint: {weights_path}")

    model = UNetMobileNet(num_classes=num_classes, pretrained=False)
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def export_onnx(model: torch.nn.Module, output_path: Path, image_size: int, opset: int, seed: int) -> np.ndarray:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    dummy = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True,
    )

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    return dummy.numpy()


def verify_outputs(model: torch.nn.Module, dummy_np: np.ndarray, onnx_path: Path) -> float:
    with torch.no_grad():
        torch_out = model(torch.from_numpy(dummy_np)).cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = session.run(["output"], {"input": dummy_np})[0]

    max_abs_diff = float(np.max(np.abs(torch_out - ort_out)))
    return max_abs_diff


def main() -> None:
    args = parse_args()

    model = load_model(args.input, args.num_classes)
    dummy_np = export_onnx(model, args.output, args.image_size, args.opset, args.seed)
    max_diff = verify_outputs(model, dummy_np, args.output)

    print(f"✓ ONNX exportado en: {args.output}")
    print(f"✓ Diferencia máxima absoluta (PyTorch vs ONNX Runtime): {max_diff:.6e}")


if __name__ == "__main__":
    main()
