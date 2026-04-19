import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import onnx
import tensorflow as tf
import torch
from onnx_tf.backend import prepare

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.datasets.lusnar_dataset import LuSNARDataset
from src.models.unet_mobilenet import UNetMobileNet

MODEL_PATH = PROJECT_ROOT / "best_model.pth"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
ONNX_PATH = OUTPUT_DIR / "model.onnx"
TF_PATH = OUTPUT_DIR / "model_tf"
TFLITE_PATH = OUTPUT_DIR / "model_int8.tflite"

IMAGE_SIZE = 384
NUM_CLASSES = 5
VAL_SCENES = [3, 5, 7]
CALIBRATION_SAMPLES = 100
PREVIEW_SAMPLES = 5

# Normalización estándar (ImageNet)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Índices de clase del dataset: 0=regolith, 1=crater, 2=rock, 3=mountain, 4=sky
MASK_COLOR_MAP = {
    0: (255, 0, 255),   # regolith -> magenta
    1: (0, 0, 255),     # crater -> blue
    2: (255, 255, 0),   # rock -> yellow
    3: (165, 42, 42),   # mountain -> brown
    4: (0, 255, 255),   # sky -> cyan
}


def normalize_chw(image_chw: np.ndarray) -> np.ndarray:
    """Normaliza una imagen CHW en [0,1] usando mean/std de ImageNet."""
    return (image_chw - MEAN[:, None, None]) / STD[:, None, None]


def load_pytorch_model() -> torch.nn.Module:
    print("[1/5] Cargando modelo PyTorch en CPU...")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No existe el archivo de pesos: {MODEL_PATH}")

    model = UNetMobileNet(num_classes=NUM_CLASSES, pretrained=False)
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    print(f"  ✓ Modelo cargado desde: {MODEL_PATH}")
    return model


def export_to_onnx(model: torch.nn.Module) -> None:
    print("[2/5] Exportando a ONNX...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True,
    )

    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)
    print(f"  ✓ ONNX guardado en: {ONNX_PATH}")


def convert_onnx_to_tf() -> None:
    print("[3/5] Convirtiendo ONNX -> TensorFlow SavedModel...")
    if TF_PATH.exists():
        shutil.rmtree(TF_PATH)

    onnx_model = onnx.load(str(ONNX_PATH))
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(str(TF_PATH))

    print(f"  ✓ SavedModel guardado en: {TF_PATH}")


def build_validation_dataset() -> LuSNARDataset:
    dataset = LuSNARDataset(
        root_dir=PROJECT_ROOT / "data",
        image_size=IMAGE_SIZE,
        scenes=VAL_SCENES,
    )
    return dataset


def representative_dataset(dataset: LuSNARDataset, signature_shape) -> callable:
    """Devuelve generador representativo para cuantización PTQ INT8."""
    is_nhwc = len(signature_shape) == 4 and signature_shape[1] == IMAGE_SIZE and signature_shape[-1] == 3

    def _generator():
        limit = min(CALIBRATION_SAMPLES, len(dataset))
        for idx in range(limit):
            image, _ = dataset[idx]  # image CHW float en [0,1]
            image_np = image.numpy().astype(np.float32)
            image_np = normalize_chw(image_np)
            image_np = np.expand_dims(image_np, axis=0)  # (1,3,384,384)

            if is_nhwc:
                image_np = np.transpose(image_np, (0, 2, 3, 1))

            yield [image_np]

    return _generator


def convert_tf_to_tflite_int8(dataset: LuSNARDataset) -> None:
    print("[4/5] Convirtiendo SavedModel -> TFLite INT8 (PTQ)...")

    loaded = tf.saved_model.load(str(TF_PATH))
    serving_fn = loaded.signatures["serving_default"]
    input_tensor = list(serving_fn.structured_input_signature[1].values())[0]
    input_shape = tuple(int(d) if d is not None else -1 for d in input_tensor.shape)

    converter = tf.lite.TFLiteConverter.from_saved_model(str(TF_PATH))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset(dataset, input_shape)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    TFLITE_PATH.write_bytes(tflite_model)

    print(f"  ✓ TFLite INT8 guardado en: {TFLITE_PATH}")


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in MASK_COLOR_MAP.items():
        colored[mask == class_id] = color
    return colored


def quantize_input(normalized_chw_batch: np.ndarray, input_details: dict) -> np.ndarray:
    """Cuantiza float32 normalizado a INT8 usando scale/zero_point del input tensor."""
    scale, zero_point = input_details["quantization"]
    if scale == 0:
        raise ValueError("Scale de cuantización inválido (0).")

    quantized = np.round(normalized_chw_batch / scale + zero_point)
    quantized = np.clip(quantized, -128, 127).astype(np.int8)
    return quantized


def run_tflite_previews(dataset: LuSNARDataset) -> None:
    print("[5/5] Ejecutando inferencia TFLite INT8 y guardando visualizaciones...")

    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_shape = input_details["shape"].tolist()
    output_shape = output_details["shape"].tolist()

    is_nhwc_input = len(input_shape) == 4 and input_shape[1] == IMAGE_SIZE and input_shape[-1] == 3
    is_nhwc_output = len(output_shape) == 4 and output_shape[-1] == NUM_CLASSES

    for idx in range(min(PREVIEW_SAMPLES, len(dataset))):
        image, mask_gt = dataset[idx]

        image_chw = image.numpy().astype(np.float32)
        image_norm = normalize_chw(image_chw)
        batch_nchw = np.expand_dims(image_norm, axis=0)

        if is_nhwc_input:
            model_input = np.transpose(batch_nchw, (0, 2, 3, 1))
        else:
            model_input = batch_nchw

        quant_input = quantize_input(model_input, input_details)

        interpreter.set_tensor(input_details["index"], quant_input)
        interpreter.invoke()

        pred_raw = interpreter.get_tensor(output_details["index"])

        out_scale, out_zero = output_details["quantization"]
        if out_scale != 0:
            pred_raw = (pred_raw.astype(np.float32) - out_zero) * out_scale

        if is_nhwc_output:
            pred_logits = pred_raw[0]
            pred_mask = np.argmax(pred_logits, axis=-1).astype(np.uint8)
        else:
            pred_logits = pred_raw[0]
            pred_mask = np.argmax(pred_logits, axis=0).astype(np.uint8)

        orig_hwc = np.transpose(image_chw, (1, 2, 0))
        gt_mask = mask_gt.numpy().astype(np.uint8)

        gt_color = mask_to_color(gt_mask)
        pred_color = mask_to_color(pred_mask)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(np.clip(orig_hwc, 0, 1))
        axs[0].set_title("Imagen original")
        axs[0].axis("off")

        axs[1].imshow(gt_color)
        axs[1].set_title("Máscara GT")
        axs[1].axis("off")

        axs[2].imshow(pred_color)
        axs[2].set_title("Máscara predicha (TFLite INT8)")
        axs[2].axis("off")

        out_file = OUTPUT_DIR / f"segmentation_preview_{idx + 1}.png"
        fig.tight_layout()
        fig.savefig(out_file, dpi=150)
        plt.close(fig)

        print(f"  ✓ Preview guardado: {out_file}")


def main():
    print("=== Pipeline de conversión PyTorch -> ONNX -> TF -> TFLite INT8 ===")

    model = load_pytorch_model()
    export_to_onnx(model)
    convert_onnx_to_tf()

    val_dataset = build_validation_dataset()
    convert_tf_to_tflite_int8(val_dataset)
    run_tflite_previews(val_dataset)

    print("✅ Proceso completo finalizado.")


if __name__ == "__main__":
    main()
