"""
conversion.py
Pipeline: PyTorch -> ONNX -> SavedModel (onnx-tf) -> TFLite FP16

IMPORTANTE:
Este modelo fue entrenado SIN normalización ImageNet.
El dataset usa TF.to_tensor() solamente, por lo que el input esperado es [0,1].
Por eso NO se aplica (x - mean) / std en este pipeline.

FP16: pesos en float16, input/output en float32.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import onnx
import tensorflow as tf
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.datasets.lusnar_dataset import LuSNARDataset
from src.models.unet_mobilenet import UNetMobileNet


# -----------------------------
# Configuración
# -----------------------------
MODEL_PATH    = PROJECT_ROOT / "best_model.pth"
OUTPUT_DIR    = PROJECT_ROOT / "outputs"
ONNX_PATH     = OUTPUT_DIR / "model.onnx"
ONNX_SIM_PATH = OUTPUT_DIR / "model_simplified.onnx"
TF_PATH       = OUTPUT_DIR / "model_tf"
TFLITE_PATH   = OUTPUT_DIR / "model_fp16.tflite"

IMAGE_SIZE      = 384
NUM_CLASSES     = 5
VAL_SCENES      = [3, 5, 7]
PREVIEW_SAMPLES = 5

# Índices: 0=regolith, 1=crater, 2=rock, 3=mountain, 4=sky
MASK_COLOR_MAP = {
    0: (187,  70, 156),
    1: (120,   0, 200),
    2: (232, 250,  80),
    3: (173,  69,  31),
    4: ( 34, 201, 248),
}


# -----------------------------
# Utilidades
# -----------------------------
def mask_to_color(mask: np.ndarray) -> np.ndarray:
    colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in MASK_COLOR_MAP.items():
        colored[mask == class_id] = color
    return colored


def build_val_dataset() -> LuSNARDataset:
    return LuSNARDataset(
        root_dir=PROJECT_ROOT / "data",
        image_size=IMAGE_SIZE,
        scenes=VAL_SCENES,
    )


# -----------------------------
# Paso 1: Cargar modelo PyTorch
# -----------------------------
def load_pytorch_model() -> torch.nn.Module:
    print("[1/5] Cargando modelo PyTorch en CPU...")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No existe: {MODEL_PATH}")

    model = UNetMobileNet(num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    print(f"  ✓ Modelo cargado desde: {MODEL_PATH}")
    return model


# -----------------------------
# Paso 2: Exportar a ONNX
# -----------------------------
def export_to_onnx(model: torch.nn.Module) -> None:
    print("[2/5] Exportando a ONNX...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    torch.onnx.export(
        model,
        dummy,
        str(ONNX_PATH),
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True,
    )

    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)
    print(f"  ✓ ONNX guardado en: {ONNX_PATH}")

    try:
        from onnxsim import simplify
        simplified, ok = simplify(onnx_model)
        if ok:
            onnx.save(simplified, str(ONNX_SIM_PATH))
            print(f"  ✓ ONNX simplificado en: {ONNX_SIM_PATH}")
        else:
            import shutil
            shutil.copy(str(ONNX_PATH), str(ONNX_SIM_PATH))
    except Exception:
        import shutil
        shutil.copy(str(ONNX_PATH), str(ONNX_SIM_PATH))


# -----------------------------
# Paso 3: ONNX -> SavedModel con onnx-tf
# -----------------------------
def convert_onnx_to_savedmodel() -> None:
    print("[3/5] Convirtiendo ONNX -> SavedModel con onnx-tf...")

    import shutil
    if TF_PATH.exists():
        shutil.rmtree(TF_PATH)
    TF_PATH.mkdir(parents=True, exist_ok=True)

    from onnx_tf.backend import prepare

    onnx_model = onnx.load(str(ONNX_SIM_PATH))
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(str(TF_PATH))

    pb_candidates = list(TF_PATH.rglob("saved_model.pb"))
    if not pb_candidates:
        raise FileNotFoundError(f"No se generó saved_model.pb en {TF_PATH}")

    print(f"  ✓ SavedModel guardado en: {pb_candidates[0].parent}")


# -----------------------------
# Paso 4: SavedModel -> TFLite FP16
# -----------------------------
def convert_to_tflite_fp16() -> None:
    print("[4/5] Convirtiendo SavedModel -> TFLite FP16...")

    pb_candidates = list(TF_PATH.rglob("saved_model.pb"))
    if not pb_candidates:
        raise FileNotFoundError(f"No se encontró saved_model.pb en {TF_PATH}")

    savedmodel_dir = str(pb_candidates[0].parent)

    converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_dir)

    # FP16: pesos float16, I/O float32
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    TFLITE_PATH.write_bytes(tflite_model)

    size_mb = TFLITE_PATH.stat().st_size / 1024 / 1024
    print(f"  ✓ TFLite FP16 guardado en: {TFLITE_PATH}")
    print(f"  Tamaño: {size_mb:.1f} MB")
    print("  Nota: input/output son float32, pesos almacenados en FP16")


# -----------------------------
# Paso 5: Inferencia y previews
# -----------------------------
def run_previews(dataset: LuSNARDataset) -> None:
    print("[5/5] Ejecutando inferencia TFLite FP16 y guardando visualizaciones...")

    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    in_shape  = input_details["shape"].tolist()
    out_shape = output_details["shape"].tolist()

    is_nhwc_in  = len(in_shape)  == 4 and in_shape[-1]  == 3
    is_nhwc_out = len(out_shape) == 4 and out_shape[-1] == NUM_CLASSES

    print(f"  Input:  {in_shape}  ({'NHWC' if is_nhwc_in  else 'NCHW'})")
    print(f"  Output: {out_shape} ({'NHWC' if is_nhwc_out else 'NCHW'})")
    print(f"  Input dtype:  {input_details['dtype']}")
    print(f"  Output dtype: {output_details['dtype']}")

    for idx in range(min(PREVIEW_SAMPLES, len(dataset))):
        image, mask_gt = dataset[idx]

        # Dataset ya entrega imagen en [0,1] (TF.to_tensor), sin normalización
        image_chw = image.numpy().astype(np.float32)

        # IMPORTANTE: NO aplicar (x-mean)/std
        image_input = image_chw

        if is_nhwc_in:
            model_input = np.transpose(image_input, (1, 2, 0))
        else:
            model_input = image_input

        model_input = np.expand_dims(model_input, axis=0).astype(np.float32)

        interpreter.set_tensor(input_details["index"], model_input)
        interpreter.invoke()
        pred_raw = interpreter.get_tensor(output_details["index"])

        if is_nhwc_out:
            pred_mask = np.argmax(pred_raw[0], axis=-1).astype(np.uint8)
        else:
            pred_mask = np.argmax(pred_raw[0], axis=0).astype(np.uint8)

        orig_hwc = np.transpose(image_chw, (1, 2, 0))
        gt_mask  = mask_gt.numpy().astype(np.uint8)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].imshow(np.clip(orig_hwc, 0, 1))
        axs[0].set_title("Imagen original")
        axs[0].axis("off")

        axs[1].imshow(mask_to_color(gt_mask))
        axs[1].set_title("Máscara GT")
        axs[1].axis("off")

        axs[2].imshow(mask_to_color(pred_mask))
        axs[2].set_title("Predicción TFLite FP16")
        axs[2].axis("off")

        out_file = OUTPUT_DIR / f"segmentation_preview_{idx + 1}.png"
        fig.tight_layout()
        fig.savefig(str(out_file), dpi=150)
        plt.close(fig)

        print(f"  ✓ Preview {idx + 1}/{PREVIEW_SAMPLES}: {out_file}")


# -----------------------------
# Main
# -----------------------------
def main():
    print("=" * 60)
    print(" Pipeline: PyTorch -> ONNX -> SavedModel -> TFLite FP16")
    print("=" * 60)

    try:
        model = load_pytorch_model()
    except Exception as e:
        print(f"✗ Paso 1 falló: {e}")
        return

    try:
        export_to_onnx(model)
    except Exception as e:
        print(f"✗ Paso 2 falló: {e}")
        return

    try:
        convert_onnx_to_savedmodel()
    except Exception as e:
        print(f"✗ Paso 3 falló: {e}")
        return

    try:
        convert_to_tflite_fp16()
    except Exception as e:
        print(f"✗ Paso 4 falló: {e}")
        return

    val_dataset = build_val_dataset()

    try:
        run_previews(val_dataset)
    except Exception as e:
        print(f"✗ Paso 5 falló: {e}")
        return

    print()
    print("=" * 60)
    print("✅ Pipeline completado.")
    print(f"   Modelo TFLite FP16: {TFLITE_PATH}")
    print(f"   Previews: {OUTPUT_DIR}/segmentation_preview_*.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
