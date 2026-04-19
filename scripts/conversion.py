"""
conversion.py
Pipeline: PyTorch -> ONNX -> TFLite INT8
Usando onnx2tf en lugar de onnx-tf para evitar dependencias problemáticas.
"""

import subprocess
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
TFLITE_PATH   = OUTPUT_DIR / "model_int8.tflite"

IMAGE_SIZE          = 384
NUM_CLASSES         = 5
VAL_SCENES          = [3, 5, 7]
CALIBRATION_SAMPLES = 100
PREVIEW_SAMPLES     = 5

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Índices: 0=regolith, 1=crater, 2=rock, 3=mountain, 4=sky
MASK_COLOR_MAP = {
    0: (187, 70, 156),  # regolith
    1: (120, 0, 200),  # crater
    2: (232, 250, 80),  # rock 
    3: (173, 69, 31),  # mountain 
    4: (34, 201, 248),  # sky
}

CLASS_NAMES = ["Regolith", "Crater", "Rock", "Mountain", "Sky"]


# -----------------------------
# Utilidades
# -----------------------------
def normalize_chw(image_chw: np.ndarray) -> np.ndarray:
    """Normaliza imagen CHW en [0,1] con mean/std de ImageNet."""
    return (image_chw - MEAN[:, None, None]) / STD[:, None, None]


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
        raise FileNotFoundError(f"No existe el archivo de pesos: {MODEL_PATH}")

    model = UNetMobileNet(num_classes=NUM_CLASSES, pretrained=False)
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
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

    # Verificar
    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)
    print(f"  ✓ ONNX guardado en: {ONNX_PATH}")

    # Simplificar (reduce operadores redundantes, mejora compatibilidad)
    try:
        from onnxsim import simplify
        simplified, ok = simplify(onnx_model)
        if ok:
            onnx.save(simplified, str(ONNX_SIM_PATH))
            print(f"  ✓ ONNX simplificado guardado en: {ONNX_SIM_PATH}")
        else:
            print("  ⚠ Simplificación falló, usando ONNX original")
            ONNX_SIM_PATH.write_bytes(ONNX_PATH.read_bytes())
    except Exception as e:
        print(f"  ⚠ onnx-simplifier no disponible ({e}), usando ONNX original")
        ONNX_SIM_PATH.write_bytes(ONNX_PATH.read_bytes())


# -----------------------------
# Paso 3: ONNX -> TFLite INT8 con onnx2tf
# -----------------------------
def convert_to_tflite(dataset: LuSNARDataset) -> None:
    print("[3/5] Generando datos de calibración...")

    # Guardamos imágenes de calibración como npz para pasarlas a onnx2tf
    calib_dir = OUTPUT_DIR / "calibration_data"
    calib_dir.mkdir(parents=True, exist_ok=True)

    limit = min(CALIBRATION_SAMPLES, len(dataset))
    for idx in range(limit):
        image, _ = dataset[idx]
        image_np = image.numpy().astype(np.float32)
        image_np = normalize_chw(image_np)
        # onnx2tf espera NHWC para calibración
        image_nhwc = np.transpose(image_np, (1, 2, 0))
        np.save(str(calib_dir / f"calib_{idx:04d}.npy"), image_nhwc)

    print(f"  ✓ {limit} imágenes de calibración guardadas en: {calib_dir}")

    print("[4/5] Convirtiendo ONNX -> TFLite INT8 con onnx2tf...")

    cmd = [
        "onnx2tf",
        "-i",    str(ONNX_SIM_PATH),
        "-o",    str(TF_PATH),
        "-oiqt",                        # genera INT8 quantized tflite
        "-cind", "input",               # nombre del tensor de input
                 str(calib_dir),        # directorio con datos de calibración
                 "[[[[0.485,0.456,0.406]]]]",  # mean para normalización interna
                 "[[[[0.229,0.224,0.225]]]]",  # std  para normalización interna
        "--non_verbose",
    ]

    print(f"  Ejecutando: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("  ✗ onnx2tf falló con el siguiente error:")
        print(result.stderr)
        raise RuntimeError("Conversión onnx2tf falló.")

    # onnx2tf genera el tflite dentro de TF_PATH con nombre automático
    # buscamos el archivo INT8
    tflite_candidates = list(TF_PATH.glob("*int8*.tflite"))
    if not tflite_candidates:
        tflite_candidates = list(TF_PATH.glob("*.tflite"))

    if not tflite_candidates:
        raise FileNotFoundError(
            f"No se encontró ningún archivo .tflite en {TF_PATH}. "
            "Revisá la salida de onnx2tf."
        )

    # Copiamos al path esperado
    import shutil
    shutil.copy(str(tflite_candidates[0]), str(TFLITE_PATH))
    print(f"  ✓ TFLite INT8 guardado en: {TFLITE_PATH}")


# -----------------------------
# Paso 4 (fallback): Cuantización manual si onnx2tf no genera INT8
# -----------------------------
def convert_savedmodel_to_tflite_int8(dataset: LuSNARDataset) -> None:
    """
    Fallback: usa tf.lite.TFLiteConverter directamente sobre el SavedModel
    generado por onnx2tf si el flag -oiqt no funcionó.
    """
    print("[4b] Fallback: cuantización manual con TFLiteConverter...")

    def representative_dataset_gen():
        limit = min(CALIBRATION_SAMPLES, len(dataset))
        for idx in range(limit):
            image, _ = dataset[idx]
            image_np = image.numpy().astype(np.float32)
            image_np = normalize_chw(image_np)
            # TFLiteConverter espera NHWC
            image_nhwc = np.transpose(image_np, (1, 2, 0))
            batch = np.expand_dims(image_nhwc, axis=0)
            yield [batch]

    converter = tf.lite.TFLiteConverter.from_saved_model(str(TF_PATH))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    TFLITE_PATH.write_bytes(tflite_model)
    print(f"  ✓ TFLite INT8 (fallback) guardado en: {TFLITE_PATH}")


# -----------------------------
# Paso 5: Inferencia y visualizaciones
# -----------------------------
def run_previews(dataset: LuSNARDataset) -> None:
    print("[5/5] Ejecutando inferencia TFLite INT8 y guardando visualizaciones...")

    if not TFLITE_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo TFLite en: {TFLITE_PATH}")

    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    in_scale,  in_zero  = input_details["quantization"]
    out_scale, out_zero = output_details["quantization"]

    in_shape  = input_details["shape"].tolist()
    out_shape = output_details["shape"].tolist()

    # Detectar si el modelo espera NHWC o NCHW
    is_nhwc_in  = len(in_shape)  == 4 and in_shape[-1]  == 3
    is_nhwc_out = len(out_shape) == 4 and out_shape[-1] == NUM_CLASSES

    print(f"  Input shape:  {in_shape}  ({'NHWC' if is_nhwc_in  else 'NCHW'})")
    print(f"  Output shape: {out_shape} ({'NHWC' if is_nhwc_out else 'NCHW'})")

    for idx in range(min(PREVIEW_SAMPLES, len(dataset))):
        image, mask_gt = dataset[idx]

        # Preprocesar
        image_chw = image.numpy().astype(np.float32)
        image_norm = normalize_chw(image_chw)

        if is_nhwc_in:
            model_input = np.transpose(image_norm, (1, 2, 0))
            model_input = np.expand_dims(model_input, axis=0)   # (1,H,W,3)
        else:
            model_input = np.expand_dims(image_norm, axis=0)    # (1,3,H,W)

        # Cuantizar input a INT8
        if in_scale != 0:
            quant_input = np.round(model_input / in_scale + in_zero)
            quant_input = np.clip(quant_input, -128, 127).astype(np.int8)
        else:
            quant_input = model_input.astype(np.int8)

        # Inferencia
        interpreter.set_tensor(input_details["index"], quant_input)
        interpreter.invoke()
        pred_raw = interpreter.get_tensor(output_details["index"])

        # Dequantizar output
        if out_scale != 0:
            pred_float = (pred_raw.astype(np.float32) - out_zero) * out_scale
        else:
            pred_float = pred_raw.astype(np.float32)

        # Obtener máscara de clase (argmax)
        if is_nhwc_out:
            pred_mask = np.argmax(pred_float[0], axis=-1).astype(np.uint8)
        else:
            pred_mask = np.argmax(pred_float[0], axis=0).astype(np.uint8)

        # Visualización
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
        axs[2].set_title("Predicción TFLite INT8")
        axs[2].axis("off")

        out_file = OUTPUT_DIR / f"segmentation_preview_{idx + 1}.png"
        fig.suptitle(
            f"Escena val - muestra {idx + 1} | "
            f"in_scale={in_scale:.4f} out_scale={out_scale:.4f}",
            fontsize=9,
        )
        fig.tight_layout()
        fig.savefig(str(out_file), dpi=150)
        plt.close(fig)
        print(f"  ✓ Preview {idx + 1}/5 guardado: {out_file}")


# -----------------------------
# Main
# -----------------------------
def main():
    print("=" * 60)
    print(" Pipeline: PyTorch -> ONNX -> TFLite INT8")
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

    val_dataset = build_val_dataset()

    try:
        convert_to_tflite(val_dataset)
    except Exception as e:
        print(f"  ⚠ onnx2tf falló ({e}), intentando fallback con TFLiteConverter...")
        try:
            convert_savedmodel_to_tflite_int8(val_dataset)
        except Exception as e2:
            print(f"✗ Fallback también falló: {e2}")
            print("  El modelo TFLite no pudo generarse.")
            return

    try:
        run_previews(val_dataset)
    except Exception as e:
        print(f"✗ Paso 5 falló: {e}")
        return

    print()
    print("=" * 60)
    print("✅ Pipeline completado.")
    print(f"   Modelo TFLite: {TFLITE_PATH}")
    print(f"   Previews:      {OUTPUT_DIR}/segmentation_preview_*.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
