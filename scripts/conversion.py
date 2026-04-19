"""
conversion.py
Pipeline: PyTorch -> ONNX -> SavedModel (tf2onnx) -> TFLite INT8
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
    0: (187,  70, 156),  # regolith -> morado rosado
    1: (120,   0, 200),  # crater   -> morado
    2: (232, 250,  80),  # rock     -> amarillo
    3: (173,  69,  31),  # mountain -> café
    4: ( 34, 201, 248),  # sky      -> celeste
}


# -----------------------------
# Utilidades
# -----------------------------
def normalize_chw(image_chw: np.ndarray) -> np.ndarray:
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
        raise FileNotFoundError(f"No existe: {MODEL_PATH}")

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

    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)
    print(f"  ✓ ONNX guardado en: {ONNX_PATH}")

    # Simplificar
    try:
        from onnxsim import simplify
        simplified, ok = simplify(onnx_model)
        if ok:
            onnx.save(simplified, str(ONNX_SIM_PATH))
            print(f"  ✓ ONNX simplificado guardado en: {ONNX_SIM_PATH}")
        else:
            print("  ⚠ Simplificación no exitosa, usando ONNX original")
            import shutil
            shutil.copy(str(ONNX_PATH), str(ONNX_SIM_PATH))
    except Exception as e:
        print(f"  ⚠ onnxsim no disponible ({e}), usando ONNX original")
        import shutil
        shutil.copy(str(ONNX_PATH), str(ONNX_SIM_PATH))


# -----------------------------
# Paso 3: ONNX -> SavedModel con onnx y tf
# -----------------------------
def convert_onnx_to_savedmodel() -> None:
    print("[3/5] Convirtiendo ONNX -> TF Function -> SavedModel...")

    TF_PATH.mkdir(parents=True, exist_ok=True)

    # Usamos onnx-tf via import directo si está disponible,
    # si no, usamos el wrapper manual con tf.
    # Estrategia: cargar ONNX y reconstruir como TF SavedModel
    # usando onnxruntime + tf para inferencia delegada.

    # Primero intentamos con onnx-tf directamente (puede funcionar
    # si tensorflow-probability está instalado)
    try:
        from onnx_tf.backend import prepare as onnx_tf_prepare
        onnx_model = onnx.load(str(ONNX_SIM_PATH))
        tf_rep = onnx_tf_prepare(onnx_model)
        tf_rep.export_graph(str(TF_PATH))
        print(f"  ✓ SavedModel guardado en: {TF_PATH} (via onnx-tf)")
        return
    except Exception as e:
        print(f"  ⚠ onnx-tf no disponible ({type(e).__name__}), usando onnxruntime wrapper...")

    # Alternativa: crear SavedModel que delega a onnxruntime
    try:
        import onnxruntime as ort

        onnx_path_str = str(ONNX_SIM_PATH)
        sess = ort.InferenceSession(onnx_path_str, providers=["CPUExecutionProvider"])
        input_name  = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        @tf.function(input_signature=[
            tf.TensorSpec(shape=[1, 3, IMAGE_SIZE, IMAGE_SIZE], dtype=tf.float32, name="input")
        ])
        def serving_fn(input_tensor):
            # Llamada a onnxruntime desde dentro de tf.function via py_function
            def _run(x):
                result = sess.run(
                    [output_name],
                    {input_name: x.numpy()}
                )[0]
                return result

            output = tf.py_function(_run, [input_tensor], tf.float32)
            output.set_shape([1, NUM_CLASSES, IMAGE_SIZE, IMAGE_SIZE])
            return output

        tf.saved_model.save(
            obj=serving_fn,
            export_dir=str(TF_PATH),
            signatures={"serving_default": serving_fn},
        )
        print(f"  ✓ SavedModel (onnxruntime wrapper) guardado en: {TF_PATH}")

    except ImportError:
        raise RuntimeError(
            "Ni onnx-tf ni onnxruntime están disponibles.\n"
            "Instalá uno de los dos:\n"
            "  pip install onnxruntime\n"
            "  pip install onnx-tf tensorflow-probability"
        )


# -----------------------------
# Paso 4: SavedModel -> TFLite INT8
# -----------------------------
def convert_to_tflite_int8(dataset: LuSNARDataset) -> None:
    print("[4/5] Cuantizando SavedModel -> TFLite INT8...")

    # Encontrar el directorio correcto del SavedModel
    pb_candidates = list(TF_PATH.rglob("saved_model.pb"))
    if not pb_candidates:
        raise FileNotFoundError(f"No se encontró saved_model.pb en {TF_PATH}")
    savedmodel_dir = str(pb_candidates[0].parent)

    # Detectar layout del modelo
    loaded = tf.saved_model.load(savedmodel_dir)
    sig_key = list(loaded.signatures.keys())[0]
    serving_fn = loaded.signatures[sig_key]
    input_tensor = list(serving_fn.structured_input_signature[1].values())[0]
    input_shape = [
        int(d) if (d is not None and d != -1) else -1
        for d in input_tensor.shape
    ]
    print(f"  Input shape detectado: {input_shape}")

    is_nhwc = len(input_shape) == 4 and input_shape[-1] == 3

    def representative_dataset_gen():
        limit = min(CALIBRATION_SAMPLES, len(dataset))
        for idx in range(limit):
            image, _ = dataset[idx]
            image_np = normalize_chw(image.numpy().astype(np.float32))

            if is_nhwc:
                image_np = np.transpose(image_np, (1, 2, 0))

            yield [np.expand_dims(image_np, axis=0).astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    TFLITE_PATH.write_bytes(tflite_model)

    size_mb = TFLITE_PATH.stat().st_size / 1024 / 1024
    print(f"  ✓ TFLite INT8 guardado en: {TFLITE_PATH}")
    print(f"  Tamaño: {size_mb:.1f} MB")


# -----------------------------
# Paso 5: Inferencia y previews
# -----------------------------
def run_previews(dataset: LuSNARDataset) -> None:
    print("[5/5] Ejecutando inferencia TFLite INT8 y guardando visualizaciones...")

    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    in_scale,  in_zero  = input_details["quantization"]
    out_scale, out_zero = output_details["quantization"]

    in_shape  = input_details["shape"].tolist()
    out_shape = output_details["shape"].tolist()

    is_nhwc_in  = len(in_shape)  == 4 and in_shape[-1]  == 3
    is_nhwc_out = len(out_shape) == 4 and out_shape[-1] == NUM_CLASSES

    print(f"  Input:  {in_shape}  ({'NHWC' if is_nhwc_in  else 'NCHW'})")
    print(f"  Output: {out_shape} ({'NHWC' if is_nhwc_out else 'NCHW'})")

    for idx in range(min(PREVIEW_SAMPLES, len(dataset))):
        image, mask_gt = dataset[idx]

        image_chw  = image.numpy().astype(np.float32)
        image_norm = normalize_chw(image_chw)

        if is_nhwc_in:
            model_input = np.transpose(image_norm, (1, 2, 0))
        else:
            model_input = image_norm

        model_input = np.expand_dims(model_input, axis=0).astype(np.float32)

        # Cuantizar a INT8
        if in_scale != 0:
            quant_input = np.round(model_input / in_scale + in_zero)
            quant_input = np.clip(quant_input, -128, 127).astype(np.int8)
        else:
            quant_input = model_input.astype(np.int8)

        # Inferencia
        interpreter.set_tensor(input_details["index"], quant_input)
        interpreter.invoke()
        pred_raw = interpreter.get_tensor(output_details["index"])

        # Dequantizar
        if out_scale != 0:
            pred_float = (pred_raw.astype(np.float32) - out_zero) * out_scale
        else:
            pred_float = pred_raw.astype(np.float32)

        # Argmax
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
        fig.tight_layout()
        fig.savefig(str(out_file), dpi=150)
        plt.close(fig)
        print(f"  ✓ Preview {idx + 1}/{PREVIEW_SAMPLES}: {out_file}")


# -----------------------------
# Main
# -----------------------------
def main():
    print("=" * 60)
    print(" Pipeline: PyTorch -> ONNX -> SavedModel -> TFLite INT8")
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

    val_dataset = build_val_dataset()

    try:
        convert_to_tflite_int8(val_dataset)
    except Exception as e:
        print(f"✗ Paso 4 falló: {e}")
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
