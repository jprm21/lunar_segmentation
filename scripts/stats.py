"""
analyze_onnx.py
Analiza GFLOPs, MACs, parámetros y tamaño de archivos ONNX.

Uso:
    python analyze_onnx.py --input modelos/

Dependencias:
    pip install onnx onnx-tool numpy tabulate
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import onnx


def check_dependencies():
    missing = []
    for pkg in ["onnx", "onnx_tool", "tabulate"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[ERROR] Dependencias faltantes: {', '.join(missing)}")
        print(f"        Instalar con: pip install {' '.join(missing)}")
        sys.exit(1)


def get_input_shape(onnx_path: str) -> tuple:
    """Extrae el shape del input desde el grafo ONNX."""
    model = onnx.load(onnx_path)
    input_info = model.graph.input[0]
    shape = []
    for dim in input_info.type.tensor_type.shape.dim:
        shape.append(dim.dim_value if dim.dim_value > 0 else 1)
    return tuple(shape)


def analyze_model(onnx_path: str) -> dict:
    """Analiza un modelo ONNX y retorna sus estadísticas."""
    import onnx_tool

    path = Path(onnx_path)
    file_size_mb = path.stat().st_size / (1024 * 1024)

    # Shape del input
    input_shape = get_input_shape(onnx_path)
    resolution = input_shape[2] if len(input_shape) >= 3 else "?"

    # Perfil con onnx_tool
    # Redirigir stdout temporalmente para capturar salida de onnx_tool
    import io
    from contextlib import redirect_stdout

    f = io.StringIO()
    with redirect_stdout(f):
        try:
            model_proto = onnx_tool.model_profile(
                onnx_path,
                savenode=None,
                saveshapesmodel=None,
            )
        except TypeError:
            # Algunas versiones no aceptan kwargs
            model_proto = onnx_tool.model_profile(onnx_path)

    # onnx_tool retorna un objeto con atributos de resumen
    # Intentar extraer MACs y parámetros del objeto retornado
    macs = None
    params = None

    if model_proto is not None:
        if hasattr(model_proto, 'macs'):
            macs = model_proto.macs
        if hasattr(model_proto, 'params'):
            params = model_proto.params

    # Fallback: parsear la salida de texto si el objeto no tiene atributos directos
    if macs is None or params is None:
        output_text = f.getvalue()
        for line in output_text.splitlines():
            line_lower = line.lower()
            if 'total' in line_lower or 'sum' in line_lower:
                parts = line.split()
                for i, p in enumerate(parts):
                    try:
                        val = float(p.replace(',', ''))
                        if macs is None and val > 1e6:
                            macs = val
                        elif params is None and val > 1e3:
                            params = val
                    except ValueError:
                        continue

    # Convertir a unidades legibles
    gflops = (2 * macs / 1e9) if macs is not None else None  # MACs -> FLOPs (*2)
    gmacs  = (macs / 1e9)     if macs is not None else None
    params_m = (params / 1e6) if params is not None else None

    return {
        "archivo":    path.name,
        "resolucion": f"{resolution}px",
        "size_mb":    round(file_size_mb, 1),
        "params_m":   round(params_m, 2) if params_m is not None else "N/A",
        "gmacs":      round(gmacs, 2)    if gmacs   is not None else "N/A",
        "gflops":     round(gflops, 2)   if gflops  is not None else "N/A",
        "input_shape": str(input_shape),
    }


def analyze_with_onnxruntime(onnx_path: str) -> dict:
    """
    Alternativa usando onnxruntime para verificar que el modelo corre
    y medir un warmup básico. No mide GFLOPs pero confirma compatibilidad.
    """
    try:
        import onnxruntime as ort
        import time

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        input_info = session.get_inputs()[0]
        shape = list(input_info.shape)
        # Reemplazar dimensiones dinámicas con 1
        shape = [s if isinstance(s, int) and s > 0 else 1 for s in shape]

        dummy = np.random.rand(*shape).astype(np.float32)

        # Warmup
        for _ in range(3):
            session.run(None, {input_info.name: dummy})

        # Medición rápida (10 runs)
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            session.run(None, {input_info.name: dummy})
            times.append((time.perf_counter() - t0) * 1000)

        return {
            "ort_compatible": True,
            "warmup_ms_mean": round(np.mean(times), 1),
            "warmup_ms_std":  round(np.std(times), 1),
        }
    except Exception as e:
        return {
            "ort_compatible": False,
            "warmup_ms_mean": f"ERROR: {e}",
            "warmup_ms_std":  "",
        }


def print_results(results: list, ort_results: list):
    from tabulate import tabulate

    # Tabla principal
    headers_main = [
        "Archivo", "Resolución", "Tamaño (MB)",
        "Params (M)", "GMACs", "GFLOPs", "Input Shape"
    ]
    rows_main = [
        [
            r["archivo"], r["resolucion"], r["size_mb"],
            r["params_m"], r["gmacs"], r["gflops"], r["input_shape"]
        ]
        for r in results
    ]

    print("\n" + "=" * 70)
    print(" ANÁLISIS DE MODELOS ONNX — Estadísticas estáticas")
    print("=" * 70)
    print(tabulate(rows_main, headers=headers_main, tablefmt="github"))

    # Tabla ORT
    headers_ort = [
        "Archivo", "ORT Compatible", "Latencia media (ms)", "Std (ms)"
    ]
    rows_ort = [
        [
            results[i]["archivo"],
            "✓" if o["ort_compatible"] else "✗",
            o["warmup_ms_mean"],
            o["warmup_ms_std"],
        ]
        for i, o in enumerate(ort_results)
    ]

    print("\n" + "=" * 70)
    print(" VERIFICACIÓN ONNX RUNTIME (CPU, 10 runs warmup en host)")
    print(" NOTA: estos tiempos son del host, NO de la Jetson")
    print("=" * 70)
    print(tabulate(rows_ort, headers=headers_ort, tablefmt="github"))

    # Notas de requerimientos
    print("\n" + "=" * 70)
    print(" VERIFICACIÓN PRELIMINAR DE REQUERIMIENTOS")
    print("=" * 70)
    for r in results:
        nombre = r["archivo"]
        size   = r["size_mb"]
        params = r["params_m"]

        hw_req2 = "✓" if isinstance(size, float) and size <= 100 else "✗"
        sw_req2 = "✓" if isinstance(params, float) and params < 20 else "✗"

        print(f"  {nombre}")
        print(f"    HW-REQ-002 (≤100MB FP32):     {hw_req2}  {size} MB")
        print(f"    SW-REQ-002 (<20M parámetros): {sw_req2}  {params} M")
        print()


def save_csv(results: list, ort_results: list, output_path: Path):
    import csv
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "archivo", "resolucion", "size_mb", "params_m",
            "gmacs", "gflops", "input_shape",
            "ort_compatible", "host_latencia_ms_mean", "host_latencia_ms_std"
        ])
        for r, o in zip(results, ort_results):
            writer.writerow([
                r["archivo"], r["resolucion"], r["size_mb"],
                r["params_m"], r["gmacs"], r["gflops"], r["input_shape"],
                o["ort_compatible"], o["warmup_ms_mean"], o["warmup_ms_std"]
            ])
    print(f"\n  CSV guardado en: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analiza modelos ONNX: GFLOPs, parámetros, tamaño y compatibilidad ORT."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Carpeta con archivos .onnx"
    )
    parser.add_argument(
        "--csv", "-o",
        default=None,
        help="Ruta opcional para guardar resultados en CSV (ej: resultados.csv)"
    )
    args = parser.parse_args()

    check_dependencies()

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"[ERROR] Carpeta no encontrada: {input_dir}")
        sys.exit(1)

    onnx_files = sorted(input_dir.glob("*.onnx"))
    if not onnx_files:
        print(f"[ERROR] No se encontraron archivos .onnx en: {input_dir}")
        sys.exit(1)

    print(f"\nEncontrados {len(onnx_files)} modelos ONNX en '{input_dir}':")
    for f in onnx_files:
        print(f"  - {f.name}")

    results     = []
    ort_results = []

    for onnx_path in onnx_files:
        print(f"\n[Analizando] {onnx_path.name} ...")

        try:
            stats = analyze_model(str(onnx_path))
        except Exception as e:
            print(f"  [WARN] onnx_tool falló para {onnx_path.name}: {e}")
            path = Path(str(onnx_path))
            stats = {
                "archivo":     path.name,
                "resolucion":  "?",
                "size_mb":     round(path.stat().st_size / (1024 * 1024), 1),
                "params_m":    "N/A",
                "gmacs":       "N/A",
                "gflops":      "N/A",
                "input_shape": "N/A",
            }
            try:
                shape = get_input_shape(str(onnx_path))
                stats["resolucion"]  = f"{shape[2]}px" if len(shape) >= 3 else "?"
                stats["input_shape"] = str(shape)
            except Exception:
                pass

        results.append(stats)

        ort_stats = analyze_with_onnxruntime(str(onnx_path))
        ort_results.append(ort_stats)

    print_results(results, ort_results)

    if args.csv:
        save_csv(results, ort_results, Path(args.csv))


if __name__ == "__main__":
    main()
