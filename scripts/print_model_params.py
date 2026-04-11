#!/usr/bin/env python3
"""Imprime la cantidad de parámetros de un modelo guardado en .pth."""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.unet_mobilenet import UNetMobileNet


def extract_state_dict(checkpoint):
    """Extrae el state_dict desde distintos formatos de checkpoint."""
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
    return checkpoint


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_parser():
    parser = argparse.ArgumentParser(
        description="Carga un .pth y muestra la cantidad de parámetros del modelo"
    )
    parser.add_argument(
        "weights",
        type=Path,
        help="Ruta al archivo .pth guardado durante training",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=5,
        help="Cantidad de clases configuradas para el modelo (default: 5)",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Inicializa backbone con pesos preentrenados",
    )
    return parser


def main():
    args = build_parser().parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {args.weights}")

    checkpoint = torch.load(args.weights, map_location="cpu")
    state_dict = extract_state_dict(checkpoint)

    model = UNetMobileNet(num_classes=args.num_classes, pretrained=args.pretrained)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    total, trainable = count_parameters(model)

    print(f"Archivo: {args.weights}")
    print(f"Parámetros totales: {total:,}")
    print(f"Parámetros entrenables: {trainable:,}")

    if missing:
        print(f"[Aviso] Missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"[Aviso] Unexpected keys ({len(unexpected)}): {unexpected[:5]}")


if __name__ == "__main__":
    main()
