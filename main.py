"""
main.py
-------
Prueba la extraccion de colores y su enriquecimiento.

Uso:
    python main.py <imagen> [--colors 5] [--sample 8000]

Dependencias:
    pip install pillow scikit-learn scikit-image numpy
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import color as skcolor
from sklearn.cluster import MiniBatchKMeans

from art_color_analyzer import enrich_colors, ColorInfo


# ─────────────────────────────────────────────
# Extraccion de colores (tu pipeline)
# ─────────────────────────────────────────────

def extract_dominant_colors(
    image_path: str | Path,
    n_colors: int = 10,
    sample_size: int = 8_000,
    max_dimension: int = 1000,
) -> list[tuple[tuple[int, int, int], float]]:
    """
    Devuelve una lista de ((R, G, B), porcentaje) ordenada desc. por porcentaje.
    """
    img = Image.open(image_path).convert("RGB")

    w, h = img.size
    if max(w, h) > max_dimension:
        scale = max_dimension / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    pixels_rgb = np.array(img).reshape(-1, 3) / 255.0
    pixels_lab = skcolor.rgb2lab(pixels_rgb.reshape(-1, 1, 3)).reshape(-1, 3)

    n_sample = min(sample_size, len(pixels_lab))
    idx = np.random.default_rng(42).choice(len(pixels_lab), size=n_sample, replace=False)
    
    unique_colors = np.unique(pixels_rgb, axis=0)
    n_clusters = min(n_colors, len(unique_colors))

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, n_init=5, random_state=42)
    kmeans.fit(pixels_lab[idx])

    all_labels = kmeans.predict(pixels_lab)
    counts = np.bincount(all_labels, minlength=n_clusters)
    percentages = counts / len(all_labels) * 100

    centers_lab = kmeans.cluster_centers_.reshape(-1, 1, 3)
    centers_rgb = (skcolor.lab2rgb(centers_lab).reshape(-1, 3) * 255).clip(0, 255).astype(int)

    order = np.argsort(percentages)[::-1]
    return [
        ((int(centers_rgb[i][0]), int(centers_rgb[i][1]), int(centers_rgb[i][2])),
         round(float(percentages[i]), 1))
        for i in order
    ]


# ─────────────────────────────────────────────
# Presentacion en consola
# ─────────────────────────────────────────────

TEMP_LABEL = {"warm": "cálido", "cool": "frío", "neutral": "neutro"}
ROLE_LABEL = {"dominant": "dominante", "secondary": "secundario", "accent": "acento"}
RESET = "\033[0m"

def _ansi_block(r: int, g: int, b: int) -> str:
    return f"\033[48;2;{r};{g};{b}m   {RESET}"

def _bar(percentage: float, width: int = 30) -> str:
    filled = round(percentage / 100 * width)
    return "█" * filled + "░" * (width - filled)

def print_results(image_path: str, raw_colors: list, enriched: list[ColorInfo]) -> None:
    print(f"\n  Imagen : {image_path}")
    print(f"  Colores: {len(enriched)}\n")
    print(f"  {'#':<4} {'Bloque':<6} {'HEX':<10} {'RGB':<18} {'Rol':<12} {'Temp':<10} {'Sat':>5} {'Bril':>5} {'% lienzo':>9}")
    print(f"  {'─'*4} {'─'*6} {'─'*10} {'─'*18} {'─'*12} {'─'*10} {'─'*5} {'─'*5} {'─'*9}")

    for i, (c, (rgb, pct)) in enumerate(zip(enriched, raw_colors)):
        block = _ansi_block(*rgb)
        bar   = _bar(pct)
        print(
            f"  {i+1:<4} {block}  {c.hex:<10} "
            f"({c.rgb[0]:>3},{c.rgb[1]:>3},{c.rgb[2]:>3})   "
            f"{ROLE_LABEL[c.harmony_role]:<12} "
            f"{TEMP_LABEL[c.temperature]:<10} "
            f"{c.saturation:>4}% "
            f"{c.brightness:>4}% "
            f"  {pct:>5.1f}%"
        )
        print(f"  {'':>30} {bar} {pct:.1f}%\n" if i < len(enriched) - 1 else "")

    print()


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Extrae y enriquece colores dominantes de una imagen.")
    parser.add_argument("image", help="Ruta a la imagen (JPEG, PNG, WEBP...)")
    parser.add_argument("--colors",  type=int, default=10,     metavar="N", help="Numero de colores (default: 5)")
    parser.add_argument("--sample",  type=int, default=8_000, metavar="N", help="Pixeles a muestrear (default: 8000)")
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"Error: no se encontro '{args.image}'", file=sys.stderr)
        sys.exit(1)

    print("\nExtrayendo colores...", end=" ", flush=True)
    raw_colors = extract_dominant_colors(args.image, n_colors=args.colors, sample_size=args.sample)
    print("hecho.")

    print("Enriqueciendo metadata...", end=" ", flush=True)
    enriched = enrich_colors(raw_colors)
    print("hecho.")
    
    print(enriched)
    # print_results(args.image, raw_colors, enriched)


if __name__ == "__main__":
    main()
