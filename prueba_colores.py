from PIL import Image
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from skimage import color
from pathlib import Path
import argparse

def extract_art_colors(image_path, k=6):
    img = Image.open(image_path).convert("RGB")
    pixels = np.array(img).reshape(-1, 3) / 255.0
    
    # Convertir a LAB — agrupa colores como los percibe el ojo humano
    lab_pixels = color.rgb2lab(pixels.reshape(-1, 1, 3)).reshape(-1, 3)
    
    idx = np.random.choice(len(lab_pixels), size=5000, replace=False)
    sample = lab_pixels[idx]
    
    kmeans = MiniBatchKMeans(n_clusters=k, n_init=3)
    kmeans.fit(sample)
    
    # Convertir centroides de vuelta a RGB
    centers_lab = kmeans.cluster_centers_.reshape(-1, 1, 3)
    centers_rgb = (color.lab2rgb(centers_lab).reshape(-1, 3) * 255).astype(int)
    
    # Calcular peso de cada color (% del lienzo)
    labels = kmeans.predict(lab_pixels)
    counts = np.bincount(labels, minlength=k)
    percentages = (counts / len(labels) * 100).round(1)
    
    return list(zip(centers_rgb.tolist(), percentages))

def main():
    parser = argparse.ArgumentParser(description="Extract dominant colors from an image.")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("-k", type=int, default=6, help="Number of dominant colors")
    parser.add_argument("--max-side", type=int, default=1024, help="Resize so the longest side equals this value while preserving ratio (0 disables resize)")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.is_file():
        print(f"Error: image file not found: {image_path}")
        return 1

    colors = extract_art_colors(image_path, k=args.k)
    print(colors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
                