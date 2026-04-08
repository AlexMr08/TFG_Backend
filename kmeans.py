from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
from pathlib import Path
import argparse

def resize_keep_ratio(image, max_side=1024):
    width, height = image.size
    longest_side = max(width, height)

    if max_side <= 0 or longest_side <= max_side:
        return image

    scale = max_side / float(longest_side)
    new_size = (int(width * scale), int(height * scale))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def get_dominant_colors(image, k=5, max_side=1024):
    try:
        image = resize_keep_ratio(image, max_side=max_side)
        img_array = np.array(image)
        if img_array.shape[2] == 4: img_array = img_array[:, :, :3]
        img_array = img_array.reshape((img_array.shape[0] * img_array.shape[1], 3))
        
        clt = KMeans(n_clusters=k, n_init='auto', random_state=42)
        clt.fit(img_array)
        
        colors_hex = []
        for rgb in clt.cluster_centers_:
            r, g, b = rgb.astype(int)
            colors_hex.append(f"#{r:02x}{g:02x}{b:02x}")
        return colors_hex
    except Exception:
        return ["#000000"]

def main():
    parser = argparse.ArgumentParser(description="Extract dominant colors from an image.")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("-k", type=int, default=5, help="Number of dominant colors")
    parser.add_argument("--max-side", type=int, default=1024, help="Resize so the longest side equals this value while preserving ratio (0 disables resize)")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.is_file():
        print(f"Error: image file not found: {image_path}")
        return 1

    image = Image.open(image_path).convert("RGB")
    colors = get_dominant_colors(image, k=args.k, max_side=args.max_side)
    print(colors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())