import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def _ensure_binary_u8(img_gray):
    """
    Make sure the image is strictly binary {0,255}. If not, apply Otsu.
    Returns uint8 with text=0, background=255.
    """
    if img_gray.dtype != np.uint8:
        img_gray = img_gray.astype(np.uint8)
    u = np.unique(img_gray)
    if u.size <= 4 and set(u.tolist()).issubset({0, 255}):
        return (img_gray > 127).astype(np.uint8) * 255
    _, th = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def _remove_border_touching_components(binary_img):
    """
    Remove connected components that touch the image border.
    Assumes text=0, background=255.
    """
    inv = 255 - binary_img  # text -> 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    h, w = binary_img.shape[:2]
    keep = np.zeros_like(inv, dtype=np.uint8)

    for i in range(1, num_labels):
        x, y, bw, bh = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3]
        touches = (x == 0) or (y == 0) or (x + bw >= w) or (y + bh >= h)
        if not touches:
            keep[labels == i] = 255
    return 255 - keep  # back to text=0

def _trim_white_margins(binary_img):
    """
    Crop away fully-white margins. Returns possibly smaller image.
    Assumes text=0, background=255.
    """
    ys, xs = np.where(binary_img < 128)
    if xs.size == 0 or ys.size == 0:
        return binary_img
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return binary_img[y0:y1, x0:x1]

def remove_small_components_soft(binary_img, min_area=15, debug=False):
    """
    Removes small connected components (noise) while keeping meaningful text.
    Expects text=0, background=255.
    """
    binary = _ensure_binary_u8(binary_img)
    inv = 255 - binary
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)

    mask = np.zeros_like(binary)
    kept = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            mask[labels == i] = 255
            kept += 1

    if debug:
        print(f"Kept components: {kept} out of {num_labels - 1} (min_area={min_area})")

    return 255 - mask

def show_comparison(original, cleaned):
    """Displays original and cleaned images side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(original, cmap='gray'); axes[0].set_title("Original")
    axes[1].imshow(cleaned, cmap='gray');  axes[1].set_title("Cleaned")
    for ax in axes: ax.axis('off')
    plt.tight_layout(); plt.show()

def clean_output_lines(input_dir, output_dir, min_area=15, preview=False,
                       drop_border_touching=True, trim=True):
    """
    Cleans all line images by:
      1) ensuring binary,
      2) removing small components (< min_area),
      3) optionally removing components that touch the crop border (good for PAGE Baseline crops),
      4) optionally trimming white margins,
      5) saving to output_dir (keeps filename, normalizes to PNG extension if desired).
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # handle multiple extensions (PAGE/ALTO pipelines may output tif/jpg too)
    exts = ("*.png", "*.tif", "*.tiff", "*.jpg", "*.jpeg")
    image_paths = []
    for pat in exts:
        image_paths.extend(input_dir.glob(pat))
    image_paths = sorted(image_paths)

    for path in image_paths:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to read image: {path}")
            continue

        # 1) binary normalization
        img_bin = _ensure_binary_u8(img)

        # 2) remove small components
        cleaned = remove_small_components_soft(img_bin, min_area=min_area, debug=True)

        # 3) optional: remove border-touching comps (useful for PAGE Baseline ribbon)
        if drop_border_touching:
            cleaned = _remove_border_touching_components(cleaned)

        # 4) optional: trim white margins
        if trim:
            cleaned = _trim_white_margins(cleaned)

        if preview:
            show_comparison(img, cleaned)

        output_path = output_dir / path.name  # keep same name
        cv2.imwrite(str(output_path), cleaned)

    print(f"Cleaning complete: {len(image_paths)} images processed.")
