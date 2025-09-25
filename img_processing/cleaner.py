import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def remove_small_components_soft(binary_img, min_area=15, debug=False):
    """
    Removes small connected components (noise) while keeping meaningful text.
    """
    binary = np.where(binary_img < 128, 0, 255).astype(np.uint8)
    inverted = 255 - binary
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)

    mask = np.zeros_like(binary_img)
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
    """
    Displays original and cleaned images side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title("Original")
    axes[1].imshow(cleaned, cmap='gray')
    axes[1].set_title("Cleaned")
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def clean_output_lines(input_dir, output_dir, min_area=15, preview=False):
    """
    Processes all PNG images in input_dir by removing small components and saves the result to output_dir.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(input_dir.glob("*.png"))
    for path in image_paths:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to read image: {path}")
            continue

        cleaned = remove_small_components_soft(img, min_area=min_area, debug=True)

        if preview:
            show_comparison(img, cleaned)

        output_path = output_dir / path.name
        cv2.imwrite(str(output_path), cleaned)

    print(f"✔ Cleaning complete: {len(image_paths)} images processed.")
