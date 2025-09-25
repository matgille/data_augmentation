import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.filters import threshold_sauvola

def parse_polygon(polygon_str):
    coords = polygon_str.strip().split()
    return [(int(float(coords[i])), int(float(coords[i + 1]))) for i in range(0, len(coords), 2)]

def adaptive_binarize(image, mask, sauvola_window_size=35, sauvola_k=0.3, debug=False):

    if debug:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        plt.imshow(img_rgb)
        plt.axis('off')  # Hide axis ticks
        plt.title('RGB Image')
        plt.show()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if debug:
        windows = [25,35,55,75]
        ks = [0.1, 0.3, 0.5, 0.7]

        fig, axes = plt.subplots(len(windows), len(ks), figsize=(16, 12))
        fig.suptitle("Sauvola Thresholding: All Window Sizes and k Values", fontsize=16)

        for i, window in enumerate(windows):
            for j, k in enumerate(ks):
                # Sauvola binarization
                sauvola_thresh = threshold_sauvola(gray, window_size=window, k=k)  # ws25 k0,3
                binarized_sauvola = (gray > sauvola_thresh).astype(np.uint8) * 255

                result = np.where(mask == 255, binarized_sauvola, 255).astype(np.uint8)

                # Convert to RGBA for display
                img_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGBA)

                # Plot in the corresponding subplot
                ax = axes[i, j]
                ax.imshow(img_rgb)
                ax.set_title(f"w={window}, k={k}")
                ax.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust to fit suptitle
        plt.show()
    else:
        sauvola_thresh = threshold_sauvola(gray, window_size=sauvola_window_size, k=sauvola_k)
        binarized_sauvola = (gray > sauvola_thresh).astype(np.uint8) * 255
        result = np.where(mask == 255, binarized_sauvola, 255).astype(np.uint8)
    return result

def extract_and_crop_polygon(image, points, args):
    margin_v = args.crop_margin_v
    margin_h = args.crop_margin_h

    h, w = image.shape[:2]
    polygon = [(x, y) for x, y in points if 0 <= x < w and 0 <= y < h]
    if len(polygon) < 3:
        return None

    # Bounding box of the polygon
    x, y, bw, bh = cv2.boundingRect(np.array(polygon))
    x = max(0, x - margin_h)
    y = max(0, y - margin_v)
    bw += 2 * margin_h
    bh += 2 * margin_v

    cropped = image[y:y+bh, x:x+bw].copy()

    # Shift polygon to cropped coordinate space
    shifted_polygon = [(px - x, py - y) for px, py in polygon]

    # Create mask for polygon within cropped region
    mask = np.zeros((bh, bw), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(shifted_polygon, dtype=np.int32)], 255)

    # Erosione più forte per evitare contorno
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.erode(mask, kernel, iterations=1)

    # Binarizzazione solo dell'interno del poligono eroso
    binarized = adaptive_binarize(cropped, mask, args.sauvola_window_size, args.sauvola_k, debug=False)
    return binarized

def process_image(image_path, xml_path, output_dir, args):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Image not found: {image_path}")
        return

    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {'a': 'http://www.loc.gov/standards/alto/ns-v4#'}
    textlines = root.findall(".//a:TextLine", ns)

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, line in enumerate(textlines):
        polygon = line.find(".//a:Polygon", ns)
        if polygon is not None and 'POINTS' in polygon.attrib:
            points = parse_polygon(polygon.attrib['POINTS'])
            cropped = extract_and_crop_polygon(image, points, args)
            if cropped is not None and cropped.size > 0:
                out_path = output_dir / f"{image_path.stem}_line_{i:04}.png"
                cv2.imwrite(str(out_path), cropped)

    print(f"Processed {image_path.name} with {len(textlines)} lines")

def polygon_extraction(args):
    data_folder = Path(args.src)
    output_folder = Path(args.binarized_lines)
    output_folder.mkdir(exist_ok=True)

    extensions = [".jpg", ".jpeg", ".png"]

    for image_path in data_folder.iterdir():
        if image_path.suffix.lower() in extensions and image_path.is_file():
            xml_path = data_folder / f"{image_path.stem}.xml"
            if xml_path.exists():
                process_image(image_path, xml_path, output_folder, args)
            else:
                print(f"Missing XML for: {image_path.name}")
