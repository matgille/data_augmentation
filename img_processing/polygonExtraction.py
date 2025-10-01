import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.filters import threshold_sauvola
import re  # used by robust parse_polygon


# Previous simple version (kept here for reference)
# def parse_polygon(polygon_str):
#     coords = polygon_str.strip().split()
#     return [(int(float(coords[i])), int(float(coords[i + 1]))) for i in range(0, len(coords), 2)]

def _mask_from_baseline(image_shape, baseline_points, thickness=9, dilation=1):
    """
    Build a narrow 'ribbon' mask along the baseline to isolate just the line.
    thickness ~ line height/3; small dilation to include ascenders/descenders.
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(baseline_points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=int(thickness))
    if dilation > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + 2*dilation, 1 + 2*dilation))
        mask = cv2.dilate(mask, k, iterations=1)
    return mask

def extract_from_baseline_ribbon(image_bgr, baseline_points, args):
    """
    Crop using a 'ribbon' mask around the baseline (PAGE fallback when Coords are missing).
    Applies Sauvola only inside the mask; outside -> white.
    """
    h, w = image_bgr.shape[:2]
    thickness = int(getattr(args, "ribbon_thickness", 9))
    dilation  = int(getattr(args, "ribbon_dilation", 1))
    margin_v  = int(getattr(args, "crop_margin_v", 3))
    margin_h  = int(getattr(args, "crop_margin_h", 4))

    line_mask = _mask_from_baseline((h, w), baseline_points, thickness=thickness, dilation=dilation)

    ys, xs = np.where(line_mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None

    x, y, bw, bh = cv2.boundingRect(np.column_stack((xs, ys)))
    x0 = max(0, x - margin_h); y0 = max(0, y - margin_v)
    x1 = min(w, x + bw + margin_h); y1 = min(h, y + bh + margin_v)
    if x1 <= x0 or y1 <= y0:
        return None

    crop = image_bgr[y0:y1, x0:x1].copy()

    # Shift mask to crop space + slight erosion to avoid halo borders
    crop_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    crop_mask[line_mask[y0:y1, x0:x1] > 0] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    crop_mask = cv2.erode(crop_mask, kernel, iterations=1)

    # Binarize only inside mask; outside -> white
    return adaptive_binarize(crop, crop_mask,
                             sauvola_window_size=getattr(args, "sauvola_window_size", 35),
                             sauvola_k=getattr(args, "sauvola_k", 0.2),
                             debug=False)

# PAGE-ready parser: accepts "x,y x,y ..." and "x y x y ..."
def parse_polygon(points_str):
    s = points_str.strip()
    if not s:
        return []
    # Normalize separators (commas/semicolons -> spaces)
    s = s.replace(",", " ").replace(";", " ")
    tokens = re.split(r"\s+", s)
    pts = []
    # Read pairs (x, y)
    for i in range(0, len(tokens) - 1, 2):
        try:
            x = int(float(tokens[i]))
            y = int(float(tokens[i + 1]))
            pts.append((x, y))
        except ValueError:
            # skip malformed pairs and continue
            continue
    return pts

def adaptive_binarize(image, mask, sauvola_window_size=35, sauvola_k=0.3, debug=False):
    """
    Apply Sauvola adaptive thresholding. Only pixels inside `mask` are kept;
    outside pixels are set to white (255).
    If debug=True, show a grid of window/k combinations for visual tuning.
    """
    if debug:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        plt.imshow(img_rgb)
        plt.axis('off')  # hide axes
        plt.title('RGB Image')
        plt.show()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if debug:
        windows = [25, 35, 55, 75]
        ks = [0.1, 0.3, 0.5, 0.7]

        fig, axes = plt.subplots(len(windows), len(ks), figsize=(16, 12))
        fig.suptitle("Sauvola Thresholding: All Window Sizes and k Values", fontsize=16)

        for i, window in enumerate(windows):
            for j, k in enumerate(ks):
                # Sauvola binarization
                sauvola_thresh = threshold_sauvola(gray, window_size=window, k=k)
                binarized_sauvola = (gray > sauvola_thresh).astype(np.uint8) * 255

                result = np.where(mask == 255, binarized_sauvola, 255).astype(np.uint8)

                # Convert to RGBA for display
                img_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGBA)

                # Plot in the corresponding subplot
                ax = axes[i, j]
                ax.imshow(img_rgb)
                ax.set_title(f"w={window}, k={k}")
                ax.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # fit suptitle
        plt.show()
    else:
        sauvola_thresh = threshold_sauvola(gray, window_size=sauvola_window_size, k=sauvola_k)
        binarized_sauvola = (gray > sauvola_thresh).astype(np.uint8) * 255
        result = np.where(mask == 255, binarized_sauvola, 255).astype(np.uint8)
    return result

def extract_polygon_gray(image_bgr, points, args):
    """
    Crop following the exact polygon `points`:
      - compute a tight bounding box + margins,
      - create a polygon mask in the crop space,
      - apply Sauvola ONLY inside the mask,
      - set outside the mask to white (255).
    Returns an 8-bit grayscale image.
    """
    h, w = image_bgr.shape[:2]
    poly = np.array(points, dtype=np.int32)

    # 1) tight bounding box + margins
    x, y, bw, bh = cv2.boundingRect(poly)
    x0 = max(0, x - args.crop_margin_h)
    y0 = max(0, y - args.crop_margin_v)
    x1 = min(w, x + bw + args.crop_margin_h)
    y1 = min(h, y + bh + args.crop_margin_v)
    if x1 <= x0 or y1 <= y0:
        return None

    crop = image_bgr[y0:y1, x0:x1].copy()
    poly_shift = poly - np.array([x0, y0], dtype=np.int32)

    # 2) polygon mask in crop space
    mask = np.zeros((crop.shape[0], crop.shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask, [poly_shift], 255)

    # 3) Sauvola ONLY inside the mask
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sau = threshold_sauvola(gray, window_size=args.sauvola_window_size, k=args.sauvola_k)
    bina = (gray > sau).astype(np.uint8) * 255

    # 4) outside the mask -> white
    out = np.where(mask == 255, bina, 255).astype(np.uint8)
    return out


def extract_and_crop_polygon(image, points, args):
    """
    Legacy polygon-based crop (kept for compatibility).
    Uses a polygon mask inside a rectangular crop and applies Sauvola inside.
    """
    margin_v = args.crop_margin_v
    margin_h = args.crop_margin_h

    h, w = image.shape[:2]

    # Do not filter out-of-bounds points here; keep original vertices
    polygon = [(int(x), int(y)) for x, y in points]
    if len(polygon) < 3:
        return None

    # Bounding box may extend beyond image origin; clamp later
    x, y, bw, bh = cv2.boundingRect(np.array(polygon, dtype=np.int32))
    x0 = max(0, x - margin_h)
    y0 = max(0, y - margin_v)
    bw = min(w - x0, bw + 2 * margin_h)
    bh = min(h - y0, bh + 2 * margin_v)
    if bw <= 0 or bh <= 0:
        return None

    cropped = image[y0:y0+bh, x0:x0+bw].copy()

    # Shift polygon into crop space and clip to mask bounds
    shifted_polygon = np.array([(px - x0, py - y0) for px, py in polygon], dtype=np.int32)
    shifted_polygon[:, 0] = np.clip(shifted_polygon[:, 0], 0, bw - 1)
    shifted_polygon[:, 1] = np.clip(shifted_polygon[:, 1], 0, bh - 1)

    # Build mask and slightly erode to remove border artifacts
    mask = np.zeros((bh, bw), dtype=np.uint8)
    cv2.fillPoly(mask, [shifted_polygon], 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.erode(mask, kernel, iterations=1)

    # Apply Sauvola only inside the eroded polygon mask
    binarized = adaptive_binarize(cropped, mask, args.sauvola_window_size, args.sauvola_k, debug=False)
    return binarized

# ================
# PAGE / ALTO support
# ================
def _localname(tag):
    """Return the localname of an XML tag (strip namespace)."""
    return tag.split('}')[-1] if '}' in tag else tag

def _get_xml_flavour(root):
    """
    Detect whether the XML is ALTO or PAGE by namespace/root tag and typical elements.
    """
    ns_uri = ""
    if root.tag.startswith("{"):
        ns_uri = root.tag[1:].split("}")[0]
    tag0 = _localname(root.tag).lower()
    if "alto" in ns_uri.lower() or tag0 == "alto":
        return "ALTO"
    if "primaresearch.org/page" in ns_uri.lower() or tag0 in ("pcgts", "page"):
        return "PAGE"
    # Fallback: look for PAGE-typical nodes
    for el in root.iter():
        ln = _localname(el.tag)
        if ln in ("PcGts", "TextRegion"):
            return "PAGE"
    return "ALTO"

def _get_textline_polygons_alto(root):
    """
    Extract TextLine polygons from ALTO:
    prefers TextLine->Shape->Polygon, falls back to TextLine->Polygon.
    """
    polys = []
    for el in root.iter():
        if _localname(el.tag) == "TextLine":
            poly_el = None
            # Some ALTO files use Shape->Polygon
            for ch in el:
                if _localname(ch.tag) == "Shape":
                    for ch2 in ch:
                        if _localname(ch2.tag) == "Polygon":
                            poly_el = ch2
                            break
                if poly_el is not None:
                    break
            # Or direct Polygon under TextLine
            if poly_el is None:
                for ch in el:
                    if _localname(ch.tag) == "Polygon":
                        poly_el = ch
                        break
            if poly_el is not None:
                pts_attr = poly_el.attrib.get("POINTS") or poly_el.attrib.get("points")
                if pts_attr:
                    pts = parse_polygon(pts_attr)
                    if len(pts) >= 3:
                        polys.append(pts)
    return polys

def _get_textline_polygons_and_baselines_page(root):
    """
    Return (polygons, baselines):
      - polygons: list of Coords polygons (tight line shapes)
      - baselines: list of baseline polylines (used only if Coords missing)
    """
    polys, baselines = [], []
    for el in root.iter():
        if _localname(el.tag) != "TextLine":
            continue
        coords = None
        baseline = None
        for ch in el:
            ln = _localname(ch.tag)
            if ln == "Coords" and ("points" in ch.attrib):
                coords = ch.attrib["points"]
            elif ln == "Baseline" and ("points" in ch.attrib):
                baseline = ch.attrib["points"]

        if coords:
            pts = parse_polygon(coords)
            if len(pts) >= 3:
                polys.append(pts)
        elif baseline:
            bpts = parse_polygon(baseline)
            if len(bpts) >= 2:
                baselines.append(bpts)
    return polys, baselines


def process_image(image_path, xml_path, output_dir, args):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Image not found: {image_path}")
        return

    tree = ET.parse(xml_path)
    root = tree.getroot()

    flavour = _get_xml_flavour(root)
    if flavour == "PAGE":
        polygons, baselines = _get_textline_polygons_and_baselines_page(root)
    else:
        polygons = _get_textline_polygons_alto(root)
        baselines = []

    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    # 1) Tight Coords polygons (minimal noise)
    for i, points in enumerate(polygons):
        cropped = extract_polygon_gray(image, points, args)
        if cropped is not None and cropped.size > 0:
            out_path = output_dir / f"{image_path.stem}_line_{saved:04}.png"
            cv2.imwrite(str(out_path), cropped)
            saved += 1

    # 2) Fallback: baseline ribbon (when Coords missing)
    for bpts in baselines:
        cropped = extract_from_baseline_ribbon(image, bpts, args)
        if cropped is not None and cropped.size > 0:
            out_path = output_dir / f"{image_path.stem}_line_{saved:04}.png"
            cv2.imwrite(str(out_path), cropped)
            saved += 1

    print(f"Processed {image_path.name} with {len(polygons)} polygons + {len(baselines)} baselines, saved {saved}")


def polygon_extraction(args):
    """
    Batch over a folder:
      - for each supported image, look for a same-stem .xml,
      - process and write cropped lines to output.
    """
    data_folder = Path(args.src)
    output_folder = Path(args.binarized_lines)
    output_folder.mkdir(exist_ok=True)

    # Extend this list if you also use TIFF, etc.
    extensions = [".jpg", ".jpeg", ".png"]

    for image_path in data_folder.iterdir():
        if image_path.suffix.lower() in extensions and image_path.is_file():
            xml_path = data_folder / f"{image_path.stem}.xml"
            if xml_path.exists():
                process_image(image_path, xml_path, output_folder, args)
            else:
                print(f"Missing XML for: {image_path.name}")
