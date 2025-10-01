import os
import re
from glob import glob
from collections import defaultdict

import cv2
import numpy as np
import xml.etree.ElementTree as ET


# -----------------------
# Utilities
# -----------------------

def extract_line_number(filename: str):
    """Extract the line number from a filename (matches '...line_0005...' or '...line-0005...')."""
    m = re.search(r'line[_\-]?(\d+)', filename)
    return int(m.group(1)) if m else float('inf')


def _localname(tag: str) -> str:
    """Return the localname (strip namespace) of an XML tag."""
    return tag.split('}')[-1] if '}' in tag else tag


def _detect_flavour(root) -> str:
    """Heuristically detect whether the XML is PAGE or ALTO by namespace/root tag/content."""
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag[1:].split("}")[0].lower()
    tag0 = _localname(root.tag).lower()
    if "primaresearch.org/page" in ns or tag0 in ("pcgts", "page"):
        return "PAGE"
    if "alto" in ns or tag0 == "alto":
        return "ALTO"
    # Fallback: check typical PAGE nodes
    for el in root.iter():
        ln = _localname(el.tag)
        if ln in ("PcGts", "TextRegion"):
            return "PAGE"
    return "ALTO"


def _register_default_and_common_ns(root):
    """
    Register the document's default namespace to avoid 'ns0:' prefixes in output,
    and a couple of common prefixes (xsi, xlink) to avoid ns1/ns2 surprises.
    """
    if root.tag.startswith('{'):
        uri = root.tag[1:].split('}')[0]
        ET.register_namespace('', uri)  # default ns: no prefix on tags
    ET.register_namespace('xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')


# -----------------------
# ALTO / PAGE parsers
# -----------------------

def parse_alto_xml(xml_path):
    """
    Return (tree, root, lines) where lines is a list of (x, y, w, h, element).
    Prefers a Polygon under TextLine (tight bbox). Falls back to HPOS/VPOS/WIDTH/HEIGHT.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = []

    for tl in root.iter():
        if _localname(tl.tag) != "TextLine":
            continue

        # 1) Try TextLine->Shape->Polygon or TextLine->Polygon for a tight bbox
        poly_el = None
        for ch in tl:
            if _localname(ch.tag) == "Shape":
                for ch2 in ch:
                    if _localname(ch2.tag) == "Polygon":
                        poly_el = ch2
                        break
            if poly_el is not None:
                break
        if poly_el is None:
            for ch in tl:
                if _localname(ch.tag) == "Polygon":
                    poly_el = ch
                    break

        if poly_el is not None and (poly_el.get("POINTS") or poly_el.get("points")):
            pts_attr = poly_el.get("POINTS") or poly_el.get("points")
            s = pts_attr.replace(",", " ")
            toks = s.split()
            pts = []
            for i in range(0, len(toks) - 1, 2):
                try:
                    x = int(float(toks[i])); y = int(float(toks[i + 1]))
                    pts.append((x, y))
                except ValueError:
                    continue
            if len(pts) >= 3:
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                x, y, w, h = min(xs), min(ys), (max(xs) - min(xs)), (max(ys) - min(ys))
                lines.append((x, y, w, h, tl))
                continue

        # 2) Fallback: rectangular attributes on TextLine
        hpos = int(tl.attrib.get('HPOS', 0))
        vpos = int(tl.attrib.get('VPOS', 0))
        height = int(tl.attrib.get('HEIGHT', 0))
        width = int(tl.attrib.get('WIDTH', 0))
        lines.append((hpos, vpos, width, height, tl))

    return tree, root, lines


def parse_page_xml(xml_path):
    """
    Return (tree, root, lines) where lines is a list of (x, y, w, h, element) from PAGE:
    read TextLine/Coords@points and compute a tight bbox. If Coords is missing, skip the line.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = []

    for tl in root.iter():
        if _localname(tl.tag) != "TextLine":
            continue

        coords = None
        for ch in tl:
            if _localname(ch.tag) == "Coords" and ch.get("points"):
                coords = ch.get("points")
                break
        if not coords:
            # For page rebuilding we skip Baseline-only lines to avoid noisy placement
            continue

        s = coords.replace(",", " ")
        toks = s.split()
        pts = []
        for i in range(0, len(toks) - 1, 2):
            try:
                x = int(float(toks[i])); y = int(float(toks[i + 1]))
                pts.append((x, y))
            except ValueError:
                continue
        if len(pts) < 3:
            continue

        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        x, y, w, h = min(xs), min(ys), (max(xs) - min(xs)), (max(ys) - min(ys))
        lines.append((x, y, w, h, tl))

    return tree, root, lines


# -----------------------
# Minimal XML metadata updates
# -----------------------

def update_alto_with_augmented(tree, root, subdir):
    """
    Minimal ALTO update: assign deterministic IDs and a STYLE marker to track reconstruction source.
    Uses the natural order of TextLine elements in the document.
    """
    for i, tl in enumerate(el for el in root.iter() if _localname(el.tag) == "TextLine"):
        tl.set("ID", f"{subdir}_line_{i:04d}")
        tl.set("STYLE", f"reconstructed_from_{subdir}")
    return tree


def update_page_with_augmented(tree, root, subdir):
    """
    Minimal PAGE update: append a marker into @custom on each TextLine to track reconstruction source.
    """
    for i, tl in enumerate(el for el in root.iter() if _localname(el.tag) == "TextLine"):
        prev = tl.get("custom", "")
        marker = f"reconstructed_from:{subdir};index:{i:04d}"
        tl.set("custom", (prev + ";" if prev else "") + marker)
    return tree


# -----------------------
# Page rebuilding
# -----------------------

def rebuild_pages_by_method(base_folder="augmented_output",
                            data_folder="data",
                            output_folder="rebuilt_pages",
                            augmentations=4):
    os.makedirs(output_folder, exist_ok=True)

    # Find source image (support common formats)
    img_candidates = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"):
        img_candidates += glob(os.path.join(data_folder, ext))
    if not img_candidates:
        raise FileNotFoundError("No source image found in data_folder.")
    original_image_path = img_candidates[0]

    # Find source XML
    xml_candidates = glob(os.path.join(data_folder, '*.xml'))
    if not xml_candidates:
        raise FileNotFoundError("No XML found in data_folder.")
    xml_path = xml_candidates[0]

    original_image = cv2.imread(original_image_path, cv2.IMREAD_COLOR)
    if original_image is None:
        raise RuntimeError(f"Cannot read source image: {original_image_path}")
    H, W = original_image.shape[:2]

    # Parse once to obtain the line bboxes used for compositing
    tree0 = ET.parse(xml_path)
    root0 = tree0.getroot()
    flavour = _detect_flavour(root0)

    if flavour == "PAGE":
        _, _, boxes = parse_page_xml(xml_path)
    else:
        _, _, boxes = parse_alto_xml(xml_path)

    # For each subfolder in base_folder (each set of line crops)
    for subdir in os.listdir(base_folder):
        subfolder_path = os.path.join(base_folder, subdir)
        if not os.path.isdir(subfolder_path):
            continue

        # Collect line crops (jpg/png), ignoring any debug overlays
        image_files = [f for f in glob(os.path.join(subfolder_path, "*.jpg")) if "debug overlay" not in f.lower()]
        image_files += [f for f in glob(os.path.join(subfolder_path, "*.png")) if "debug overlay" not in f.lower()]
        if not image_files:
            continue

        # Group by line number inferred from filename
        grouped_by_line = defaultdict(list)
        for f in image_files:
            filename = os.path.basename(f)
            line_number = extract_line_number(filename)
            grouped_by_line[line_number].append(f)

        # Sort
        for ln in grouped_by_line.keys():
            grouped_by_line[ln].sort()
        sorted_line_numbers = sorted(grouped_by_line.keys())

        # Create one page per augmentation index
        for augmentation_index in range(augmentations):
            # White RGB canvas (no alpha needed)
            canvas = np.full((H, W, 3), 255, dtype=np.uint8)

            # Composite line-by-line
            for i, line_number in enumerate(sorted_line_numbers):
                if i >= len(boxes):
                    break

                x, y, w, h, _ = boxes[i]
                files_for_line = grouped_by_line[line_number]
                if augmentation_index >= len(files_for_line):
                    continue

                selected = files_for_line[augmentation_index]
                line_img = cv2.imread(selected, cv2.IMREAD_COLOR)
                if line_img is None:
                    continue

                if w <= 0 or h <= 0:
                    continue

                # Resize crop to the target bbox
                line_img = cv2.resize(line_img, (w, h), interpolation=cv2.INTER_LINEAR)

                # Build a mask: treat near-white as background
                gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
                mask = (gray < 240)  # True where there is ink/content

                # Paste only where mask=True
                region = canvas[y:y + h, x:x + w]
                region[mask] = line_img[mask]
                canvas[y:y + h, x:x + w] = region

            # Save reconstructed image (PNG keeps quality; JPG is also fine)
            save_img_path = os.path.join(output_folder, f"reconstructed_{subdir}_{augmentation_index + 1}.png")
            cv2.imwrite(save_img_path, canvas)

            # Re-parse a fresh XML (to avoid cumulative edits),
            # register default/common namespaces, apply minimal updates, and write.
            xml_tree = ET.parse(xml_path)
            xml_root = xml_tree.getroot()
            _register_default_and_common_ns(xml_root)

            if flavour == "PAGE":
                xml_tree = update_page_with_augmented(xml_tree, xml_root, subdir)
            else:
                xml_tree = update_alto_with_augmented(xml_tree, xml_root, subdir)

            save_xml_path = os.path.join(output_folder, f"reconstructed_{subdir}_{augmentation_index + 1}.xml")
            xml_tree.write(save_xml_path, encoding="utf-8", xml_declaration=True)

            print(f"Rebuilt image+XML for '{subdir}': {save_img_path} , {save_xml_path}")
