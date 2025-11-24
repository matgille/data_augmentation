import os
import re
from glob import glob
from collections import defaultdict
import tqdm
import multiprocessing as mp
import cv2
import numpy as np
import xml.etree.ElementTree as ET

# ---------- helpers you already had (kept) ----------
def extract_line_number(filename: str):
    m = re.search(r'line[_\-]?(\d+)', filename)
    return int(m.group(1)) if m else float('inf')

def _localname(tag: str) -> str:
    return tag.split('}')[-1] if '}' in tag else tag

def _detect_flavour(root) -> str:
    # Heuristically detect PAGE vs ALTO based on namespace/root tags
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag[1:].split("}")[0].lower()
    tag0 = _localname(root.tag).lower()
    if "primaresearch.org/page" in ns or tag0 in ("pcgts", "page"):
        return "PAGE"
    if "alto" in ns or tag0 == "alto":
        return "ALTO"
    # Fallback: scan for typical PAGE nodes
    for el in root.iter():
        ln = _localname(el.tag)
        if ln in ("PcGts", "TextRegion"):
            return "PAGE"
    return "ALTO"

def _register_default_and_common_ns(root):
    # Register default ns so output tags are not prefixed (avoid ns0:)
    if root.tag.startswith('{'):
        uri = root.tag[1:].split('}')[0]
        ET.register_namespace('', uri)
    # Also register a couple of common prefixes to avoid ns1/ns2 surprises
    ET.register_namespace('xsi', 'http://www.w3.org/2001/XMLSchema-instance')
    ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')

# ---------- new helpers ----------
_PAGE_KEY_RE = re.compile(r'^(?P<key>.+?)_?line[_\-]?\d+', re.IGNORECASE)

def infer_page_key_from_filename(fname: str) -> str | None:
    """
    From a crop filename like 'XXVI.14_1_line_0005_aug2.png' return 'XXVI.14_1'.
    Falls back to the stem before '_line' if present.
    """
    stem = os.path.splitext(os.path.basename(fname))[0]
    m = _PAGE_KEY_RE.match(stem)
    return m.group('key') if m else None

def find_source_xml_and_image(data_root: str, page_key: str):
    """
    Recursively search under data_root for an XML and an image whose basename contains page_key.
    Prefer exact startswith/page_key tokens to avoid false matches.
    """
    # XML candidates
    xml_candidates = glob(os.path.join(data_root, '**', '*.xml'), recursive=True)
    # Prefer exact start, then 'contains'
    xml_matches = [p for p in xml_candidates if os.path.splitext(os.path.basename(p))[0].startswith(page_key)]
    if not xml_matches:
        xml_matches = [p for p in xml_candidates if page_key in os.path.basename(p)]
    xml_path = xml_matches[0] if xml_matches else None

    # Image candidates
    img_candidates = []
    for ext in ('*.jpg','*.jpeg','*.png','*.tif','*.tiff'):
        img_candidates += glob(os.path.join(data_root, '**', ext), recursive=True)
    img_matches = [p for p in img_candidates if os.path.splitext(os.path.basename(p))[0].startswith(page_key)]
    if not img_matches:
        img_matches = [p for p in img_candidates if page_key in os.path.basename(p)]
    img_path = img_matches[0] if img_matches else None

    return xml_path, img_path

def parse_alto_xml(xml_path):
    """
    Parse ALTO XML and return (tree, root, lines) where lines is a list of (x, y, w, h, element).
    Prefer TextLine->Shape->Polygon (tight bbox). Fallback to HPOS/VPOS/WIDTH/HEIGHT.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = []
    for tl in root.iter():
        if _localname(tl.tag) != "TextLine":
            continue
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
                x, y, w, h = min(xs), min(ys), (max(xs)-min(xs)), (max(ys)-min(ys))
                lines.append((x, y, w, h, tl))
                continue
        # Fallback: rectangular attributes on TextLine
        hpos = int(float(tl.attrib.get('HPOS', 0)))
        vpos = int(float(tl.attrib.get('VPOS', 0)))
        height = int(float(tl.attrib.get('HEIGHT', 0)))
        width  = int(float(tl.attrib.get('WIDTH', 0)))
        lines.append((hpos, vpos, width, height, tl))
    return tree, root, lines

def parse_page_xml(xml_path):
    """
    Parse PAGE XML and return (tree, root, lines) where lines is a list of (x, y, w, h, element).
    Read TextLine/Coords@points and compute a tight bbox. Skip Baseline-only lines.
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
            # Skip lines with no polygon coords to avoid noisy placement
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
        x, y, w, h = min(xs), min(ys), (max(xs)-min(xs)), (max(ys)-min(ys))
        lines.append((x, y, w, h, tl))
    return tree, root, lines

# ---------- safe paste helper ----------
def _paste_line_safe(canvas, x, y, w, h, line_img, threshold=240):
    """
    Safely paste a line (line_img) into canvas at (x,y,w,h), using a near-white mask.
    Handles 3-channel images and clipping at borders.
    """
    H, W = canvas.shape[:2]
    if w <= 0 or h <= 0:
        return

    # Clip target ROI if bbox exceeds canvas
    x0 = max(0, x); y0 = max(0, y)
    x1 = min(W, x + w); y1 = min(H, y + h)
    rw, rh = x1 - x0, y1 - y0
    if rw <= 0 or rh <= 0:
        return

    # Resize the source to the *requested* (w,h), then crop to the clipped ROI
    src = cv2.resize(line_img, (w, h), interpolation=cv2.INTER_LINEAR)
    # Compute the local offsets if the bbox was clipped
    off_x = x0 - x
    off_y = y0 - y
    src = src[off_y:off_y + rh, off_x:off_x + rw]

    # Build mask on the (clipped) source
    if src.ndim == 3:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        gray = src
    mask = (gray < threshold)  # True where there is ink/content

    # Paste only ink pixels (vectorized by indices)
    region = canvas[y0:y0 + rh, x0:x0 + rw]
    ys, xs = np.where(mask)
    if ys.size == 0:
        return
    region[ys, xs] = src[ys, xs]
    canvas[y0:y0 + rh, x0:x0 + rw] = region

# ---------- minimal metadata updates (kept) ----------
def update_alto_with_augmented(tree, root, subdir):
    # Assign deterministic IDs and a STYLE marker to track reconstruction source
    for i, tl in enumerate(el for el in root.iter() if _localname(el.tag) == "TextLine"):
        tl.set("ID", f"{subdir}_line_{i:04d}")
        tl.set("STYLE", f"reconstructed_from_{subdir}")
    return tree

def update_page_with_augmented(tree, root, subdir):
    # Append a marker into @custom on each TextLine to track reconstruction source
    for i, tl in enumerate(el for el in root.iter() if _localname(el.tag) == "TextLine"):
        prev = tl.get("custom", "")
        marker = f"reconstructed_from:{subdir};index:{i:04d}"
        tl.set("custom", (prev + ";" if prev else "") + marker)
    return tree

# ---------- main rebuild (updated to isolate manuscripts/pages) ----------
def rebuild_pages_by_method(base_folder="augmented_output",
                            data_root="data_root_with_many_manuscripts",
                            output_folder="rebuilt_pages",
                            augmentations=4,
                            args=None):
    """
    Rebuild pages without mixing manuscripts:
    - If the structure is <base>/<method>/<page_key>/*, rebuild per page_key.
    - Otherwise, partition files by page_key inside each subfolder and rebuild per group.
    """
    print("Rebuilding pages.")
    os.makedirs(output_folder, exist_ok=True)

    def page_rebuild_subfolder(level1_path, page_key):
        page_dir = os.path.join(level1_path, page_key)
        if not os.path.isdir(page_dir):
            return
        imgs = [f for f in glob(os.path.join(page_dir, "*.jpg")) if "debug overlay" not in f.lower()]
        imgs += [f for f in glob(os.path.join(page_dir, "*.png")) if "debug overlay" not in f.lower()]
        imgs.sort()
        # Rebuild directly if the folder is “pure” (contains only one page_key)
        _rebuild_for_group(imgs, page_key)

    # Level-1: methods (e.g., bezier, L2A, affine, perspective) or leaf dirs already containing images
    for level1 in os.listdir(base_folder):
        level1_path = os.path.join(base_folder, level1)
        if not os.path.isdir(level1_path):
            continue

        # Detect whether page_key subfolders exist under level1 (CASE A)
        subfolders = [d for d in os.listdir(level1_path)
                      if os.path.isdir(os.path.join(level1_path, d))]

        # Helper: rebuild for a group of files that all belong to the same page_key
        def _rebuild_for_group(image_files, page_key_label):
            if not image_files:
                return

            # Find XML + original page image for this page_key
            xml_path, original_image_path = find_source_xml_and_image(data_root, page_key_label)
            if not xml_path or not original_image_path:
                print(f"[WARN] No XML/image for page_key '{page_key_label}' under '{data_root}'. Skip.")
                return

            original_image = cv2.imread(original_image_path, cv2.IMREAD_COLOR)
            if original_image is None:
                print(f"[WARN] Cannot read source image: {original_image_path}. Skip.")
                return
            H, W = original_image.shape[:2]

            # Parse line bounding boxes from XML
            tree0 = ET.parse(xml_path)
            root0 = tree0.getroot()
            flavour = _detect_flavour(root0)
            if flavour == "PAGE":
                _, _, boxes = parse_page_xml(xml_path)
            else:
                _, _, boxes = parse_alto_xml(xml_path)

            if not boxes:
                print(f"[WARN] No line bboxes in '{xml_path}'. Skip.")
                return

            # Group by line number inferred from filename
            grouped_by_line = defaultdict(list)
            for f in image_files:
                ln = extract_line_number(os.path.basename(f))
                grouped_by_line[ln].append(f)
            for ln in grouped_by_line:
                grouped_by_line[ln].sort()
            sorted_line_numbers = sorted(grouped_by_line.keys())

            # Prepare output: separate per method/page_key
            out_dir = os.path.join(output_folder, f"{level1}_{page_key_label}")
            os.makedirs(out_dir, exist_ok=True)



            # Map “line_xxxx” to bbox index when plausible
            # Common assumption: line_0000 → boxes[0], line_0001 → boxes[1], ...
            def _bbox_index_for_line(ln):
                if ln != float('inf') and 0 <= ln < len(boxes):
                    return ln
                return None  # not plausible

            def treat_augmentation(augmentation_index):
                canvas = np.full((H, W, 3), 255, dtype=np.uint8)

                # Try direct placement by line index first, then fall back to sequential fill
                used_boxes = set()

                # 1) Direct placement: use the same index as in filename if plausible
                for ln in sorted_line_numbers:
                    idx = _bbox_index_for_line(ln)
                    if idx is None or idx in used_boxes:
                        continue
                    files_for_line = grouped_by_line[ln]
                    if augmentation_index >= len(files_for_line):
                        continue
                    x, y, w, h, _ = boxes[idx]
                    line_img = cv2.imread(files_for_line[augmentation_index], cv2.IMREAD_COLOR)
                    if line_img is None:
                        continue
                    _paste_line_safe(canvas, x, y, w, h, line_img)
                    used_boxes.add(idx)

                # 2) Fallback: for lines without a valid index, place them in remaining boxes sequentially
                seq_boxes = [i for i in range(len(boxes)) if i not in used_boxes]
                seq_ptr = 0
                for ln in sorted_line_numbers:
                    if _bbox_index_for_line(ln) is not None:
                        continue  # already placed above
                    if seq_ptr >= len(seq_boxes):
                        break
                    files_for_line = grouped_by_line[ln]
                    if augmentation_index >= len(files_for_line):
                        continue
                    idx = seq_boxes[seq_ptr]
                    seq_ptr += 1
                    x, y, w, h, _ = boxes[idx]
                    line_img = cv2.imread(files_for_line[augmentation_index], cv2.IMREAD_COLOR)
                    if line_img is None:
                        continue
                    _paste_line_safe(canvas, x, y, w, h, line_img)

                # Save rebuilt image
                img_out = os.path.join(out_dir, f"reconstructed_{page_key_label}_{augmentation_index + 1}.png")
                cv2.imwrite(img_out, canvas)

                # Save a fresh XML annotated with the reconstruction marker
                xml_tree = ET.parse(xml_path)
                xml_root = xml_tree.getroot()
                _register_default_and_common_ns(xml_root)
                if flavour == "PAGE":
                    xml_tree = update_page_with_augmented(xml_tree, xml_root, page_key_label)
                else:
                    xml_tree = update_alto_with_augmented(xml_tree, xml_root, page_key_label)
                xml_out = os.path.join(out_dir, f"reconstructed_{page_key_label}_{augmentation_index + 1}.xml")
                xml_tree.write(xml_out, encoding="utf-8", xml_declaration=True)

                print(f"[OK] Page rebuilt '{page_key_label}' from '{level1}': {img_out} , {xml_out}")

            data = [item for item in range(augmentations)]
            print(data)
            exit(0)
            with mp.Pool(processes=args.workers) as pool:
                for _ in tqdm.tqdm(pool.starmap(treat_augmentation, data),
                                   total=len(data)):
                    pass


        # ===== CASE A: there are page_key subfolders =====
        if subfolders:
            print("CASE A")
            data = [(level1_path, page_key) for page_key in subfolders]
            with mp.Pool(processes=args.workers) as pool:
                for _ in tqdm.tqdm(pool.starmap(page_rebuild_subfolder, data),
                                   total=len(data)):
                    pass
            continue
        print("CASE B")
        # ===== CASE B: no subfolders → partition files by page_key =====
        flat_imgs = [f for f in glob(os.path.join(level1_path, "*.jpg")) if "debug overlay" not in f.lower()]
        flat_imgs += [f for f in glob(os.path.join(level1_path, "*.png")) if "debug overlay" not in f.lower()]
        if not flat_imgs:
            continue

        buckets = defaultdict(list)
        for f in flat_imgs:
            key = infer_page_key_from_filename(os.path.basename(f)) or "unknown_page"
            buckets[key].append(f)
        print("Rebuilding pages")
        for key, files in buckets.items():
            files.sort()
            _rebuild_for_group(files, key)



