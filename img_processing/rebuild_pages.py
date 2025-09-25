import cv2
import numpy as np
import os
import re
from glob import glob
from collections import defaultdict
import xml.etree.ElementTree as ET

def extract_line_number(filename):
    match = re.search(r'line[_\-]?(\d+)', filename)
    return int(match.group(1)) if match else float('inf')

def parse_alto_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    namespace = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}
    lines = []
    for text_line in root.findall('.//alto:TextLine', namespace):
        hpos = int(text_line.attrib.get('HPOS', 0))
        vpos = int(text_line.attrib.get('VPOS', 0))
        height = int(text_line.attrib.get('HEIGHT', 0))
        width = int(text_line.attrib.get('WIDTH', 0))
        lines.append((hpos, vpos, width, height, text_line))
    return tree, root, lines

def update_alto_with_augmented(tree, root, lines, subdir):
    namespace = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}
    ET.register_namespace('', namespace['alto'])
    for i, (hpos, vpos, width, height, text_line) in enumerate(lines):
        text_line.set("ID", f"{subdir}_line_{i}")
        text_line.set("STYLE", f"reconstructed_from_{subdir}")
    return tree

def rebuild_pages_by_method(base_folder="augmented_output", data_folder="data", output_folder="rebuilt_pages", augmentations=4):
    os.makedirs(output_folder, exist_ok=True)

    original_image_path = glob(os.path.join(data_folder, '*.jpg'))[0]
    alto_path = glob(os.path.join(data_folder, '*.xml'))[0]

    original_image = cv2.imread(original_image_path)
    alto_tree, alto_root, alto_boxes = parse_alto_xml(alto_path)

    for subdir in os.listdir(base_folder):
        subfolder_path = os.path.join(base_folder, subdir)
        if not os.path.isdir(subfolder_path):
            continue

        image_files = [f for f in glob(os.path.join(subfolder_path, "*.jpg")) if "debug overlay" not in f.lower()]

        grouped_by_line = defaultdict(list)
        for f in image_files:
            filename = os.path.basename(f)
            line_number = extract_line_number(filename)
            grouped_by_line[line_number].append(f)

        #grouped_by_line = [sorted(grouped_by_line[line]) for line in grouped_by_line.keys()]
        for line in grouped_by_line.keys():
            grouped_by_line[line].sort()

        sorted_line_numbers = sorted(grouped_by_line.keys())

        # Create a page for each variation
        for augmentation_index in range(augmentations):
            # Create transparent canvas (RGBA)
            canvas = np.ones((*original_image.shape[:2], 4), dtype=np.uint8) * 255
            canvas[:, :, 3] = 0  # Fully transparent initially
            for i, line_number in enumerate(sorted_line_numbers):
                if i >= len(alto_boxes):
                    break
                x, y, w, h, _ = alto_boxes[i]
                selected = grouped_by_line[line_number][augmentation_index]
                line_img = cv2.imread(selected)
                if line_img is None:
                    continue

                # Resize line image
                line_img = cv2.resize(line_img, (w, h))

                # Convert to RGBA
                rgba = cv2.cvtColor(line_img, cv2.COLOR_BGR2BGRA)

                # Make white pixels transparent
                white_mask = np.all(rgba[:, :, :3] > 240, axis=2)
                rgba[white_mask, 3] = 0  # Set alpha to 0 for white pixels

                # Ensure target region on canvas matches
                canvas_region = canvas[y:y + h, x:x + w]

                # Alpha blending: replace only non-transparent pixels from line
                non_transparent_mask = rgba[:, :, 3] > 0
                canvas_region[non_transparent_mask] = rgba[non_transparent_mask]

                # Assign back to canvas
                canvas[y:y + h, x:x + w] = canvas_region
            save_img_path = os.path.join(output_folder, f"reconstructed_{subdir}_{augmentation_index+1}.jpg")
            save_xml_path = os.path.join(output_folder, f"reconstructed_{subdir}_{augmentation_index+1}.xml")

            cv2.imwrite(save_img_path, canvas)
            updated_tree = update_alto_with_augmented(alto_tree, alto_root, alto_boxes, subdir)
            updated_tree.write(save_xml_path, encoding="utf-8", xml_declaration=True)

            print(f"✅ Page et ALTO reconstruits pour '{subdir}' sauvegardés dans : {save_img_path} et {save_xml_path}")

