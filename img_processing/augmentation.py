import os
import cv2
import numpy as np
import lxml.etree as ET  # Faster XML parsing
import shutil
import random
import multiprocessing as mp
import tqdm

def apply_bleeding_effect(image, blur_strength=15, stretch_factor=1.2, alpha=0.4):
    """ Simulates ink bleeding effect by blurring and stretching the image slightly. """
    flipped = cv2.flip(image, 1)  # Mirror flip
    blurred = cv2.GaussianBlur(flipped, (blur_strength, blur_strength), blur_strength // 3)

    h, w = blurred.shape[:2]
    stretched = cv2.resize(blurred, (w, int(h * stretch_factor)))
    resized = cv2.resize(stretched, (w, h))

    return cv2.addWeighted(image.astype(np.float32), 1 - alpha, resized.astype(np.float32), alpha, 0).astype(np.uint8)

def apply_morphological_transforms(image):
    """ Applies random morphological transformations such as dilation, erosion, and blurring. """
    kernel = np.ones((2, 2), np.uint8)

    if random.random() < 0.5:
        image = cv2.dilate(image, kernel, iterations=random.randint(1, 3))  # Thickens text
    if random.random() < 0.5:
        image = cv2.erode(image, kernel, iterations=random.randint(1, 2))  # Thins text
    if random.random() < 0.3:
        image = cv2.GaussianBlur(image, (3, 3), 0)  # Slight blurring
    if random.random() < 0.5:
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)  # Simulate worn-out characters

    return image

def apply_random_shadow(image):
    """ Adds random shadows or ink smudges to the image. """
    h, w = image.shape[:2]
    num_shadows = random.randint(1, 3)  # Reduce the number of shadows for clarity

    for _ in range(num_shadows):
        shadow_mask = np.zeros_like(image, dtype=np.uint8)
        center = (random.randint(0, w), random.randint(0, h))
        axes = (random.randint(10, 50), random.randint(15, 50))
        angle = random.randint(0, 180)
        color = random.randint(10, 100)  # Darker shadows for realism
        cv2.ellipse(shadow_mask, center, axes, angle, 0, 360, (color, color, color), -1)

        alpha = np.random.uniform(0.2, 0.7)  # Adjust shadow intensity
        image = cv2.addWeighted(image, 1, shadow_mask, alpha, 0)

    return image

def process_image(file_name, args):
    """ Processes a single image and its corresponding ALTO XML file, applying augmentations. """
    print(f"Processing {file_name}")
    img_path = os.path.join(args.src, file_name)
    xml_path = os.path.join(args.src, os.path.splitext(file_name)[0] + ".xml")

    if not os.path.exists(xml_path):
        print(f"Skipping {file_name} (missing XML).")
        return

    # Load image and XML
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extract text baselines from XML
    textlines = []
    for textline in root.findall(".//TextLine"):
        baseline = textline.find("Baseline")
        if baseline is not None:
            points = np.array([list(map(int, p.split(','))) for p in baseline.attrib["points"].split()])
            textlines.append((textline, points))

    # Save original image and XML
    shutil.copy(img_path, os.path.join(args.augmented_pages, file_name))
    shutil.copy(xml_path, os.path.join(args.augmented_pages, os.path.splitext(file_name)[0] + ".xml"))

    # Generate augmented images
    for i in range(args.augmentation_times):
        aug_img = img.copy()
        aug_img = apply_morphological_transforms(aug_img)
        aug_img = apply_random_shadow(aug_img)
        if random.random() < 0.5:
            aug_img = apply_bleeding_effect(aug_img)

        # Modify baseline coordinates randomly
        aug_textlines = [
            (textline, points + np.random.randint(-args.baseline_noise, args.baseline_noise + 1, size=points.shape))
            for textline, points in textlines
        ]

        # Generate new file names
        aug_img_name = f"{os.path.splitext(file_name)[0]}_aug{i}.png"
        aug_xml_name = f"{os.path.splitext(file_name)[0]}_aug{i}.xml"

        # Update XML baseline points
        for textline, new_coords in aug_textlines:
            textline.find("Baseline").attrib["points"] = " ".join([f"{x},{y}" for x, y in new_coords])

        # Save augmented image and XML
        cv2.imwrite(os.path.join(args.augmented_pages, aug_img_name), aug_img)
        tree.write(os.path.join(args.augmented_pages, aug_xml_name), pretty_print=True, encoding="UTF-8")

def augmentation(args):
    """ Main function to start the data augmentation process. """
    os.makedirs(args.augmented_pages, exist_ok=True)  # Ensure the output directory exists
    files = [(f, args) for f in os.listdir(args.src) if f.lower().endswith((".png", ".jpg"))]
    with mp.Pool(processes=args.workers) as pool:
        for _ in tqdm.tqdm(pool.starmap(process_image, files),
                           total=len(files)):
            pass

    print(f"Data augmentation completed! Augmented data saved in {args.augmented_pages}")


