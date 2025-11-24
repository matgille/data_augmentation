import cv2 as cv
import numpy as np
import os
import random
import argparse
import glob
from scriptlevelaugICFHR2022.Information_extraction import information_extraction
from scriptlevelaugICFHR2022.transformation import flag_judge, identify_reference_corner
from scriptlevelaugICFHR2022.transformation import bezier_transformation, affine_transformation, L2A_transformation
import matplotlib.pyplot as plt
import multiprocessing as mp
import tqdm

def plot_test(input_img):

    img_rgb = cv.cvtColor(input_img, cv.COLOR_BGR2RGBA)
    plt.imshow(img_rgb)
    plt.axis('off')  # Hide axis ticks
    plt.title('RGB Image')
    plt.show()

def new_local(src, times=1, stroke_radius=2, k1_control_field_corner=0.6,
              k2_control_field_third_bezier=0.6, segment=2, choice_mode=3,
              deformation_strength=1.0):

    def deformation(list_all):
        list_already, list_pt_reference = [], []
        len_list_all = len(list_all)
        for _ in range(len_list_all):
            list_already.append([[[], []], []])

        for idx in range(len_list_all):
            current_already = list_already[idx]
            current_all = list_all[idx]
            flag = flag_judge(current_already)

            if flag == 4:
                list_already[idx] = bezier_transformation(current_already, current_all, flag)
                continue

            if flag == 1 and idx != 0:
                flag = 3
                ref = identify_reference_corner(current_all[0][0], list_pt_reference)
                i_ref, j_ref = ref[2], ref[1]
                dx = list_already[i_ref][0][j_ref][0] - list_all[i_ref][0][j_ref][0]
                dy = list_already[i_ref][0][j_ref][1] - list_all[i_ref][0][j_ref][1]
                x_get = current_all[0][0][0] + dx
                y_get = current_all[0][0][1] + dy
                delta_x = current_all[2][0][2][0][0] * deformation_strength
                delta_y = current_all[2][0][2][0][1] * deformation_strength
                x_already = random.uniform(x_get - delta_x, x_get + delta_x + 0.1)
                y_already = random.uniform(y_get - delta_y, y_get + delta_y + 0.1)
                current_already[0][0] = [x_already, y_already]

            if choice_mode == 1:
                list_already[idx] = bezier_transformation(current_already, current_all, flag)
            elif choice_mode == 2:
                list_already[idx] = affine_transformation(current_already, current_all, flag, stroke_radius)
            else:
                list_already[idx] = L2A_transformation(current_already, current_all, flag, segment, stroke_radius)

            for pt_idx in range(2):
                pt_ref = current_all[0][pt_idx]
                pt_new = list_already[idx][0][pt_idx]
                list_pt_reference.append((pt_ref, pt_idx, idx))
                for next_idx in range(idx + 1, len_list_all):
                    for k in range(2):
                        if pt_ref == list_all[next_idx][0][k]:
                            list_already[next_idx][0][k] = pt_new

        return list_already

    def draw_src(list_all):
        all_pts = [pt for stroke in list_all for pt in stroke[1]]
        if not all_pts:
            return None
        xs = [pt[0] for pt in all_pts]
        ys = [pt[1] for pt in all_pts]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_len, y_len = x_max - x_min, y_max - y_min
        k1, k2 = 1.1, 1.1
        width, height = int(x_len * k1 + 10), int(y_len * k2 + 10)
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        offset = [int((width - x_len) / 2), int((height - y_len) / 2)]
        for pt in all_pts:
            x_draw = int(pt[0] - x_min + offset[0])
            y_draw = int(pt[1] - y_min + offset[1])
            cv.circle(canvas, (x_draw, y_draw), stroke_radius, (0, 0, 0), -1)
        return canvas

    def perspective_transform_image(image, strength=0.2):
        height, width = image.shape[:2]
        max_dx = int(width * strength)
        max_dy = int(height * strength)
        pts1 = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])
        pts2 = np.float32([
            [random.randint(0, max_dx), random.randint(0, max_dy)],
            [width - 1 - random.randint(0, max_dx), random.randint(0, max_dy)],
            [random.randint(0, max_dx), height - 1 - random.randint(0, max_dy)],
            [width - 1 - random.randint(0, max_dx), height - 1 - random.randint(0, max_dy)],
        ])
        M = cv.getPerspectiveTransform(pts1, pts2)
        return cv.warpPerspective(image, M, (width, height), borderValue=(255, 255, 255))

    list_info = information_extraction(src, k1_control_field_corner, k2_control_field_third_bezier)
    if list_info is None:
        print("Nessuna informazione estratta: salvo direttamente l’immagine originale senza deformazione.")
        return [src.copy() for _ in range(times)]

    results = []
    while len(results) < times:
        list_final = deformation(list_info)
        image = draw_src(list_final)
        #plot_test(image)
        if image is None or np.mean(image < 128) < 0.01:
            continue
        if choice_mode == 4:
            image = perspective_transform_image(image, strength=deformation_strength)
        #image = cv.GaussianBlur(image, (3, 3), 0) #not helping
        #plot_test(image)
        if image.shape[2] != 3:
            continue
        results.append(image)
    return results

def treat_single_image(method_name, method_id, img_path, args):
    output_subfolder = os.path.join(args.augmented_lines, method_name)
    filename = os.path.basename(img_path)
    if "_debug_overlay" in filename.lower():
        return

    gray = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    if gray is None:
        print(f"Could not read: {img_path}")
        return

    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    img_bgr = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)

    print(f"Processing: {filename} using {method_name} transformation")
    augmented_list = new_local(img_bgr, times=args.augmentation_times,
                               choice_mode=method_id,
                               deformation_strength=args.augmentation_deformation_strength,
                               stroke_radius=args.augmentation_stroke_radius,
                               k1_control_field_corner=args.augmentation_k1,
                               k2_control_field_third_bezier=args.augmentation_k2)

    base_name = os.path.splitext(filename)[0]
    for i, aug_img in enumerate(augmented_list):
        save_name = f"{base_name}_{method_name}_{i + 1}.jpg"
        save_path = os.path.join(output_subfolder, save_name)
        cv.imwrite(save_path, aug_img)

def generate(args):

    method_names = {"bezier": 1, "affine": 2, "L2A": 3, "perspective": 4}
    os.makedirs(args.augmented_lines, exist_ok=True)
    methods_to_run = method_names.items() if args.augmentation_transform == "all" else [(args.augmentation_transform, method_names[args.augmentation_transform])]

    for method_name, method_id in methods_to_run:
        output_subfolder = os.path.join(args.augmented_lines, method_name)
        os.makedirs(output_subfolder, exist_ok=True)

        image_paths = glob.glob(os.path.join(args.binarized_lines_clean, "**", "*.png"), recursive=True)
        image_paths += glob.glob(os.path.join(args.binarized_lines_clean, "**", "*.jpg"), recursive=True)

        image_paths.sort()

        with mp.Pool(processes=args.workers) as pool:
            data = [(method_name, method_id, img_path, args) for img_path in image_paths]
            for _ in tqdm.tqdm(pool.starmap(treat_single_image, data),
                               total=len(data)):
                pass

    print("✅ Done. Output salvato in:", args.augmented_lines)
