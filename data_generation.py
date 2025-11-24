import argparse
import json
from img_processing import polygonExtraction
from img_processing import cleaner
from img_processing import augmentation
from scriptlevelaugICFHR2022 import script_aug
from img_processing import rebuild_pages

def load_config(path):
    with open(path) as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json')
    args, _ = parser.parse_known_args()

    config = load_config(args.config)

    parser.set_defaults(**config)
    args = parser.parse_args()

    augmentation.augmentation(args)


    polygonExtraction.polygon_extraction(args)

    cleaner.clean_output_lines(input_dir=args.binarized_lines,
                               output_dir=args.binarized_lines_clean,
                               min_area=args.cleaner_min_area,
                               preview=False,
                               args=args)

    script_aug.generate(args)

    # --- Backward compatibility between `data_root` (new) and `data_folder` (old) ---
    data_root = getattr(args, 'data_root', None)
    if data_root is None:
        data_root = getattr(args, 'data_folder', None)
    if data_root is None:
        raise ValueError("Missing input root: provide `data_root` in config.json (or legacy `data_folder`).")

    rebuild_pages.rebuild_pages_by_method(
        base_folder=args.augmented_lines,
        data_root=data_root,                
        output_folder=args.rebuilt_pages,
        augmentations=args.augmentation_times,
        args=args
    )



if __name__ == "__main__":
    main()

