[! Forked from https://scm.univ-tours.fr/cesr/prima/data_augmentation]

# 🖋️ Low-Cost Synthetic Data Generation for HTR Training: Evaluating a Multimodal Strategy for Historical Manuscript Processing

This tool performs **data augmentation on historical document images** (e.g., manuscript scans) and their corresponding **ALTO XML** files. It incorporates both
a denoising phase and various augmentation techniques to create diverse and realistic training data.


## Features

- Applies **morphological transformations** (dilation, erosion, blurring)
- Simulates **ink bleeding**
- Adds **random shadows and smudges**
- Randomly perturbs **baseline coordinates** in ALTO XML
- Maintains 1:1 correspondence between augmented images and updated XML
- Applies **Sauvola binarization** to reduce noice
- Generates different script variations using **Bézier curves**
- Reconstructs pages from synthetic data

---

## How to Use

1. **Prepare your input data**:
   - Place `.png` or `.jpg` images and their corresponding `.xml` ALTO files in the `data` folder.
   - Each image must have a corresponding `.xml` file with the **same filename (excluding extension)**.


2. **Run the script**:
   ```bash
   python data_generation.py
    ```
    Check your output:
   
     Augmented images and XMLs will be saved in the `data/rebuilt_pages` folder.
   
     For each input image, 4 augmented versions will be created by default.


# Configuration

You can customize the following parameters under `config.json`:

| parameter | default value              | description                                  |
|-----------|----------------------------|----------------------------------------------|
| src       | data/src                   | iput folder                                  |
| binarized_lines       | data/binarized_lines       | output folder for binarized lines            |
| binarized_lines_clean       | data/binarized_lines_clean | output folder for clean binarized lines      |
| augmented_lines       | data/augmented_lines       | output folder for generated lines            |
| rebuilt_pages       | data/rebuilt_pages         | output folder for rebuilt pages              |
| augmented_pages       | data/augmented_pages       | output folder for augmented pages            |
| baseline_noise       | 4                          | baseline noise of visual augmentation        |
| crop_margin_v       | 5                          | Crop margin vertical value                   |
| crop_margin_h       | 15                         | Crop margin horizontal value                 |
| sauvola_window_size       | 35                         | Window size for Sauvola binarization         |
| sauvola_k       | 0.3                        | Window size for Sauvola's positive parameter |
| cleaner_min_area       | 15                         | Min area to clean                            |
| augmentation_k1       | 0.1                        | Bézier k1 control field                      |
| augmentation_k2       | 0.2                        | Bézier k2 control field                      |
| augmentation_stroke_radius       | 4                          | Augmentation stroke radius                   |



# Requirements

Create a virtual environment and Install dependencies using conda:

`conda env create -f augmentation_htr.yaml`

# Notes

- Augmentation is non-destructive: originals are copied as-is to the output folder.
- Random operations are used for realism — each run will produce slightly different outputs.
- Bézier augmentation's core is a fork from [script-level_aug_ICFHR2022](https://github.com/IMU-MachineLearningSXD/script-level_aug_ICFHR2022), which we adapted to work with historical documents.

# How to cite

You can cite it using the [CITATION.cff](cm.univ-tours.fr/cesr/prima/data_augmentation/-/blob/main/CITATION.cff) file or cite as following:

```bash
Crespi, Serena Carlamaria, and Carlos Emiliano González-Gallardo.
PRIMA HTR Augmentation Code. Version 1.0, 25 Sept. 2025,
ERC PRIMA (hosted at CESR, Université de Tours), https://gitlab.com/cesr/prima/data_augmentation.
```


# License & Attribution
This script is provided freely for research and experimental use.
It is part of the ongoing experiments and research conducted within the ERC PRIMA project (Grant No. 101142242), 
by Serena Carlamaria Crespi and Carlos Emiliano González-Gallardo.
