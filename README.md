# Cricket Object Detection

A machine learning project for detecting cricket objects (ball, bat, stumps) in images using hand-crafted feature engineering.

## Project Overview

This project implements a grid-based object detection system that:
- Divides cricket images (800×600) into 8×8 grids (64 cells of 100×75 pixels)
- Classifies each cell as: **0**=no object, **1**=ball, **2**=bat, **3**=stump
- Uses **ONLY taught hand-crafted features** (no CNNs or deep learning)
- Achieves robust detection through comprehensive feature engineering

## Features (88 Total)

### Color Features (28)
- **RGB Histograms (14)**: Channel statistics, ratios, color variance
- **HSV Features (14)**: Hue/saturation/value statistics, color bins, purity

### Texture Features (60)
- **LBP (10)**: Local binary patterns for texture discrimination
- **HOG (20)**: Histogram of oriented gradients for shape detection
- **Gabor Filters (20)**: Multi-scale texture and pattern analysis
- **GLCM (10)**: Gray-level co-occurrence matrix for texture statistics

## Project Structure

```
project/
├── dataset/
│   ├── train/
│   │   ├── images/          # Training images (800×600)
│   │   └── labels/          # Training labels CSV
│   └── test/
│       ├── images/          # Test images (800×600)
│       └── labels/          # Test labels CSV
│
├── src/
│   ├── features/
│   │   ├── rgb_features.py
│   │   ├── hsv_features.py
│   │   ├── lbp_features.py
│   │   ├── hog_features.py
│   │   ├── gabor_features.py
│   │   ├── glcm_features.py
│   │   └── feature_extractor.py
│   │
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   │
│   └── data_prep/
│       ├── grid_utils.py
│       ├── annotator.py
│       └── csv_label_converter.py  # Convert row-column-tag to cell format
│
├── models/                   # Saved models
├── outputs/                  # Predictions and metrics
├── main_pipeline.py         # Main execution script
├── requirements.txt
└── README.md
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Verify installation:
```bash
python -c "import cv2, numpy, sklearn, skimage; print('All dependencies installed!')"
```

## Usage

### 1. Prepare Labels

You have two options for creating label files:

#### Option A: Use the Interactive Annotator

Use the annotation tool to label grid cells in your images (0=no object, 1=ball, 2=bat, 3=stump):

```bash
python src/data_prep/annotator.py --image_dir dataset/train/images --output_csv dataset/train/labels/labels.csv
```

**Expected Output:** `dataset/train/labels/labels.csv` with format: `ImageFileName,c01,c02,...,c64`

**Controls:**
- Click cells to label them
- Press `0`, `1`, `2`, `3` to select class (no object, ball, bat, stump)
- Press `n` to save and move to next image
- Press `s` to save current image

#### Option B: Convert Existing Labels

If you already have labels in row-column-tag format, convert them using the CSV label converter:

```bash
python src/data_prep/csv_label_converter.py -i <input_csv> -o <output_csv>
```

**Example:**
```bash
python src/data_prep/csv_label_converter.py -i dataset/train/labels/atul_tagged_data.csv -o dataset/train/labels/labels-atul.csv
```

This converts from row-column-tag format:
```csv
imagename,cell_1_row,cell_1_column,cell_1_object_tag,cell_2_row,cell_2_column,cell_2_object_tag,...
ball_img_2439.jpg,1,4,Stump,1,5,Stump,1,7,Bat,1,8,Bat
```

To cell-based format:
```csv
ImageFileName,c01,c02,c03,c04,c05,c06,c07,c08,...,c64
ball_img_2439.jpg,0,0,0,3,3,0,2,2,...,0
```

**Object Mapping:** Ball→1, Bat→2, Stump→3, No object→0


### 2. Run Complete Pipeline

Extract features, train model, evaluate, and generate predictions:

```bash
python main_pipeline.py --mode full --tune
```

**With Visualization** (generates color-coded grid overlays on both train and test images):

```bash
python main_pipeline.py --mode full --tune --visualize
```

### 3. Individual Pipeline Steps

**Extract features only:**
```bash
python main_pipeline.py --mode extract
```
**Expected Output:** `outputs/X_train.npy`, `outputs/y_train.npy`, `outputs/X_test.npy`, `outputs/y_test.npy`

**Train model only (with hyperparameter tuning):**
```bash
python main_pipeline.py --mode train --tune
```
**Expected Output:** `models/cricket_model.pkl` (trained Random Forest model with optimal hyperparameters)

**Evaluate model:**
```bash
python main_pipeline.py --mode evaluate
```
**Expected Output:** `outputs/metrics.txt`, `outputs/confusion_matrix.png`

**Generate predictions:**
```bash
python main_pipeline.py --mode predict

# With visualization
python main_pipeline.py --mode predict --visualize
```
**Expected Output:** `outputs/predictions.csv` (format: `ImageFileName,TrainOrTest,c01,...,c64`)

**Predict with custom paths and visualization:**
```bash
python main_pipeline.py --mode predict --model_path models/cricket_model.pkl --train_images dataset/train/images --test_images dataset/test/images --output_dir outputs --visualize
```
**Expected Output:** `outputs/test/predictions.csv` + `outputs/train/visualizations/*.png` + `outputs/test/visualizations/*.png` (color-coded grids)

### 4. Custom Paths

```bash
python main_pipeline.py \
    --mode full \
    --train_images dataset/train/images \
    --train_labels dataset/train/labels/labels.csv \
    --test_images dataset/test/images \
    --test_labels dataset/test/labels/labels.csv \
    --model_path models/my_model.pkl \
    --output_dir outputs \
    --tune
```

## Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | Pipeline mode: `extract`, `train`, `evaluate`, `predict`, `full` | Required |
| `--train_images` | Training images directory | `dataset/train/images` |
| `--test_images` | Test images directory | `dataset/test/images` |
| `--train_labels` | Training labels CSV | `dataset/train/labels/labels.csv` |
| `--test_labels` | Test labels CSV | `dataset/test/labels/labels.csv` |
| `--model_path` | Model save/load path | `models/cricket_model.pkl` |
| `--output_dir` | Output directory | `outputs` |
| `--tune` | Enable hyperparameter tuning | Flag (off by default) |
| `--visualize` | Generate visualization images for train and test | Flag (off by default) |

## Output Files

After running the pipeline, you'll find:

- `outputs/X_train.npy` - Training features
- `outputs/y_train.npy` - Training labels
- `outputs/X_test.npy` - Test features
- `outputs/y_test.npy` - Test labels
- `models/cricket_model.pkl` - Trained model
- `outputs/predictions.csv` - Predictions in required format
- `outputs/metrics.txt` - Evaluation metrics
- `outputs/confusion_matrix.png` - Confusion matrix visualization
- `outputs/train/visualizations/*.png` - Training images with colored prediction grids (if `--visualize` flag used)
- `outputs/test/visualizations/*.png` - Test images with colored prediction grids (if `--visualize` flag used)

## CSV Format

### Labels CSV Format
```
ImageFileName,c01,c02,c03,...,c64
image001.png,0,0,1,0,...,2
image002.png,0,0,0,3,...,0
```

### Predictions CSV Format
```
ImageFileName,TrainOrTest,c01,c02,c03,...,c64
image001.png,train,0,0,1,0,...,2
image002.png,test,0,0,0,3,...,0
```

## Feature Extraction Details

### RGB Features (14)
Captures raw color information essential for detecting:
- Red cricket ball (high R channel)
- Brown bat (R≈G > B)
- Green grass background (high G channel)
- White stumps (R≈G≈B, all high)

### HSV Features (14)
Lighting-invariant color detection:
- Ball detection robust to shadows
- Hue-based color discrimination
- Saturation for vivid vs dull colors

### LBP Features (10)
Texture discrimination:
- Smooth ball/bat vs textured grass
- Rotation-invariant patterns

### HOG Features (20)
Shape and edge detection:
- Vertical stumps (90° gradients)
- Horizontal bat (0°, 180° gradients)
- Circular ball (uniform gradient distribution)

### Gabor Features (20)
Multi-scale texture analysis:
- 5 scales × 4 orientations
- Circular vs elongated pattern detection

### GLCM Features (10)
Texture statistics:
- Smooth objects: high homogeneity, low contrast
- Textured grass: low homogeneity, high contrast

## Visualization Feature

The pipeline includes an optional visualization feature that generates both train and test images with color-coded grid overlays showing model predictions.

### How to Use

Add the `--visualize` flag to any pipeline command:

```bash
python main_pipeline.py --mode full --visualize
python main_pipeline.py --mode predict --visualize
```

### Color Scheme

Each grid cell is colored based on its predicted class:

| Class | Object | Color | BGR Value |
|-------|--------|-------|-----------|
| 1 | Ball | Green | `(0, 255, 0)` |
| 2 | Bat | Red | `(0, 0, 255)` |
| 3 | Stump | Blue | `(255, 0, 0)` |

Cells with class 0 (no object) remain uncolored, showing the original image.

### Output Structure

Visualized images are automatically organized into separate directories:

```
outputs/
├── train/
│   └── visualizations/
│       ├── train_image001.png
│       ├── train_image002.png
│       └── ...
└── test/
    └── visualizations/
        ├── test_image001.png
        ├── test_image002.png
        └── ...
```

Each visualized image includes:
- Semi-transparent colored fills for predicted object cells (ball/bat/stump)
- Dark gray grid lines showing cell boundaries (8×8 grid)
- Cell labels (c01, c02, ..., c64) in the top-left corner of each cell
- Original image visible underneath

### Benefits

- **Visual Debugging**: Quickly see where model makes errors on both train and test sets
- **Overfitting Detection**: Compare train vs test predictions visually
- **Error Analysis**: Identify systematic misclassifications by class/position
- **Presentation**: Use visualized images in slides/reports/video
- **Quality Check**: Verify predictions without parsing CSV
- **Annotation Verification**: Check if training labels are correct by comparing with predictions

## Model Training

The pipeline uses **Random Forest Classifier** with:
- Balanced class weights (handles imbalanced data)
- 5-fold cross-validation
- Optional grid search for hyperparameter tuning

**Hyperparameter Grid:**
- `n_estimators`: [100, 200, 300]
- `max_depth`: [15, 20, 25]
- `min_samples_split`: [5, 10, 15]
- `min_samples_leaf`: [2, 5, 10]

**Training Time:**
- Without tuning: ~2-5 minutes (70 images)
- With tuning: ~15-30 minutes (70 images)

## Troubleshooting

### ImportError: No module named 'cv2'
```bash
pip install opencv-python
```

### Image size mismatch
The pipeline automatically resizes images to 800×600. Ensure your images are at least 800×600 to avoid upscaling.

### Memory issues
If processing many images, extract features in batches:
```python
# Process 50 images at a time
for i in range(0, len(images), 50):
    batch = images[i:i+50]
    # Extract features for batch
```

### Low accuracy
1. Check annotation quality (are labels correct?)
2. Enable hyperparameter tuning: `--tune`
3. Increase training data
4. Check class balance in labels

## Development

### Testing Feature Extraction
```bash
cd src/features
python feature_extractor.py
```

### Testing Individual Features
```bash
cd src/features
python rgb_features.py
python hsv_features.py
# etc.
```


---

# Image Crop & Scale Tool

A command-line Python tool that:

* Detects real image formats (JPEG/PNG/WEBP/GIF/BMP/AVIF/HEIC)
* Converts **AVIF/HEIC → PNG**
* Crops images to **4:3 aspect ratio** (center crop)
* Resizes them to **800 × 600 pixels**
* Saves results into a new **scaled/** directory
* Preserves original format for all other image types

---

## 🚀 Features

### ✔ Format Detection (Magic Number Based)

Works even when files have incorrect extensions.

### ✔ AVIF/HEIC Support

`pillow-avif-plugin` enables Pillow to decode AVIF & HEIC files.

### ✔ Intelligent Cropping

Images are center-cropped to **4:3** without distortion.

### ✔ Scaling

All cropped images are resized to **800×600** using high-quality LANCZOS filtering.

### ✔ Output Structure

Images are written to a new directory:

```
scaled/
```

Maintains original file extensions, except:

* **AVIF → PNG**

### ✔ Safe Processing

* Skips non-image files
* Handles corrupted/truncated images
* Converts modes safely (e.g., RGBA → RGB for JPG)

---

## 📦 Requirements

Create a `requirements.txt` with:

```
Pillow>=10.0.0
pillow-avif-plugin>=1.4.3
```

Install:

```bash
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
image_scaler.py
requirements.txt
README.md
filtered/             ← Input folder (example)
scaled/               ← Output folder (auto-created)
```

---

## 🛠 Usage (Command Line)

### Basic usage:

```bash
python image_scaler.py filtered
```

### With a custom output folder:

```bash
python image_scaler.py filtered --output final_scaled
```

or shorter:

```bash
python image_scaler.py filtered -o final_scaled
```

---

## 📝 Script Behavior

### 🖼 Supported Formats (input)

| Format     | Supported | Notes                                    |
| ---------- | --------- | ---------------------------------------- |
| JPG / JPEG | ✔         | Preserved as JPG                         |
| PNG        | ✔         | Preserved as PNG (supports transparency) |
| WEBP       | ✔         | Preserved as WEBP                        |
| GIF        | ✔         | Static frame saved                       |
| BMP        | ✔         | Preserved as BMP                         |
| AVIF       | ✔         | Converted to PNG                         |
| HEIC/HEIF  | ✔         | Converted to PNG                         |

---

## 🔧 Processing Steps

For each image:

1. Detects real file format via signature
2. Opens image with Pillow
3. Converts mode (RGB/RGBA) if needed
4. Performs a **center crop** to 4:3
5. Resizes image to **800 × 600**
6. Saves into **scaled/** (or specified output directory)

---

## 🧪 Example

Input folder:

```
filtered/
    img1.jpg
    img2.png
    img3.avif
    doc.txt
```

After running:

```
scaled/
    img1.jpg         ← cropped & scaled
    img2.png         ← cropped & scaled
    img3.png         ← AVIF → PNG → cropped & scaled
```

Non-image file (`doc.txt`) is skipped.

---

## 🧩 Troubleshooting

### ❌ “Cannot write mode RGBA as JPEG”

Occurs when PNG/WEBP has transparency —
script auto-converts RGBA → RGB for JPG formats.

### ❌ Pillow cannot open some images

They may be corrupted or not real images.
These are automatically skipped.

---

## 📜 License

This project is for educational purposes as part of the Programming for ML course.

---

## Authors

Team: The Pynacle Group
Course: Programming for Machine Learning
Institution: IIT Bombay
