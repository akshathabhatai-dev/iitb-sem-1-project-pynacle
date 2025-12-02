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
│       └── annotator.py
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

### 1. Annotate Images

Use the annotation tool to label grid cells in your images:

```bash
cd src/data_prep
python annotator.py --image_dir ../../dataset/train/images --output_csv ../../dataset/train/labels/labels.csv
```

**Controls:**
- Click cells to label them
- Press `0`, `1`, `2`, `3` to select class (no object, ball, bat, stump)
- Press `n` to save and move to next image
- Press `s` to save current image

### 2. Run Complete Pipeline

Extract features, train model, evaluate, and generate predictions:

```bash
python main_pipeline.py --mode full --tune
```

**With Visualization** (generates color-coded grid overlays on test images):

```bash
python main_pipeline.py --mode full --tune --visualize
```

### 3. Individual Pipeline Steps

**Extract features only:**
```bash
python main_pipeline.py --mode extract
```

**Train model only:**
```bash
python main_pipeline.py --mode train --tune
```

**Evaluate model:**
```bash
python main_pipeline.py --mode evaluate
```

**Generate predictions:**
```bash
python main_pipeline.py --mode predict

# With visualization
python main_pipeline.py --mode predict --visualize
```

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
| `--visualize` | Generate visualization images | Flag (off by default) |
| `--viz_output_dir` | Visualization output directory | `outputs/visualizations` |

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
- `outputs/visualizations/test/*.png` - Test images with colored prediction grids (if `--visualize` flag used)

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

The pipeline includes an optional visualization feature that generates test images with color-coded grid overlays showing model predictions.

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
| 0 | No Object | Light Gray | `(240, 240, 240)` |
| 1 | Ball | Light Green | `(144, 238, 144)` |
| 2 | Bat | Light Coral/Salmon | `(173, 216, 230)` |
| 3 | Stump | Light Blue | `(255, 182, 193)` |

### Output

Visualized images are saved to `outputs/visualizations/test/` with:
- Semi-transparent colored fills for predicted cells
- Dark gray grid lines showing cell boundaries
- Original image visible underneath

### Benefits

- **Visual Debugging**: Quickly see where model makes errors
- **Error Analysis**: Identify systematic misclassifications
- **Presentation**: Use visualized images in slides/reports
- **Quality Check**: Verify predictions without parsing CSV

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

## Expected Performance

| Metric | Target |
|--------|--------|
| Overall Accuracy | 75-85% |
| Ball Recall | >85% |
| Bat Recall | >75% |
| Stump Recall | >70% |
| No-Object Precision | >85% |

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

## License

This project is for educational purposes as part of the Programming for ML course.

## Authors

Team: The Pynacle Group
Course: Programming for Machine Learning
Institution: IIT Bombay

