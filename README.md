# Cilantro HSV Analyzer

A Python-based tool for comparing cilantro images using HSV color-based segmentation and computer vision.

## Overview

This project provides tools for extracting and comparing HSV color statistics from multiple cilantro images:

1. **HSV Threshold Finder** (`hsv_threshold_finder.py`) - Interactive tool to find optimal HSV threshold values for your camera and lighting setup.

2. **Batch Cilantro Analyzer** (`batch_cilantro_analyzer.py`) - Processes multiple cilantro images using the same HSV mask and compiles statistics into a comparison table.

3. **Single Image Analyzer** (`cilantro_analyzer.py`) - Analyzes individual cilantro images with detailed visualization.

## How It Works

### HSV Color Space

The tool uses **HSV (Hue, Saturation, Value)** instead of RGB because:

- **Hue (H)**: The actual color (green for cilantro), independent of lighting
- **Saturation (S)**: Color intensity (vivid vs. dull)
- **Value (V)**: Brightness level (bright vs. dark)

This separation makes it easier to isolate green cilantro from backgrounds under varying lighting conditions.

### Process

1. **Convert to HSV**: Transform images from RGB to HSV color space
2. **Apply Threshold Mask**: Use the same H, S, V ranges to segment cilantro from background
3. **Extract Statistics**: Calculate mean and standard deviation for H, S, V values
4. **Compare**: Compile results into a table for side-by-side comparison

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Setup

1. Navigate to the repository:
```bash
cd 604-P3
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

This installs:
- `opencv-python` - Computer vision library
- `numpy` - Numerical computing library

## Usage

### Step 1: Find Your HSV Thresholds (One-Time Setup)

Find the optimal HSV threshold values for your camera and lighting:

```bash
python hsv_threshold_finder.py images/cilantro.jpg
```

**Interactive Controls:**
- Adjust **H, S, V min/max** sliders in real-time
- Watch 4 windows update live:
  - Original image
  - HSV representation
  - Binary mask (white = cilantro, black = background)
  - Segmented result
- Press **'s'** to save threshold values
- Press **'q'** to quit

**Tips:**
- Default values (H: 35-85, S: 40-255, V: 40-255) work well for fresh green cilantro
- Adjust **H Min/Max** to capture all green hues
- Increase **S Min** to filter out dull/gray backgrounds
- Goal: Clean white mask covering only cilantro

### Step 2: Update Threshold Values (Optional)

If you customized the thresholds, update them in `batch_cilantro_analyzer.py`:

```python
LOWER_GREEN = np.array([35, 40, 40])   # [H_min, S_min, V_min]
UPPER_GREEN = np.array([85, 255, 255]) # [H_max, S_max, V_max]
```

### Step 3: Batch Process Images

Add all your cilantro images to the `images/` folder, then run:

```bash
python batch_cilantro_analyzer.py
```

**Options:**
```bash
# Process different directory
python batch_cilantro_analyzer.py --dir my_photos/

# Custom output filename
python batch_cilantro_analyzer.py --output results.csv

# Save mask visualizations
python batch_cilantro_analyzer.py --save-masks
```

**Output:**

**Console table:**
```
Filename                      Pixels   Coverage    H (μ)    H (σ)    S (μ)    S (σ)    V (μ)    V (σ)
cilantro_day1.jpg          1,213,284     9.95%     41.2      2.6    140.3     40.8     70.7     21.9
cilantro_day3.jpg            987,542     8.10%     43.5      3.1    125.8     45.2     65.3     23.4
```

**CSV file** (`cilantro_stats.csv`):
- All HSV statistics (mean + std dev)
- BGR color values
- Pixel counts and coverage percentages
- Image dimensions
- Ready for Excel/statistical analysis

## Example Workflow

```bash
# 1. Find optimal thresholds (one-time setup)
python hsv_threshold_finder.py images/sample.jpg

# 2. Add all cilantro images to images/ folder
cp ~/Photos/cilantro_*.jpg images/

# 3. Batch process all images
python batch_cilantro_analyzer.py --save-masks

# 4. Open cilantro_stats.csv in Excel for comparison
```

## Understanding the Output

### HSV Statistics

- **Mean Hue (μ)**: Average color value (0-179)
  - Lower values: yellowish-green
  - Higher values: bluish-green

- **Mean Saturation (μ)**: Average color intensity (0-255)
  - Lower: dull, grayish
  - Higher: vivid, intense

- **Mean Value (μ)**: Average brightness (0-255)
  - Lower: darker
  - Higher: brighter

- **Standard Deviation (σ)**: Color variation within the image
  - Lower: uniform color
  - Higher: varied coloring

### Comparing Images

Use the CSV or console table to compare:
- Color changes over time (hue shifts)
- Loss of vibrancy (saturation drops)
- Darkening (value decreases)
- Color uniformity (std dev changes)

## Troubleshooting

### "No cilantro detected!"

**Solutions:**
1. Run `hsv_threshold_finder.py` to visualize the mask
2. Adjust thresholds to be more inclusive
3. Ensure cilantro is visible in the image

### Background being detected as cilantro

**Solutions:**
1. Increase **S Min** to filter out dull colors
2. Narrow **H Min/Max** range
3. Adjust **V Min** to exclude shadows

### Cilantro not fully detected

**Solutions:**
1. Widen **H Min/Max** range
2. Lower **S Min** to include less saturated areas
3. Lower **V Min** to include darker areas

## Project Structure

```
604-P3/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── hsv_threshold_finder.py       # Interactive threshold calibration tool
├── batch_cilantro_analyzer.py    # Batch processing and comparison
├── cilantro_analyzer.py          # Single image analysis
├── images/                        # Image directory
└── cilantro_stats.csv            # Output statistics (generated)
```

## Technical Details

### HSV Ranges (OpenCV)
- **Hue (H)**: 0-179 (360° mapped to 0-179)
- **Saturation (S)**: 0-255 (0-100% mapped to 0-255)
- **Value (V)**: 0-255 (brightness)

### Morphological Operations
- **MORPH_CLOSE**: Fills small holes
- **MORPH_OPEN**: Removes noise/speckles
- Results in cleaner segmentation

## License

Open source for educational and research purposes.
