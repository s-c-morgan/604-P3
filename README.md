# Cilantro Freshness Analyzer

A Python-based tool for analyzing cilantro freshness using HSV color-based segmentation and computer vision.

## Overview

This project provides two main tools:

1. **HSV Threshold Finder** (`hsv_threshold_finder.py`) - An interactive tool with real-time sliders to find the perfect HSV color threshold values for your specific camera and lighting conditions.

2. **Cilantro Analyzer** (`cilantro_analyzer.py`) - Analyzes cilantro images using color segmentation to assess freshness based on hue, saturation, and brightness.

## How It Works

### Color-Based Thresholding

The tool uses **HSV (Hue, Saturation, Value) color space** instead of RGB because:

- **Hue (H)**: Represents the actual color (green for cilantro), independent of lighting
- **Saturation (S)**: Represents color intensity (vivid vs. dull)
- **Value (V)**: Represents brightness (bright vs. dark)

This separation makes it much easier to isolate green cilantro from backgrounds, even under varying lighting conditions.

### Process

1. **Convert to HSV**: Transform the image from RGB to HSV color space
2. **Define Thresholds**: Set minimum and maximum values for H, S, and V that represent "fresh green"
3. **Create Mask**: Generate a binary mask where white pixels = cilantro, black pixels = background
4. **Analyze Colors**: Calculate statistics on the segmented cilantro pixels
5. **Assess Freshness**: Score freshness based on color properties

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Setup

1. Clone or download this repository:
```bash
cd 604-P3
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

This will install:
- `opencv-python` - Computer vision library
- `numpy` - Numerical computing library

## Usage

### Step 1: Find Your HSV Thresholds (First Time Setup)

Before analyzing cilantro, you need to find the perfect HSV threshold values for your camera and lighting setup.

```bash
python hsv_threshold_finder.py <path_to_cilantro_image>
```

Example:
```bash
python hsv_threshold_finder.py samples/cilantro_fresh.jpg
```

**Interactive Controls:**
- Use the **trackbars** to adjust H, S, V min/max values in real-time
- Watch the **4 windows** update as you adjust:
  1. **Original** - Your input image
  2. **HSV** - HSV color space representation
  3. **Mask** - Binary mask (white = detected cilantro, black = background)
  4. **Result** - Segmented cilantro with background removed

- Press **'s'** to save your threshold values to `cilantro_hsv_thresholds.txt`
- Press **'q'** to quit

**Tips for Finding Good Thresholds:**
- Start with the default values (H: 35-85, S: 40-255, V: 40-255)
- Adjust **H Min/Max** first to capture all green colors
- Adjust **S Min** to filter out grayish/dull colors (background)
- Adjust **V Min** to filter out dark shadows
- The goal is a clean white mask on cilantro with minimal background noise

### Step 2: Update Threshold Values

Once you find good threshold values:

1. Open `cilantro_analyzer.py` in a text editor
2. Find the configuration section at the top:
```python
# ==============================================================================
# CONFIGURATION: Update these values from hsv_threshold_finder.py
# ==============================================================================

LOWER_GREEN = np.array([35, 40, 40])   # [H_min, S_min, V_min]
UPPER_GREEN = np.array([85, 255, 255]) # [H_max, S_max, V_max]
```

3. Replace the values with your saved threshold values from `cilantro_hsv_thresholds.txt`

### Step 3: Analyze Cilantro Freshness

Now you can analyze any cilantro image:

```bash
python cilantro_analyzer.py <path_to_cilantro_image>
```

Example:
```bash
python cilantro_analyzer.py my_cilantro.jpg
```

**Output:**
- **Console Report** with:
  - Segmentation statistics (pixel count, coverage)
  - Color analysis (mean hue, saturation, brightness)
  - Freshness score (0-100)
  - Freshness assessment (Fresh/Good/Fair/Poor)
  - Detailed factors explaining the score

- **Visual Window** showing:
  - Original image
  - HSV representation
  - Binary mask
  - Segmented result

## Understanding the Results

### Freshness Score (0-100)

The analyzer evaluates freshness based on three factors:

1. **Saturation (40 points max)**
   - High saturation = vivid, fresh green
   - Low saturation = dull, grayish (wilting or yellowing)

2. **Brightness/Value (30 points max)**
   - Optimal: Neither too dark nor washed out
   - Fresh cilantro has moderate to high brightness

3. **Hue/Color (30 points max)**
   - Should be in the green range (hue 45-75 is optimal)
   - Off-color may indicate decay

### Assessment Categories

- **80-100**: FRESH - Excellent condition
- **60-79**: GOOD - Still fresh, use soon
- **40-59**: FAIR - Starting to degrade
- **0-39**: POOR - Consider replacing

## Example Workflow

```bash
# 1. First time: Find your thresholds with a sample image
python hsv_threshold_finder.py samples/fresh_cilantro.jpg

# 2. Adjust sliders until mask looks perfect
#    Press 's' to save values
#    Press 'q' to quit

# 3. Update LOWER_GREEN and UPPER_GREEN in cilantro_analyzer.py

# 4. Analyze your cilantro
python cilantro_analyzer.py my_cilantro_photo.jpg
```

## Troubleshooting

### "No cilantro detected!"

This means the mask found no pixels matching your HSV thresholds.

**Solutions:**
1. Run `hsv_threshold_finder.py` again with your image
2. Adjust the threshold values to be more inclusive
3. Check that your image actually contains visible green cilantro

### Background is being detected as cilantro

Your thresholds are too broad.

**Solutions:**
1. Increase **S Min** (saturation minimum) to filter out dull colors
2. Narrow the **H Min/Max** range to be more specific to cilantro's green
3. Adjust **V Min** to filter out dark areas

### Cilantro is not fully detected

Your thresholds are too narrow.

**Solutions:**
1. Widen the **H Min/Max** range
2. Lower **S Min** to include less saturated greens
3. Lower **V Min** to include darker areas

## Project Structure

```
604-P3/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── hsv_threshold_finder.py           # Interactive threshold finder
├── cilantro_analyzer.py              # Main analysis script
└── cilantro_hsv_thresholds.txt       # Saved threshold values (generated)
```

## Technical Details

### HSV Color Space

OpenCV uses the following ranges:
- **Hue (H)**: 0-179 (0° to 360° mapped to 0-179)
- **Saturation (S)**: 0-255 (0% to 100% mapped to 0-255)
- **Value (V)**: 0-255 (brightness from black to full color)

### Morphological Operations

The tools use morphological operations to clean up the binary mask:
- **MORPH_CLOSE**: Fills small holes in the detected regions
- **MORPH_OPEN**: Removes small noise/speckles

This results in cleaner, more accurate segmentation.

## License

This project is open source and available for educational and research purposes.

## Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest improvements
- Add new features (e.g., batch processing, different freshness metrics)

## Future Enhancements

Possible improvements:
- Batch processing multiple images
- Save analysis reports to CSV/JSON
- Machine learning-based freshness classification
- Support for other herbs and vegetables
- Web interface for easier use
