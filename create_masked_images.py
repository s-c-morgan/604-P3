#!/usr/bin/env python3
"""
Create masked versions of all cilantro images.
"""

import cv2
import numpy as np
import os
import glob

# HSV thresholds (adjusted for darker/wilted cilantro)
LOWER_GREEN = np.array([35, 30, 20])
UPPER_GREEN = np.array([85, 255, 255])

# Input and output directories
input_dir = 'data/images'
output_dir = 'data/images/masked_images'

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Find all jpg files
image_files = glob.glob(os.path.join(input_dir, '*.jpg'))
image_files = sorted([f for f in image_files if 'masked_images' not in f])

print(f"Processing {len(image_files)} images...")
print("="*70)

for image_path in image_files:
    filename = os.path.basename(image_path)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"✗ Could not load: {filename}")
        continue

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create mask
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

    # Apply mask to image
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Create visualization (original | mask | masked)
    h, w = image.shape[:2]
    display_width = 400
    display_height = int(h * (display_width / w))

    original = cv2.resize(image, (display_width, display_height))
    mask_vis = cv2.resize(mask, (display_width, display_height))
    mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
    masked_vis = cv2.resize(masked_image, (display_width, display_height))

    # Stack horizontally
    visualization = np.hstack([original, mask_vis, masked_vis])

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(visualization, "Original", (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(visualization, "Binary Mask", (display_width + 10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(visualization, "Thresholded", (display_width * 2 + 10, 30), font, 0.7, (255, 255, 255), 2)

    # Save
    output_filename = os.path.splitext(filename)[0] + "_masked.jpg"
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, visualization)

    print(f"✓ Created: {output_filename}")

print("="*70)
print(f"✓ All masked images saved to: {output_dir}")
