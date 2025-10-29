#!/usr/bin/env python3
"""
Create a segmentation example showing individual sprig detection for the preanalysis plan.
Shows original image, mask, and individual labeled sprigs.
"""

import cv2
import numpy as np

# HSV thresholds from the analyzer
LOWER_GREEN = np.array([35, 40, 40])
UPPER_GREEN = np.array([85, 255, 255])

# Load the sample image
image_path = 'images/cilantro.jpg'
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not load {image_path}")
    exit(1)

# Convert to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Create mask
mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

# Find contours to identify individual sprigs
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iteratively relax min_area threshold until we find exactly 5 sprigs
min_area = 10000  # Start with large threshold
target_sprigs = 5

while min_area >= 10:  # Don't go below 10 pixels
    filtered_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)

    if len(filtered_contours) >= target_sprigs:
        # Found at least 5 sprigs, take the largest 5
        contours = filtered_contours[:target_sprigs]
        break
    elif len(filtered_contours) > 0 and min_area <= 100:
        # If we're below 100 pixels and have some contours, use what we have
        contours = filtered_contours
        break

    # Reduce threshold and try again
    min_area = int(min_area * 0.7)  # Reduce by 30% each iteration

# Final fallback: just take the largest contours available
if len(contours) == 0 or len(contours) > target_sprigs:
    all_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = all_contours[:target_sprigs]

# Create image with numbered sprigs
sprigs_labeled = image.copy()
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
          (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 128, 0)]

for idx, contour in enumerate(contours):
    color = colors[idx % len(colors)]
    # Draw contour outline
    cv2.drawContours(sprigs_labeled, [contour], -1, color, 3)
    # Add sprig number
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.putText(sprigs_labeled, str(idx + 1), (cx, cy),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

# Resize for display
h, w = image.shape[:2]
display_width = 350
display_height = int(h * (display_width / w))

original_resized = cv2.resize(image, (display_width, display_height))
mask_resized = cv2.resize(mask, (display_width, display_height))
mask_resized_bgr = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
sprigs_resized = cv2.resize(sprigs_labeled, (display_width, display_height))

# Add labels to images
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
font_color = (255, 255, 255)
font_thickness = 2
bg_color = (0, 0, 0)

def add_label(img, text, position=(10, 35)):
    # Add background rectangle for text
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    cv2.rectangle(img, (position[0]-5, position[1]-text_height-5),
                  (position[0]+text_width+5, position[1]+5), bg_color, -1)
    cv2.putText(img, text, position, font, font_scale, font_color, font_thickness)

add_label(original_resized, "Original")
add_label(mask_resized_bgr, "Binary Mask")
add_label(sprigs_resized, f"Sprigs ({len(contours)})")

# Stack horizontally
segmentation_example = np.hstack([original_resized, mask_resized_bgr, sprigs_resized])

# Save
cv2.imwrite('images/segmentation_example.jpg', segmentation_example)

print("✓ Created segmentation example image:")
print(f"  - images/segmentation_example.jpg ({len(contours)} sprigs detected)")
