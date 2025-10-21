#!/usr/bin/env python3
"""
HSV Threshold Finder for Cilantro Segmentation

This interactive tool helps you find the perfect HSV color threshold values
for segmenting cilantro (or any green object) from a background.

Usage:
    python hsv_threshold_finder.py <path_to_image>

Example:
    python hsv_threshold_finder.py images/cilantro_sample.jpg

Controls:
    - Use the trackbars to adjust HSV min/max values in real-time
    - Press 'q' to quit
    - Press 's' to save the current threshold values to a file
    - The tool will display 4 windows:
        1. Original image
        2. HSV image (for reference)
        3. Binary mask (white = cilantro, black = background)
        4. Segmented result (original image with background removed)
"""

import cv2
import numpy as np
import sys
import os


class HSVThresholdFinder:
    def __init__(self, image_path):
        """Initialize the threshold finder with an image."""
        # Load the image
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image from: {image_path}")

        # Resize if image is too large (for better display)
        height, width = self.image.shape[:2]
        max_dimension = 800
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            self.image = cv2.resize(self.image, (new_width, new_height))

        # Convert to HSV
        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        # Starting threshold values (good defaults for fresh green cilantro)
        # OpenCV uses: H: 0-179, S: 0-255, V: 0-255
        self.h_min = 35   # Yellowish-green
        self.h_max = 85   # Bluish-green
        self.s_min = 40   # Some saturation (not grayish)
        self.s_max = 255  # Maximum saturation
        self.v_min = 40   # Not too dark
        self.v_max = 255  # Maximum brightness

        # Window names
        self.window_name = "HSV Threshold Finder"
        self.trackbar_window = "Adjust Thresholds"

    def nothing(self, x):
        """Dummy callback for trackbars."""
        pass

    def create_trackbars(self):
        """Create trackbar window with sliders for HSV thresholds."""
        cv2.namedWindow(self.trackbar_window)

        # Create trackbars for HSV min/max values
        cv2.createTrackbar("H Min", self.trackbar_window, self.h_min, 179, self.nothing)
        cv2.createTrackbar("H Max", self.trackbar_window, self.h_max, 179, self.nothing)
        cv2.createTrackbar("S Min", self.trackbar_window, self.s_min, 255, self.nothing)
        cv2.createTrackbar("S Max", self.trackbar_window, self.s_max, 255, self.nothing)
        cv2.createTrackbar("V Min", self.trackbar_window, self.v_min, 255, self.nothing)
        cv2.createTrackbar("V Max", self.trackbar_window, self.v_max, 255, self.nothing)

    def get_current_thresholds(self):
        """Get current threshold values from trackbars."""
        self.h_min = cv2.getTrackbarPos("H Min", self.trackbar_window)
        self.h_max = cv2.getTrackbarPos("H Max", self.trackbar_window)
        self.s_min = cv2.getTrackbarPos("S Min", self.trackbar_window)
        self.s_max = cv2.getTrackbarPos("S Max", self.trackbar_window)
        self.v_min = cv2.getTrackbarPos("V Min", self.trackbar_window)
        self.v_max = cv2.getTrackbarPos("V Max", self.trackbar_window)

        return (self.h_min, self.s_min, self.v_min), (self.h_max, self.s_max, self.v_max)

    def create_mask(self, lower_bound, upper_bound):
        """Create binary mask based on HSV thresholds."""
        mask = cv2.inRange(self.hsv, lower_bound, upper_bound)

        # Optional: Apply morphological operations to clean up the mask
        # This removes small noise and fills small holes
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

    def apply_mask(self, mask):
        """Apply mask to original image to show segmented result."""
        result = cv2.bitwise_and(self.image, self.image, mask=mask)
        return result

    def save_thresholds(self):
        """Save current threshold values to a file."""
        filename = "cilantro_hsv_thresholds.txt"
        with open(filename, 'w') as f:
            f.write("# HSV Threshold Values for Cilantro Segmentation\n")
            f.write("# Use these values in your color analysis script\n\n")
            f.write(f"# Lower bound (H, S, V)\n")
            f.write(f"lower_green = [{self.h_min}, {self.s_min}, {self.v_min}]\n\n")
            f.write(f"# Upper bound (H, S, V)\n")
            f.write(f"upper_green = [{self.h_max}, {self.s_max}, {self.v_max}]\n\n")
            f.write(f"# Python tuple format (for cv2.inRange)\n")
            f.write(f"lower_green = ({self.h_min}, {self.s_min}, {self.v_min})\n")
            f.write(f"upper_green = ({self.h_max}, {self.s_max}, {self.v_max})\n")

        print(f"\n✓ Threshold values saved to: {filename}")
        print(f"  Lower bound: ({self.h_min}, {self.s_min}, {self.v_min})")
        print(f"  Upper bound: ({self.h_max}, {self.s_max}, {self.v_max})")

    def run(self):
        """Run the interactive threshold finder."""
        print("\n" + "="*60)
        print("HSV Threshold Finder for Cilantro Segmentation")
        print("="*60)
        print("\nControls:")
        print("  - Adjust the trackbars to find perfect threshold values")
        print("  - Press 's' to save the current values to a file")
        print("  - Press 'q' to quit")
        print("\nWindows:")
        print("  1. Original - Your input image")
        print("  2. HSV - HSV color space representation")
        print("  3. Mask - Binary mask (white = cilantro, black = background)")
        print("  4. Result - Segmented cilantro with background removed")
        print("="*60 + "\n")

        # Create trackbars
        self.create_trackbars()

        while True:
            # Get current threshold values
            lower_bound, upper_bound = self.get_current_thresholds()
            lower_bound = np.array(lower_bound)
            upper_bound = np.array(upper_bound)

            # Create mask and result
            mask = self.create_mask(lower_bound, upper_bound)
            result = self.apply_mask(mask)

            # Display all windows
            cv2.imshow("1. Original", self.image)
            cv2.imshow("2. HSV", self.hsv)
            cv2.imshow("3. Mask", mask)
            cv2.imshow("4. Result (Segmented)", result)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\nExiting...")
                break
            elif key == ord('s'):
                self.save_thresholds()

        # Clean up
        cv2.destroyAllWindows()


def main():
    """Main entry point."""
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python hsv_threshold_finder.py <path_to_image>")
        print("\nExample:")
        print("  python hsv_threshold_finder.py images/cilantro_sample.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)

    try:
        # Create and run the threshold finder
        finder = HSVThresholdFinder(image_path)
        finder.run()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
