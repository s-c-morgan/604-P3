#!/usr/bin/env python3
"""
Cilantro Freshness Analyzer

This script uses color-based HSV thresholding to segment cilantro from a background
and analyze the color properties to determine freshness.

Usage:
    python cilantro_analyzer.py <path_to_image>

Example:
    python cilantro_analyzer.py images/my_cilantro.jpg

Prerequisites:
    1. Run hsv_threshold_finder.py first to find your optimal threshold values
    2. Update the LOWER_GREEN and UPPER_GREEN values below with your findings
"""

import cv2
import numpy as np
import sys
import os


# ==============================================================================
# CONFIGURATION: Update these values from hsv_threshold_finder.py
# ==============================================================================

# Default starting values (good for fresh green cilantro in typical lighting)
# OpenCV uses: H: 0-179, S: 0-255, V: 0-255
LOWER_GREEN = np.array([35, 40, 40])   # [H_min, S_min, V_min]
UPPER_GREEN = np.array([85, 255, 255]) # [H_max, S_max, V_max]

# ==============================================================================


class CilantroAnalyzer:
    """Analyze cilantro freshness based on color segmentation."""

    def __init__(self, image_path, lower_bound=LOWER_GREEN, upper_bound=UPPER_GREEN):
        """
        Initialize the analyzer with an image and HSV thresholds.

        Args:
            image_path: Path to the cilantro image
            lower_bound: Lower HSV threshold (H, S, V)
            upper_bound: Upper HSV threshold (H, S, V)
        """
        # Load the image
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image from: {image_path}")

        self.image_path = image_path
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # Convert to HSV
        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        # Create the mask
        self.mask = self._create_mask()

        # Apply the mask to get segmented cilantro
        self.segmented = self._apply_mask()

    def _create_mask(self):
        """Create binary mask based on HSV thresholds."""
        # Create initial mask
        mask = cv2.inRange(self.hsv, self.lower_bound, self.upper_bound)

        # Clean up the mask with morphological operations
        # This removes noise and fills small holes
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

    def _apply_mask(self):
        """Apply mask to original image."""
        return cv2.bitwise_and(self.image, self.image, mask=self.mask)

    def get_cilantro_pixels(self):
        """
        Extract only the cilantro pixels (where mask is white).

        Returns:
            numpy array of shape (N, 3) where N is number of cilantro pixels
            and each row is [B, G, R] color values
        """
        # Get pixels where mask is white (255)
        cilantro_pixels = self.image[self.mask == 255]
        return cilantro_pixels

    def calculate_statistics(self):
        """
        Calculate color statistics for the segmented cilantro.

        Returns:
            dict with color statistics
        """
        cilantro_pixels = self.get_cilantro_pixels()

        if len(cilantro_pixels) == 0:
            return {
                'error': 'No cilantro detected! Try adjusting your HSV thresholds.',
                'pixel_count': 0
            }

        # Calculate statistics in BGR color space
        mean_bgr = np.mean(cilantro_pixels, axis=0)
        std_bgr = np.std(cilantro_pixels, axis=0)

        # Calculate statistics in HSV color space
        cilantro_hsv = self.hsv[self.mask == 255]
        mean_hsv = np.mean(cilantro_hsv, axis=0)
        std_hsv = np.std(cilantro_hsv, axis=0)

        # Calculate coverage (percentage of image that is cilantro)
        total_pixels = self.image.shape[0] * self.image.shape[1]
        cilantro_pixels_count = len(cilantro_pixels)
        coverage_percent = (cilantro_pixels_count / total_pixels) * 100

        return {
            'pixel_count': cilantro_pixels_count,
            'coverage_percent': coverage_percent,
            'mean_bgr': mean_bgr,
            'std_bgr': std_bgr,
            'mean_hsv': mean_hsv,
            'std_hsv': std_hsv,
            'mean_hue': mean_hsv[0],
            'mean_saturation': mean_hsv[1],
            'mean_value': mean_hsv[2],
        }

    def assess_freshness(self):
        """
        Assess cilantro freshness based on color properties.

        Fresh cilantro typically has:
        - High saturation (vivid green, not dull/grayish)
        - Moderate to high value (not too dark)
        - Hue in the green range (40-80 in OpenCV's 0-179 scale)

        Returns:
            dict with freshness assessment
        """
        stats = self.calculate_statistics()

        if 'error' in stats:
            return stats

        mean_hue = stats['mean_hue']
        mean_sat = stats['mean_saturation']
        mean_val = stats['mean_value']

        # Freshness scoring (0-100)
        freshness_score = 0
        factors = []

        # Factor 1: Saturation (40 points max)
        # Fresh cilantro has high saturation (>100 is good, >150 is excellent)
        if mean_sat > 150:
            sat_score = 40
            sat_desc = "excellent"
        elif mean_sat > 100:
            sat_score = 30
            sat_desc = "good"
        elif mean_sat > 60:
            sat_score = 20
            sat_desc = "moderate"
        else:
            sat_score = 10
            sat_desc = "low (possibly wilted or yellowing)"

        freshness_score += sat_score
        factors.append(f"Saturation: {sat_desc} ({mean_sat:.1f}/255)")

        # Factor 2: Value/Brightness (30 points max)
        # Fresh cilantro is neither too dark nor washed out
        if 80 <= mean_val <= 200:
            val_score = 30
            val_desc = "optimal"
        elif 50 <= mean_val < 80 or 200 < mean_val <= 230:
            val_score = 20
            val_desc = "acceptable"
        else:
            val_score = 10
            val_desc = "suboptimal"

        freshness_score += val_score
        factors.append(f"Brightness: {val_desc} ({mean_val:.1f}/255)")

        # Factor 3: Hue (30 points max)
        # Fresh cilantro should be in the green range
        if 45 <= mean_hue <= 75:
            hue_score = 30
            hue_desc = "vibrant green"
        elif 35 <= mean_hue < 45 or 75 < mean_hue <= 85:
            hue_score = 20
            hue_desc = "acceptable green"
        else:
            hue_score = 10
            hue_desc = "off-color (possibly decaying)"

        freshness_score += hue_score
        factors.append(f"Color: {hue_desc} (hue: {mean_hue:.1f}/179)")

        # Overall assessment
        if freshness_score >= 80:
            assessment = "FRESH - Excellent condition"
        elif freshness_score >= 60:
            assessment = "GOOD - Still fresh, use soon"
        elif freshness_score >= 40:
            assessment = "FAIR - Starting to degrade"
        else:
            assessment = "POOR - Consider replacing"

        return {
            'freshness_score': freshness_score,
            'assessment': assessment,
            'factors': factors,
            'statistics': stats
        }

    def visualize(self, show=True, save_path=None):
        """
        Create a visualization showing the analysis results.

        Args:
            show: If True, display the visualization in a window
            save_path: If provided, save the visualization to this path
        """
        # Create a 2x2 grid of images
        # Resize images if needed for consistent display
        h, w = self.image.shape[:2]
        display_width = 400
        display_height = int(h * (display_width / w))

        original = cv2.resize(self.image, (display_width, display_height))
        hsv_vis = cv2.resize(self.hsv, (display_width, display_height))
        mask_vis = cv2.resize(self.mask, (display_width, display_height))
        mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel
        segmented = cv2.resize(self.segmented, (display_width, display_height))

        # Create top and bottom rows
        top_row = np.hstack([original, hsv_vis])
        bottom_row = np.hstack([mask_vis, segmented])

        # Combine into final visualization
        visualization = np.vstack([top_row, bottom_row])

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(visualization, "Original", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(visualization, "HSV", (display_width + 10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(visualization, "Mask", (10, display_height + 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(visualization, "Segmented", (display_width + 10, display_height + 30), font, 0.7, (255, 255, 255), 2)

        if save_path:
            cv2.imwrite(save_path, visualization)
            print(f"Visualization saved to: {save_path}")

        if show:
            cv2.imshow("Cilantro Analysis", visualization)
            print("\nPress any key to close the visualization window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return visualization

    def print_report(self):
        """Print a detailed analysis report to the console."""
        freshness = self.assess_freshness()

        print("\n" + "="*70)
        print("CILANTRO FRESHNESS ANALYSIS REPORT")
        print("="*70)
        print(f"\nImage: {os.path.basename(self.image_path)}")
        print(f"HSV Thresholds: {self.lower_bound} to {self.upper_bound}")

        if 'error' in freshness:
            print(f"\nERROR: {freshness['error']}")
            print("\nSuggestions:")
            print("  1. Run hsv_threshold_finder.py to find correct threshold values")
            print("  2. Update LOWER_GREEN and UPPER_GREEN in this script")
            print("  3. Ensure your image contains visible cilantro")
            print("="*70 + "\n")
            return

        stats = freshness['statistics']

        print("\n--- Segmentation Results ---")
        print(f"Cilantro pixels detected: {stats['pixel_count']:,}")
        print(f"Coverage of image: {stats['coverage_percent']:.2f}%")

        print("\n--- Color Analysis ---")
        print(f"Mean Hue: {stats['mean_hue']:.1f} / 179")
        print(f"Mean Saturation: {stats['mean_saturation']:.1f} / 255")
        print(f"Mean Value (Brightness): {stats['mean_value']:.1f} / 255")

        print("\n--- Freshness Assessment ---")
        print(f"Overall Score: {freshness['freshness_score']}/100")
        print(f"Assessment: {freshness['assessment']}")

        print("\nFactors:")
        for factor in freshness['factors']:
            print(f"  • {factor}")

        print("\n" + "="*70 + "\n")


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python cilantro_analyzer.py <path_to_image>")
        print("\nExample:")
        print("  python cilantro_analyzer.py images/my_cilantro.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)

    try:
        # Create analyzer
        analyzer = CilantroAnalyzer(image_path)

        # Print analysis report
        analyzer.print_report()

        # Show visualization
        analyzer.visualize(show=True)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
