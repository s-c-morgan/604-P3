#!/usr/bin/env python3
"""
Batch Cilantro Analyzer

This script processes multiple cilantro images using the same HSV threshold mask
and compiles their HSV statistics into a comparison table.

Usage:
    python batch_cilantro_analyzer.py [options]

Options:
    --dir <directory>       Directory containing images (default: images/)
    --output <filename>     Output CSV filename (default: cilantro_stats.csv)
    --save-masks           Save visualization masks for each image

Examples:
    # Process all images in images/ folder
    python batch_cilantro_analyzer.py

    # Process images from specific directory
    python batch_cilantro_analyzer.py --dir my_cilantro_photos/

    # Save output to specific file and save mask visualizations
    python batch_cilantro_analyzer.py --output results.csv --save-masks
"""

import cv2
import numpy as np
import sys
import os
import glob
import argparse
import csv
from datetime import datetime


# ==============================================================================
# CONFIGURATION: Update these values from hsv_threshold_finder.py
# ==============================================================================

# Default HSV thresholds (good for fresh green cilantro in typical lighting)
# OpenCV uses: H: 0-179, S: 0-255, V: 0-255
LOWER_GREEN = np.array([35, 40, 40])   # [H_min, S_min, V_min]
UPPER_GREEN = np.array([85, 255, 255]) # [H_max, S_max, V_max]

# ==============================================================================


class BatchCilantroAnalyzer:
    """Process multiple cilantro images and compile statistics."""

    def __init__(self, lower_bound=LOWER_GREEN, upper_bound=UPPER_GREEN):
        """
        Initialize the batch analyzer with HSV thresholds.

        Args:
            lower_bound: Lower HSV threshold (H, S, V)
            upper_bound: Upper HSV threshold (H, S, V)
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.results = []

    def process_image(self, image_path):
        """
        Process a single image and extract HSV statistics.

        Args:
            image_path: Path to the cilantro image

        Returns:
            dict with image statistics, or None if processing failed
        """
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"  ✗ Could not load: {os.path.basename(image_path)}")
            return None

        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create mask
        mask = cv2.inRange(hsv, self.lower_bound, self.upper_bound)

        # Clean up the mask with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Get cilantro pixels
        cilantro_pixels_bgr = image[mask == 255]
        cilantro_pixels_hsv = hsv[mask == 255]

        if len(cilantro_pixels_hsv) == 0:
            print(f"  ⚠ No cilantro detected in: {os.path.basename(image_path)}")
            return {
                'filename': os.path.basename(image_path),
                'pixel_count': 0,
                'coverage_percent': 0.0,
                'mean_hue': 0.0,
                'std_hue': 0.0,
                'mean_saturation': 0.0,
                'std_saturation': 0.0,
                'mean_value': 0.0,
                'std_value': 0.0,
                'mean_blue': 0.0,
                'mean_green': 0.0,
                'mean_red': 0.0,
                'error': 'No cilantro detected'
            }

        # Calculate statistics
        total_pixels = image.shape[0] * image.shape[1]
        cilantro_count = len(cilantro_pixels_hsv)
        coverage = (cilantro_count / total_pixels) * 100

        # HSV statistics
        mean_hsv = np.mean(cilantro_pixels_hsv, axis=0)
        std_hsv = np.std(cilantro_pixels_hsv, axis=0)

        # BGR statistics
        mean_bgr = np.mean(cilantro_pixels_bgr, axis=0)

        print(f"  ✓ Processed: {os.path.basename(image_path)} ({cilantro_count:,} pixels, {coverage:.2f}% coverage)")

        return {
            'filename': os.path.basename(image_path),
            'pixel_count': cilantro_count,
            'coverage_percent': coverage,
            'mean_hue': mean_hsv[0],
            'std_hue': std_hsv[0],
            'mean_saturation': mean_hsv[1],
            'std_saturation': std_hsv[1],
            'mean_value': mean_hsv[2],
            'std_value': std_hsv[2],
            'mean_blue': mean_bgr[0],
            'mean_green': mean_bgr[1],
            'mean_red': mean_bgr[2],
            'image_width': image.shape[1],
            'image_height': image.shape[0],
            'mask': mask,
            'image': image
        }

    def process_directory(self, directory, save_masks=False):
        """
        Process all images in a directory.

        Args:
            directory: Path to directory containing images
            save_masks: If True, save visualization masks

        Returns:
            list of result dictionaries
        """
        # Supported image formats
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']

        # Find all images
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(directory, ext)))
            image_files.extend(glob.glob(os.path.join(directory, ext.upper())))

        if not image_files:
            print(f"\n✗ No images found in: {directory}")
            return []

        print(f"\nFound {len(image_files)} image(s) in {directory}")
        print("="*70)

        # Process each image
        results = []
        for image_path in sorted(image_files):
            result = self.process_image(image_path)
            if result:
                results.append(result)

                # Save mask visualization if requested
                if save_masks and 'mask' in result and 'image' in result:
                    self.save_mask_visualization(result, directory)

        self.results = results
        return results

    def save_mask_visualization(self, result, output_dir):
        """
        Save a visualization showing original, mask, and segmented result.

        Args:
            result: Result dictionary from process_image
            output_dir: Directory to save visualization
        """
        if result['pixel_count'] == 0:
            return  # Skip if no cilantro detected

        image = result['image']
        mask = result['mask']
        filename = result['filename']

        # Create segmented result
        segmented = cv2.bitwise_and(image, image, mask=mask)

        # Resize for display (max width 400px each)
        h, w = image.shape[:2]
        display_width = 400
        display_height = int(h * (display_width / w))

        original = cv2.resize(image, (display_width, display_height))
        mask_vis = cv2.resize(mask, (display_width, display_height))
        mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
        segmented = cv2.resize(segmented, (display_width, display_height))

        # Stack horizontally
        visualization = np.hstack([original, mask_vis, segmented])

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(visualization, "Original", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(visualization, "Mask", (display_width + 10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(visualization, "Segmented", (display_width * 2 + 10, 30), font, 0.7, (255, 255, 255), 2)

        # Save
        output_filename = os.path.splitext(filename)[0] + "_mask.jpg"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, visualization)
        print(f"    Saved mask: {output_filename}")

    def save_to_csv(self, output_file):
        """
        Save results to a CSV file.

        Args:
            output_file: Path to output CSV file
        """
        if not self.results:
            print("\n✗ No results to save!")
            return

        # Define CSV columns
        fieldnames = [
            'filename',
            'pixel_count',
            'coverage_percent',
            'mean_hue',
            'std_hue',
            'mean_saturation',
            'std_saturation',
            'mean_value',
            'std_value',
            'mean_blue',
            'mean_green',
            'mean_red',
            'image_width',
            'image_height'
        ]

        # Write CSV
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for result in self.results:
                # Filter out non-CSV fields (mask, image)
                csv_result = {k: v for k, v in result.items() if k in fieldnames}
                writer.writerow(csv_result)

        print(f"\n✓ Results saved to: {output_file}")

    def print_summary_table(self):
        """Print a summary table to console."""
        if not self.results:
            print("\n✗ No results to display!")
            return

        print("\n" + "="*120)
        print("CILANTRO HSV STATISTICS SUMMARY")
        print("="*120)
        print(f"HSV Thresholds: {self.lower_bound} to {self.upper_bound}")
        print(f"Total images processed: {len(self.results)}")
        print("="*120)

        # Print header
        print(f"\n{'Filename':<30} {'Pixels':>10} {'Coverage':>9} {'H (μ)':>8} {'H (σ)':>8} {'S (μ)':>8} {'S (σ)':>8} {'V (μ)':>8} {'V (σ)':>8}")
        print("-"*120)

        # Print each result
        for result in self.results:
            if result.get('error'):
                print(f"{result['filename']:<30} {'N/A':>10} {'N/A':>9} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
            else:
                print(f"{result['filename']:<30} {result['pixel_count']:>10,} {result['coverage_percent']:>8.2f}% "
                      f"{result['mean_hue']:>8.1f} {result['std_hue']:>8.1f} "
                      f"{result['mean_saturation']:>8.1f} {result['std_saturation']:>8.1f} "
                      f"{result['mean_value']:>8.1f} {result['std_value']:>8.1f}")

        print("="*120)
        print("\nLegend:")
        print("  H (μ) = Mean Hue (0-179)")
        print("  H (σ) = Std Dev Hue")
        print("  S (μ) = Mean Saturation (0-255)")
        print("  S (σ) = Std Dev Saturation")
        print("  V (μ) = Mean Value/Brightness (0-255)")
        print("  V (σ) = Std Dev Value")
        print()


def main():
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Batch process cilantro images and compile HSV statistics'
    )
    parser.add_argument('--dir', default='images',
                        help='Directory containing images (default: images/)')
    parser.add_argument('--output', default='cilantro_stats.csv',
                        help='Output CSV filename (default: cilantro_stats.csv)')
    parser.add_argument('--save-masks', action='store_true',
                        help='Save mask visualizations for each image')

    args = parser.parse_args()

    # Check if directory exists
    if not os.path.isdir(args.dir):
        print(f"✗ Error: Directory not found: {args.dir}")
        sys.exit(1)

    print("="*70)
    print("BATCH CILANTRO ANALYZER")
    print("="*70)
    print(f"Input directory: {args.dir}")
    print(f"Output CSV: {args.output}")
    print(f"Save masks: {args.save_masks}")
    print(f"HSV Thresholds: {LOWER_GREEN} to {UPPER_GREEN}")

    # Create analyzer
    analyzer = BatchCilantroAnalyzer()

    # Process all images
    results = analyzer.process_directory(args.dir, save_masks=args.save_masks)

    if results:
        # Print summary table
        analyzer.print_summary_table()

        # Save to CSV
        analyzer.save_to_csv(args.output)

        print(f"\n✓ Processing complete! {len(results)} image(s) analyzed.")
    else:
        print("\n✗ No images were successfully processed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
