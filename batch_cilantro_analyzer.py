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
        Process a single image and extract per-sprig HSV statistics.

        Args:
            image_path: Path to the cilantro image

        Returns:
            dict with per-sprig statistics, or None if processing failed
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

        # Find contours to identify individual sprigs
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            print(f"  ⚠ No cilantro detected in: {os.path.basename(image_path)}")
            return {
                'filename': os.path.basename(image_path),
                'num_sprigs': 0,
                'sprigs': [],
                'mask': mask,
                'image': image,
                'contours': []
            }

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
        if len(contours) == 0:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:target_sprigs]

        # Process each sprig
        sprig_stats = []
        for idx, contour in enumerate(contours):
            # Create mask for this sprig only
            sprig_mask = np.zeros_like(mask)
            cv2.drawContours(sprig_mask, [contour], -1, 255, -1)

            # Get pixels for this sprig
            sprig_pixels_hsv = hsv[sprig_mask == 255]
            sprig_pixels_bgr = image[sprig_mask == 255]

            if len(sprig_pixels_hsv) == 0:
                continue

            # Calculate statistics for this sprig
            area = cv2.contourArea(contour)
            mean_hsv = np.mean(sprig_pixels_hsv, axis=0)
            std_hsv = np.std(sprig_pixels_hsv, axis=0)
            mean_bgr = np.mean(sprig_pixels_bgr, axis=0)

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            sprig_stats.append({
                'sprig_id': idx + 1,
                'area': int(area),
                'mean_hue': float(mean_hsv[0]),
                'std_hue': float(std_hsv[0]),
                'mean_saturation': float(mean_hsv[1]),
                'std_saturation': float(std_hsv[1]),
                'mean_value': float(mean_hsv[2]),
                'std_value': float(std_hsv[2]),
                'mean_blue': float(mean_bgr[0]),
                'mean_green': float(mean_bgr[1]),
                'mean_red': float(mean_bgr[2]),
                'bbox_x': x,
                'bbox_y': y,
                'bbox_width': w,
                'bbox_height': h
            })

        total_area = sum(s['area'] for s in sprig_stats)
        total_pixels = image.shape[0] * image.shape[1]
        coverage = (total_area / total_pixels) * 100

        # Report status
        if len(sprig_stats) < 5:
            print(f"  ⚠ Only {len(sprig_stats)} sprigs found in: {os.path.basename(image_path)}")
        elif len(sprig_stats) == 5:
            print(f"  ✓ Processed: {os.path.basename(image_path)} (5 sprigs, {total_area:,} pixels, {coverage:.2f}% coverage)")
        else:
            print(f"  ✓ Processed: {os.path.basename(image_path)} ({len(sprig_stats)} sprigs, {total_area:,} pixels, {coverage:.2f}% coverage)")

        return {
            'filename': os.path.basename(image_path),
            'num_sprigs': len(sprig_stats),
            'total_area': total_area,
            'coverage_percent': coverage,
            'sprigs': sprig_stats,
            'image_width': image.shape[1],
            'image_height': image.shape[0],
            'mask': mask,
            'image': image,
            'contours': contours
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
        Save a visualization showing original, mask, and individual sprigs.

        Args:
            result: Result dictionary from process_image
            output_dir: Directory to save visualization
        """
        if result['num_sprigs'] == 0:
            return  # Skip if no cilantro detected

        image = result['image']
        mask = result['mask']
        contours = result['contours']
        filename = result['filename']

        # Create segmented result
        segmented = cv2.bitwise_and(image, image, mask=mask)

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

        # Resize for display (max width 400px each)
        h, w = image.shape[:2]
        display_width = 400
        display_height = int(h * (display_width / w))

        original = cv2.resize(image, (display_width, display_height))
        mask_vis = cv2.resize(mask, (display_width, display_height))
        mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
        sprigs_labeled = cv2.resize(sprigs_labeled, (display_width, display_height))

        # Stack horizontally
        visualization = np.hstack([original, mask_vis, sprigs_labeled])

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(visualization, "Original", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(visualization, "Mask", (display_width + 10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(visualization, f"Sprigs ({result['num_sprigs']})", (display_width * 2 + 10, 30), font, 0.7, (255, 255, 255), 2)

        # Save
        output_filename = os.path.splitext(filename)[0] + "_sprigs.jpg"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, visualization)
        print(f"    Saved visualization: {output_filename}")

    def save_to_csv(self, output_file):
        """
        Save per-sprig results to a CSV file.

        Args:
            output_file: Path to output CSV file
        """
        if not self.results:
            print("\n✗ No results to save!")
            return

        # Define CSV columns for per-sprig data
        fieldnames = [
            'filename',
            'sprig_id',
            'area',
            'mean_hue',
            'std_hue',
            'mean_saturation',
            'std_saturation',
            'mean_value',
            'std_value',
            'mean_blue',
            'mean_green',
            'mean_red',
            'bbox_x',
            'bbox_y',
            'bbox_width',
            'bbox_height'
        ]

        # Write CSV with one row per sprig
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for result in self.results:
                filename = result['filename']
                for sprig in result['sprigs']:
                    row = {'filename': filename}
                    row.update(sprig)
                    writer.writerow(row)

        print(f"\n✓ Results saved to: {output_file}")

    def print_summary_table(self):
        """Print a per-sprig summary table to console."""
        if not self.results:
            print("\n✗ No results to display!")
            return

        print("\n" + "="*140)
        print("CILANTRO PER-SPRIG HSV STATISTICS SUMMARY")
        print("="*140)
        print(f"HSV Thresholds: {self.lower_bound} to {self.upper_bound}")
        print(f"Total images processed: {len(self.results)}")
        total_sprigs = sum(r['num_sprigs'] for r in self.results)
        print(f"Total sprigs detected: {total_sprigs}")
        print("="*140)

        # Print header
        print(f"\n{'Filename':<30} {'Sprig':>6} {'Area':>8} {'H (μ)':>8} {'H (σ)':>8} {'S (μ)':>8} {'S (σ)':>8} {'V (μ)':>8} {'V (σ)':>8}")
        print("-"*140)

        # Print each result
        for result in self.results:
            if result['num_sprigs'] == 0:
                print(f"{result['filename']:<30} {'N/A':>6} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
            else:
                for sprig in result['sprigs']:
                    print(f"{result['filename']:<30} {sprig['sprig_id']:>6} {sprig['area']:>8,} "
                          f"{sprig['mean_hue']:>8.1f} {sprig['std_hue']:>8.1f} "
                          f"{sprig['mean_saturation']:>8.1f} {sprig['std_saturation']:>8.1f} "
                          f"{sprig['mean_value']:>8.1f} {sprig['std_value']:>8.1f}")

        print("="*140)
        print("\nLegend:")
        print("  Sprig = Sprig ID (sorted by size, largest first)")
        print("  Area = Pixel count for this sprig")
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
