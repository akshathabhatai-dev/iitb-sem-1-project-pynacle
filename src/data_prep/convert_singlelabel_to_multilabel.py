"""
Convert Single-Label to Multi-Label CSV Format

Converts standard 64-column labels.csv format to 196-column labels_multilabel.csv format.

Input format (64 columns):
    ImageFileName, c01, c02, ..., c64
    Where each cell has ONE label: 0=No Object, 1=Ball, 2=Bat, 3=Stump

Output format (196 columns):
    ImageFileName, c01_ball, c01_bat, c01_stump, c02_ball, c02_bat, c02_stump, ..., c64_stump
    Where each cell has THREE binary flags (0 or 1) for Ball, Bat, Stump

Conversion Logic:
    - If cell=0 (No Object) → ball=0, bat=0, stump=0
    - If cell=1 (Ball) → ball=1, bat=0, stump=0
    - If cell=2 (Bat) → ball=0, bat=1, stump=0
    - If cell=3 (Stump) → ball=0, bat=0, stump=1

Note: This assumes cells contain only ONE dominant object.
For true multi-object annotations, use annotator_multilabel.py to manually annotate.

Usage:
    python convert_singlelabel_to_multilabel.py --input labels.csv --output labels_multilabel.csv
"""

import pandas as pd
import argparse
from pathlib import Path


def convert_singlelabel_to_multilabel(input_csv_path, output_csv_path):
    """
    Convert single-label CSV to multi-label CSV format.

    Args:
        input_csv_path: Path to input labels.csv (64 columns)
        output_csv_path: Path to output labels_multilabel.csv (196 columns)

    Returns:
        tuple: (num_images_converted, label_distribution)
    """
    # Read input CSV
    input_df = pd.read_csv(input_csv_path)

    print(f"Reading input CSV: {input_csv_path}")
    print(f"Found {len(input_df)} images")

    # Prepare output data
    output_rows = []
    label_distribution = {0: 0, 1: 0, 2: 0, 3: 0}  # Count of each label type

    for _, row in input_df.iterrows():
        img_name = row['ImageFileName']
        output_row = {'ImageFileName': img_name}

        # Process each cell (c01 to c64)
        for cell_idx in range(1, 65):
            cell_col = f'c{cell_idx:02d}'
            single_label = int(row[cell_col])

            # Count label distribution
            label_distribution[single_label] += 1

            # Convert to binary flags
            ball_flag = 1 if single_label == 1 else 0
            bat_flag = 1 if single_label == 2 else 0
            stump_flag = 1 if single_label == 3 else 0

            # Add to output row
            output_row[f'{cell_col}_ball'] = ball_flag
            output_row[f'{cell_col}_bat'] = bat_flag
            output_row[f'{cell_col}_stump'] = stump_flag

        output_rows.append(output_row)

    # Create output DataFrame
    output_df = pd.DataFrame(output_rows)

    # Ensure correct column order
    columns = ['ImageFileName']
    for cell_idx in range(1, 65):
        cell_id = f'c{cell_idx:02d}'
        columns.extend([f'{cell_id}_ball', f'{cell_id}_bat', f'{cell_id}_stump'])

    output_df = output_df[columns]

    # Save to CSV
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv_path, index=False)

    print(f"\nConversion complete!")
    print(f"Output saved to: {output_csv_path}")
    print(f"Output shape: {output_df.shape} (196 columns)")

    return len(output_df), label_distribution


def print_label_statistics(label_distribution, num_images):
    """
    Print statistics about label distribution.

    Args:
        label_distribution: Dict with counts of each label type
        num_images: Total number of images processed
    """
    total_cells = sum(label_distribution.values())

    print("\nLabel Distribution Statistics:")
    print("="*60)
    print(f"{'Label':<15} {'Count':>10} {'Percentage':>15} {'Per Image':>15}")
    print("-"*60)

    label_names = {0: 'No Object', 1: 'Ball', 2: 'Bat', 3: 'Stump'}

    for label, count in sorted(label_distribution.items()):
        percentage = (count / total_cells) * 100
        per_image = count / num_images
        print(f"{label_names[label]:<15} {count:>10} {percentage:>14.2f}% {per_image:>14.2f}")

    print("-"*60)
    print(f"{'Total Cells':<15} {total_cells:>10}")
    print("="*60)


def main():
    """Main function to run the conversion."""
    parser = argparse.ArgumentParser(
        description='Convert single-label CSV to multi-label CSV format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert training labels
    python convert_singlelabel_to_multilabel.py \\
        --input ../../dataset/train/labels/labels.csv \\
        --output ../../dataset/train/labels/labels_multilabel.csv

    # Convert validation labels
    python convert_singlelabel_to_multilabel.py \\
        --input ../../dataset/validate/labels/labels.csv \\
        --output ../../dataset/validate/labels/labels_multilabel.csv

    # Convert test labels
    python convert_singlelabel_to_multilabel.py \\
        --input ../../dataset/test/labels/labels.csv \\
        --output ../../dataset/test/labels/labels_multilabel.csv
        """
    )

    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Path to input single-label CSV file (64 columns)')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Path to output multi-label CSV file (196 columns)')
    parser.add_argument('--stats', action='store_true',
                       help='Print detailed label statistics')

    args = parser.parse_args()

    # Validate input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    # Perform conversion
    try:
        num_images, label_dist = convert_singlelabel_to_multilabel(
            args.input,
            args.output
        )

        # Print statistics if requested
        if args.stats:
            print_label_statistics(label_dist, num_images)

        print("\n✓ Conversion successful!")
        print("\nNote: This conversion assumes each cell contains only ONE object.")
        print("For cells with multiple objects, use annotator_multilabel.py for manual annotation.")

        return 0

    except Exception as e:
        print(f"\nError during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
