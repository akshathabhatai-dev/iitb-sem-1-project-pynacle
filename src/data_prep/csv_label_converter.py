"""
CSV Label Converter for Cricket Object Detection Project

Converts row-column-tag format to cell-based label format for 8x8 grid.

Source format: imagename, cell_1_row, cell_1_column, cell_1_object_tag, ...
Target format: ImageFileName, c01, c02, c03, ..., c64

Object mapping:
- Ball → 1
- Bat → 2
- Stump → 3
- No object → 0

Cell numbering (8x8 grid = 64 cells):
- c01 = row 1, col 1
- c02 = row 1, col 2
- ...
- c08 = row 1, col 8
- c09 = row 2, col 1
- ...
- c64 = row 8, col 8

"""

import argparse
import csv
import os
import sys


def row_col_to_cell_number(row, col):
    """
    Convert (row, column) position to cell number (1-64).

    Args:
        row: Row number (1-8)
        col: Column number (1-8)

    Returns:
        Cell number (1-64) or None if invalid
    """
    if not (1 <= row <= 8 and 1 <= col <= 8):
        return None
    return (row - 1) * 8 + col


def object_tag_to_numeric(tag):
    """
    Convert object tag string to numeric label.

    Args:
        tag: Object tag string ("Ball", "Bat", "Stump", or empty)

    Returns:
        Numeric label (0-3)
    """
    if not tag or tag.strip() == "":
        return 0

    tag_lower = tag.strip().lower()
    mapping = {
        "ball": 1,
        "bat": 2,
        "stump": 3
    }
    return mapping.get(tag_lower, 0)


def convert_labels(input_csv, output_csv):
    """
    Convert row-column-tag format CSV to cell-based labels.csv format.

    Args:
        input_csv: Path to source CSV with row/column/tag format
        output_csv: Path to output labels.csv
    """
    print(f"Reading from: {input_csv}")
    print(f"Writing to: {output_csv}")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Read input CSV
    converted_count = 0
    skipped_count = 0

    with open(input_csv, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)

        # Prepare output data
        output_rows = []

        for row_idx, row in enumerate(reader, start=1):
            image_name = row.get('imagename', '').strip()

            if not image_name:
                skipped_count += 1
                print(f"Warning: Row {row_idx} has no image name, skipping")
                continue

            # Initialize all 64 cells to 0 (no object)
            cell_labels = [0] * 64

            # Process each cell in the source format
            # Source has: cell_1_row, cell_1_column, cell_1_object_tag, cell_2_row, ...
            cell_num = 1
            while True:
                row_key = f'cell_{cell_num}_row'
                col_key = f'cell_{cell_num}_column'
                tag_key = f'cell_{cell_num}_object_tag'

                # Check if these keys exist
                if row_key not in row:
                    break

                # Get values
                row_val = row[row_key].strip()
                col_val = row[col_key].strip()
                tag_val = row[tag_key].strip()

                # If we have valid row/column/tag, process it
                if row_val and col_val and tag_val:
                    try:
                        grid_row = int(row_val)
                        grid_col = int(col_val)
                        cell_index = row_col_to_cell_number(grid_row, grid_col)

                        if cell_index is None:
                            print(f"Warning: Invalid row/col ({grid_row}, {grid_col}) in {image_name}, cell {cell_num}")
                        else:
                            numeric_label = object_tag_to_numeric(tag_val)
                            # Convert to 0-indexed for array
                            cell_labels[cell_index - 1] = numeric_label
                    except ValueError as e:
                        print(f"Warning: Could not parse row/col for {image_name}, cell {cell_num}: {e}")

                cell_num += 1

            # Create output row
            output_row = {
                'ImageFileName': image_name
            }
            for i in range(64):
                cell_name = f'c{i+1:02d}'  # c01, c02, ..., c64
                output_row[cell_name] = cell_labels[i]

            output_rows.append(output_row)
            converted_count += 1

            # Progress reporting
            if converted_count % 50 == 0:
                print(f"Processed {converted_count} images...")

    # Write output CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
        # Create column names
        fieldnames = ['ImageFileName'] + [f'c{i+1:02d}' for i in range(64)]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\nConversion complete!")
    print(f"Successfully converted: {converted_count} images")
    print(f"Skipped: {skipped_count} rows")
    print(f"Output saved to: {output_csv}")


def main():
    """Main function to run the conversion."""
    parser = argparse.ArgumentParser(
        description='Convert row-column-tag format CSV to cell-based labels.csv format for cricket object detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python csv_label_converter.py -i input.csv -o output.csv
  python csv_label_converter.py --input ../../atul_tagged_data.csv --output ../../dataset/test/labels/labels.csv

Object Mapping:
  Ball  -> 1
  Bat   -> 2
  Stump -> 3
  None  -> 0
        '''
    )

    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Path to input CSV file (row-column-tag format)'
    )

    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Path to output CSV file (cell-based labels format)'
    )

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Run conversion
    convert_labels(args.input, args.output)


if __name__ == "__main__":
    main()
