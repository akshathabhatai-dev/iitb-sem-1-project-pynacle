import pandas as pd
import argparse
import shutil
from pathlib import Path


def find_and_copy_unlabeled_images(image_dir: str, label_csv: str, output_dir: str):
    """
    Find images in the image directory that are not in the label CSV file
    and copy them to the output directory.

    Args:
        image_dir: Directory containing all images
        label_csv: Path to the label CSV file
        output_dir: Directory to copy unlabeled images to
    """
    image_dir = Path(image_dir)
    label_csv = Path(label_csv)
    output_dir = Path(output_dir)

    # --- Read label CSV file ---
    try:
        df_label = pd.read_csv(label_csv)
        print(f"Loaded label file: {label_csv} ({len(df_label)} rows)")
    except FileNotFoundError:
        print(f"Error: Label file not found: {label_csv}")
        return
    except Exception as e:
        print(f"Error reading label file: {e}")
        return

    # --- Check if ImageFileName column exists ---
    if "ImageFileName" not in df_label.columns:
        print(f"Error: 'ImageFileName' column not found in {label_csv}")
        return

    # --- Get labeled images from CSV ---
    labeled_images = set(df_label["ImageFileName"])
    print(f"Found {len(labeled_images)} labeled images in CSV")

    # --- Get all images from directory ---
    if not image_dir.exists():
        print(f"Error: Image directory not found: {image_dir}")
        return

    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
    all_images = [f for f in image_dir.iterdir()
                  if f.is_file() and f.suffix.lower() in image_extensions]

    print(f"Found {len(all_images)} images in directory: {image_dir}")

    # --- Get image filenames from directory ---
    image_filenames = set(img.name for img in all_images)

    # --- Find unlabeled images (in folder but not in CSV) ---
    unlabeled_images = []
    for img_path in all_images:
        if img_path.name not in labeled_images:
            unlabeled_images.append(img_path)

    # --- Find missing images (in CSV but not in folder) ---
    missing_images = labeled_images - image_filenames

    print(f"\nFound {len(unlabeled_images)} unlabeled images (in folder but not in CSV)")

    if missing_images:
        print(f"\nWARNING: Found {len(missing_images)} images in CSV but NOT in folder:")
        for img_name in sorted(missing_images):
            print(f"  - {img_name}")

    if not unlabeled_images:
        print("\nAll images in the folder have been labeled!")
        if not missing_images:
            return

    # --- Create output directory ---
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # --- Copy unlabeled images to output directory ---
    print("\nCopying unlabeled images:")
    copied_count = 0
    for img_path in unlabeled_images:
        try:
            dest_path = output_dir / img_path.name
            shutil.copy2(img_path, dest_path)
            print(f"  Copied: {img_path.name}")
            copied_count += 1
        except Exception as e:
            print(f"  Error copying {img_path.name}: {e}")

    # --- Print summary ---
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Total images in directory: {len(all_images)}")
    print(f"  Total labeled images in CSV: {len(labeled_images)}")
    print(f"  Images in both folder and CSV: {len(image_filenames & labeled_images)}")
    print(f"  Unlabeled images (in folder, not in CSV): {len(unlabeled_images)}")
    print(f"  Missing images (in CSV, not in folder): {len(missing_images)}")
    print(f"  Images copied to output: {copied_count}")
    print("="*60)


def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Find unlabeled images and copy them to an output directory'
    )
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Directory containing all images')
    parser.add_argument('--label_csv', type=str, required=True,
                       help='Path to the label CSV file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to copy unlabeled images to')

    args = parser.parse_args()

    find_and_copy_unlabeled_images(args.image_dir, args.label_csv, args.output_dir)


if __name__ == '__main__':
    main()
