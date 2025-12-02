"""
Quick script to check your project progress and readiness.
"""

import os
from pathlib import Path


def count_images(directory):
    """Count images in a directory."""
    if not os.path.exists(directory):
        return 0

    dir_path = Path(directory)
    images = list(dir_path.glob('*.png')) + list(dir_path.glob('*.jpg')) + list(dir_path.glob('*.jpeg'))
    return len(images)


def check_file_exists(filepath):
    """Check if a file exists."""
    return os.path.exists(filepath)


def main():
    print("="*70)
    print("CRICKET OBJECT DETECTION - PROGRESS CHECK")
    print("="*70)

    # Check training images
    train_img_count = count_images('dataset/train/images')
    train_target = 240
    train_status = "OK" if train_img_count >= train_target else f"Need {train_target - train_img_count} more"

    print(f"\n1. Training Images: {train_img_count}/{train_target} - {train_status}")

    # Check test images
    test_img_count = count_images('dataset/test/images')
    test_target = 60
    test_status = "OK" if test_img_count >= test_target else f"Need {test_target - test_img_count} more"

    print(f"2. Test Images: {test_img_count}/{test_target} - {test_status}")

    # Check labels
    train_labels_exist = check_file_exists('dataset/train/labels/labels.csv')
    test_labels_exist = check_file_exists('dataset/test/labels/labels.csv')

    print(f"\n3. Training Labels: {'DONE' if train_labels_exist else 'NOT ANNOTATED'}")
    print(f"4. Test Labels: {'DONE' if test_labels_exist else 'NOT ANNOTATED'}")

    # Check if dependencies installed
    try:
        import cv2
        import numpy
        import sklearn
        import skimage
        deps_installed = True
    except ImportError:
        deps_installed = False

    print(f"\n5. Dependencies Installed: {'YES' if deps_installed else 'NO - Run: pip install -r requirements.txt'}")

    # Check if model exists
    model_exists = check_file_exists('models/cricket_model.pkl')
    print(f"6. Model Trained: {'YES' if model_exists else 'NO'}")

    # Check if predictions exist
    predictions_exist = check_file_exists('outputs/predictions.csv')
    print(f"7. Predictions Generated: {'YES' if predictions_exist else 'NO'}")

    # Overall readiness
    print("\n" + "="*70)
    print("READINESS CHECKLIST")
    print("="*70)

    checklist = [
        (train_img_count >= train_target, f"Training images ({train_img_count}/{train_target})"),
        (test_img_count >= test_target, f"Test images ({test_img_count}/{test_target})"),
        (train_labels_exist, "Training labels"),
        (test_labels_exist, "Test labels"),
        (deps_installed, "Dependencies"),
    ]

    for done, item in checklist:
        status = "[X]" if done else "[ ]"
        print(f"{status} {item}")

    all_ready = all(item[0] for item in checklist)

    print("\n" + "="*70)
    if all_ready:
        print("STATUS: READY TO RUN PIPELINE!")
        print("\nNext step:")
        print("  python main_pipeline.py --mode full")
    else:
        print("STATUS: NOT READY - Complete checklist items above")
        print("\nNext steps:")
        if not deps_installed:
            print("  1. pip install -r requirements.txt")
        if train_img_count < train_target:
            print(f"  2. Add {train_target - train_img_count} more training images to dataset/train/images/")
        if test_img_count < test_target:
            print(f"  3. Add {test_target - test_img_count} test images to dataset/test/images/")
        if not train_labels_exist:
            print("  4. Annotate training images: cd src/data_prep && python annotator.py ...")
        if not test_labels_exist:
            print("  5. Annotate test images: cd src/data_prep && python annotator.py ...")

    print("="*70)

    # Additional info
    if model_exists and predictions_exist:
        print("\nPROJECT COMPLETE! Check outputs/predictions.csv for your submission.")


if __name__ == '__main__':
    main()
