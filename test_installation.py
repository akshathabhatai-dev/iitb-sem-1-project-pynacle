"""
Test script to verify installation and basic functionality.
Run this after installing dependencies to ensure everything works.
"""

import sys
import os

# Fix for Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")

    try:
        import numpy as np
        print("  ✓ NumPy")
    except ImportError as e:
        print(f"  ✗ NumPy - {e}")
        return False

    try:
        import pandas as pd
        print("  ✓ Pandas")
    except ImportError as e:
        print(f"  ✗ Pandas - {e}")
        return False

    try:
        import cv2
        print("  ✓ OpenCV")
    except ImportError as e:
        print(f"  ✗ OpenCV - {e}")
        return False

    try:
        from sklearn.ensemble import RandomForestClassifier
        print("  ✓ Scikit-learn")
    except ImportError as e:
        print(f"  ✗ Scikit-learn - {e}")
        return False

    try:
        from skimage.feature import local_binary_pattern, hog
        print("  ✓ Scikit-image")
    except ImportError as e:
        print(f"  ✗ Scikit-image - {e}")
        return False

    try:
        import matplotlib.pyplot as plt
        print("  ✓ Matplotlib")
    except ImportError as e:
        print(f"  ✗ Matplotlib - {e}")
        return False

    try:
        import seaborn as sns
        print("  ✓ Seaborn")
    except ImportError as e:
        print(f"  ✗ Seaborn - {e}")
        return False

    return True


def test_feature_extraction():
    """Test feature extraction on a dummy image."""
    print("\nTesting feature extraction...")

    try:
        import numpy as np
        import sys
        import os

        # Add src paths
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'features'))
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'data_prep'))

        from feature_extractor import extract_features_from_cell

        # Create a dummy 75x100 RGB cell
        dummy_cell = np.random.randint(0, 256, (75, 100, 3), dtype=np.uint8)

        # Extract features
        features = extract_features_from_cell(dummy_cell)

        # Verify feature count
        assert len(features) == 88, f"Expected 88 features, got {len(features)}"

        print(f"  ✓ Feature extraction successful (88 features)")
        return True

    except Exception as e:
        print(f"  ✗ Feature extraction failed - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_grid_utils():
    """Test grid utilities."""
    print("\nTesting grid utilities...")

    try:
        import numpy as np
        import sys
        import os

        sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'data_prep'))

        from grid_utils import divide_into_grid

        # Create a dummy 600x800 image
        dummy_image = np.random.randint(0, 256, (600, 800, 3), dtype=np.uint8)

        # Divide into grid
        cells = divide_into_grid(dummy_image, grid_size=(8, 8))

        # Verify cell count and size
        assert len(cells) == 64, f"Expected 64 cells, got {len(cells)}"
        assert cells[0].shape == (75, 100, 3), f"Expected cell shape (75, 100, 3), got {cells[0].shape}"

        print(f"  ✓ Grid division successful (64 cells of 75×100)")
        return True

    except Exception as e:
        print(f"  ✗ Grid utilities failed - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_directory_structure():
    """Test that required directories exist."""
    print("\nTesting directory structure...")

    required_dirs = [
        'src/features',
        'src/models',
        'src/data_prep',
        'dataset/train/images',
        'models',
        'outputs'
    ]

    all_exist = True
    for dir_path in required_dirs:
        full_path = os.path.join(os.path.dirname(__file__), dir_path)
        if os.path.exists(full_path):
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} - Not found")
            all_exist = False

    return all_exist


def test_individual_features():
    """Test each feature extraction module individually."""
    print("\nTesting individual feature extractors...")

    import numpy as np
    import sys
    import os

    sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'features'))

    dummy_rgb = np.random.randint(0, 256, (75, 100, 3), dtype=np.uint8)
    dummy_gray = np.random.randint(0, 256, (75, 100), dtype=np.uint8)

    tests = []

    # Test RGB features
    try:
        from rgb_features import extract_rgb_histogram_features
        features = extract_rgb_histogram_features(dummy_rgb)
        assert len(features) == 14
        print("  ✓ RGB features (14)")
        tests.append(True)
    except Exception as e:
        print(f"  ✗ RGB features - {e}")
        tests.append(False)

    # Test HSV features
    try:
        from hsv_features import extract_hsv_color_features, rgb_to_hsv
        hsv = rgb_to_hsv(dummy_rgb)
        features = extract_hsv_color_features(hsv)
        assert len(features) == 14
        print("  ✓ HSV features (14)")
        tests.append(True)
    except Exception as e:
        print(f"  ✗ HSV features - {e}")
        tests.append(False)

    # Test LBP features
    try:
        from lbp_features import extract_lbp_features
        features = extract_lbp_features(dummy_gray)
        assert len(features) == 10
        print("  ✓ LBP features (10)")
        tests.append(True)
    except Exception as e:
        print(f"  ✗ LBP features - {e}")
        tests.append(False)

    # Test HOG features
    try:
        from hog_features import extract_hog_features
        features = extract_hog_features(dummy_gray)
        assert len(features) == 20
        print("  ✓ HOG features (20)")
        tests.append(True)
    except Exception as e:
        print(f"  ✗ HOG features - {e}")
        tests.append(False)

    # Test Gabor features
    try:
        from gabor_features import extract_gabor_features
        features = extract_gabor_features(dummy_gray)
        assert len(features) == 20
        print("  ✓ Gabor features (20)")
        tests.append(True)
    except Exception as e:
        print(f"  ✗ Gabor features - {e}")
        tests.append(False)

    # Test GLCM features
    try:
        from glcm_features import extract_glcm_features
        features = extract_glcm_features(dummy_gray)
        assert len(features) == 10
        print("  ✓ GLCM features (10)")
        tests.append(True)
    except Exception as e:
        print(f"  ✗ GLCM features - {e}")
        tests.append(False)

    return all(tests)


def main():
    """Run all tests."""
    print("="*60)
    print("CRICKET OBJECT DETECTION - INSTALLATION TEST")
    print("="*60)

    results = []

    results.append(("Imports", test_imports()))
    results.append(("Directory Structure", test_directory_structure()))
    results.append(("Grid Utilities", test_grid_utils()))
    results.append(("Individual Features", test_individual_features()))
    results.append(("Full Feature Extraction", test_feature_extraction()))

    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:30s} {status}")

    all_passed = all(result[1] for result in results)

    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED - Installation successful!")
        print("\nYou can now:")
        print("  1. Annotate images: cd src/data_prep && python annotator.py ...")
        print("  2. Run pipeline: python main_pipeline.py --mode full")
    else:
        print("✗ SOME TESTS FAILED - Please check errors above")
        print("\nCommon fixes:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Check Python version: Python 3.7+ required")
    print("="*60)

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
