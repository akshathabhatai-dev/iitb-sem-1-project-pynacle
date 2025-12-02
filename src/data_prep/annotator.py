import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Dict, List
from grid_utils import divide_into_grid, draw_grid_on_image, get_cell_index


class GridAnnotator:
    """
    Interactive tool for annotating grid cells in cricket images.

    Labels:
        0 = no object
        1 = ball
        2 = bat
        3 = stump
    """

    def __init__(self, image_dir: str, output_csv: str, grid_size=(8, 8)):
        """
        Initialize the annotator.

        Args:
            image_dir: Directory containing images to annotate
            output_csv: Path to save annotations
            grid_size: Grid dimensions (rows, cols)
        """
        self.image_dir = Path(image_dir)
        self.output_csv = Path(output_csv)
        self.grid_size = grid_size
        self.rows, self.cols = grid_size

        # Get all image files
        self.image_files = sorted(list(self.image_dir.glob('*.png')) +
                                 list(self.image_dir.glob('*.jpg')) +
                                 list(self.image_dir.glob('*.jpeg')))

        self.current_idx = 0
        self.annotations = {}  # {image_name: [64 labels]}

        # Load existing annotations if file exists
        self.load_annotations()

        # Class names and colors
        self.class_names = {
            0: 'No Object',
            1: 'Ball',
            2: 'Bat',
            3: 'Stump'
        }

        self.class_colors = {
            0: (128, 128, 128),  # Gray
            1: (0, 0, 255),      # Red
            2: (139, 69, 19),    # Brown
            3: (255, 255, 255)   # White
        }

    def load_annotations(self):
        """Load existing annotations from CSV if available."""
        if self.output_csv.exists():
            df = pd.read_csv(self.output_csv)
            for _, row in df.iterrows():
                img_name = row['ImageFileName']
                labels = [row[f'c{i:02d}'] for i in range(1, 65)]
                self.annotations[img_name] = labels
            print(f"Loaded {len(self.annotations)} existing annotations")

    def save_annotations(self):
        """Save annotations to CSV."""
        if not self.annotations:
            print("No annotations to save")
            return

        rows = []
        for img_name, labels in self.annotations.items():
            row = {'ImageFileName': img_name}
            for i, label in enumerate(labels):
                row[f'c{i+1:02d}'] = int(label)
            rows.append(row)

        df = pd.DataFrame(rows)

        # Ensure column order
        columns = ['ImageFileName'] + [f'c{i:02d}' for i in range(1, 65)]
        df = df[columns]

        # Create output directory if needed
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(self.output_csv, index=False)
        print(f"Saved {len(rows)} annotations to {self.output_csv}")

    def annotate_image(self, image_path: Path):
        """
        Annotate a single image with grid labels.

        Args:
            image_path: Path to the image
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error loading image: {image_path}")
            return

        # Check if already annotated
        img_name = image_path.name
        if img_name in self.annotations:
            cell_labels = self.annotations[img_name].copy()
            print(f"Loading existing annotations for {img_name}")
        else:
            cell_labels = [0] * 64  # Initialize all cells as "no object"

        # Create window
        window_name = 'Cricket Object Annotation'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1200, 900)

        # Mouse callback state
        state = {
            'current_label': 0,
            'selected_cell': -1
        }

        def mouse_callback(event, x, y, flags, param):
            """Handle mouse clicks to select cells."""
            if event == cv2.EVENT_LBUTTONDOWN:
                # Calculate cell position
                height, width = image.shape[:2]
                cell_height = height // self.rows
                cell_width = width // self.cols

                row = y // cell_height
                col = x // cell_width

                if 0 <= row < self.rows and 0 <= col < self.cols:
                    cell_idx = get_cell_index(row, col, self.grid_size)
                    state['selected_cell'] = cell_idx

                    # Update cell label
                    cell_labels[cell_idx] = state['current_label']
                    print(f"Cell ({row},{col}) [idx={cell_idx}] = {self.class_names[state['current_label']]}")

        cv2.setMouseCallback(window_name, mouse_callback)

        print("\n" + "="*60)
        print(f"Annotating: {img_name}")
        print("="*60)
        print("Controls:")
        print("  Click on a cell to label it with current class")
        print("  0 = No Object (Gray)")
        print("  1 = Ball (Red)")
        print("  2 = Bat (Brown)")
        print("  3 = Stump (White)")
        print("  s = Save and next")
        print("  q = Quit without saving")
        print("  n = Next image (save current)")
        print("  p = Previous image")
        print("="*60 + "\n")

        while True:
            # Create display image with grid and labels
            display = image.copy()
            display = draw_grid_on_image(display, self.grid_size, color=(0, 255, 0), thickness=2)

            # Draw cell labels
            height, width = display.shape[:2]
            cell_height = height // self.rows
            cell_width = width // self.cols

            for idx, label in enumerate(cell_labels):
                row = idx // self.cols
                col = idx % self.cols

                # Get cell center
                cx = col * cell_width + cell_width // 2
                cy = row * cell_height + cell_height // 2

                # Draw cell ID (c01, c02, etc.) at top-left corner of cell
                cell_id = f"c{idx+1:02d}"
                text_x = col * cell_width + 5
                text_y = row * cell_height + 20
                cv2.putText(display, cell_id, (text_x, text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                # Draw label indicator at center
                if label > 0:
                    color = self.class_colors[label]
                    cv2.circle(display, (cx, cy), 15, color, -1)
                    cv2.putText(display, str(label), (cx-5, cy+5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Draw instructions
            info_text = f"Current Class: {state['current_label']} ({self.class_names[state['current_label']]})"
            cv2.putText(display, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            progress_text = f"Image {self.current_idx + 1}/{len(self.image_files)}"
            cv2.putText(display, progress_text, (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow(window_name, display)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('0'):
                state['current_label'] = 0
                print(f"Selected class: {self.class_names[0]}")
            elif key == ord('1'):
                state['current_label'] = 1
                print(f"Selected class: {self.class_names[1]}")
            elif key == ord('2'):
                state['current_label'] = 2
                print(f"Selected class: {self.class_names[2]}")
            elif key == ord('3'):
                state['current_label'] = 3
                print(f"Selected class: {self.class_names[3]}")
            elif key == ord('s') or key == ord('n'):
                # Save and next
                self.annotations[img_name] = cell_labels
                self.save_annotations()
                print(f"Saved annotations for {img_name}")
                break
            elif key == ord('q'):
                # Quit without saving
                cv2.destroyAllWindows()
                return False
            elif key == ord('p'):
                # Previous image
                self.annotations[img_name] = cell_labels
                self.save_annotations()
                self.current_idx = max(0, self.current_idx - 2)
                break

        cv2.destroyAllWindows()
        return True

    def run(self):
        """Run the annotation tool for all images."""
        if not self.image_files:
            print(f"No images found in {self.image_dir}")
            return

        print(f"Found {len(self.image_files)} images")

        while self.current_idx < len(self.image_files):
            img_path = self.image_files[self.current_idx]

            if not self.annotate_image(img_path):
                print("Annotation cancelled")
                break

            self.current_idx += 1

        print("\nAnnotation complete!")
        print(f"Total annotated: {len(self.annotations)}/{len(self.image_files)}")


def main():
    """Main function to run the annotator."""
    import argparse

    parser = argparse.ArgumentParser(description='Annotate cricket images with grid labels')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Directory containing images to annotate')
    parser.add_argument('--output_csv', type=str, required=True,
                       help='Path to save annotations CSV')
    parser.add_argument('--grid_size', type=int, nargs=2, default=[8, 8],
                       help='Grid size (rows cols), default: 8 8')

    args = parser.parse_args()

    annotator = GridAnnotator(
        image_dir=args.image_dir,
        output_csv=args.output_csv,
        grid_size=tuple(args.grid_size)
    )

    annotator.run()


if __name__ == '__main__':
    main()
