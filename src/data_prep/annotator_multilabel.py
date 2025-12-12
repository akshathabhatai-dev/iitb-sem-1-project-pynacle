import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Dict, List
from grid_utils import divide_into_grid, draw_grid_on_image, get_cell_index


class MultiLabelGridAnnotator:
    """
    Interactive tool for multi-label annotating grid cells in cricket images.

    Each cell can have multiple labels simultaneously:
        - Ball (red overlay)
        - Bat (brown overlay)
        - Stump (green/yellow overlay)

    This allows annotating cells that contain multiple objects (e.g., ball AND bat).
    """

    def __init__(self, image_dir: str, output_csv: str, grid_size=(8, 8)):
        """
        Initialize the multi-label annotator.

        Args:
            image_dir: Directory containing images to annotate
            output_csv: Path to save annotations (196-column format)
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

        # Store annotations as: {image_name: {cell_idx: {ball: 0/1, bat: 0/1, stump: 0/1}}}
        self.annotations = {}

        # Load existing annotations if file exists
        self.load_annotations()

        # Class names and colors
        self.class_names = ['Ball', 'Bat', 'Stump']
        self.class_colors = {
            'ball': (0, 0, 255),      # Red
            'bat': (139, 69, 19),     # Brown
            'stump': (0, 255, 255)    # Yellow (easier to see than green)
        }

    def load_annotations(self):
        """Load existing multi-label annotations from 196-column CSV if available."""
        if self.output_csv.exists():
            df = pd.read_csv(self.output_csv)
            for _, row in df.iterrows():
                img_name = row['ImageFileName']
                cell_labels = {}

                # Load 3 binary flags per cell (64 cells total)
                for cell_idx in range(64):
                    cell_id = cell_idx + 1
                    cell_labels[cell_idx] = {
                        'ball': int(row[f'c{cell_id:02d}_ball']),
                        'bat': int(row[f'c{cell_id:02d}_bat']),
                        'stump': int(row[f'c{cell_id:02d}_stump'])
                    }

                self.annotations[img_name] = cell_labels

            print(f"Loaded {len(self.annotations)} existing multi-label annotations")

    def save_annotations(self):
        """Save annotations to 196-column CSV."""
        if not self.annotations:
            print("No annotations to save")
            return

        rows = []
        for img_name, cell_labels in self.annotations.items():
            row = {'ImageFileName': img_name}

            # Add 3 columns per cell (64 cells × 3 = 192 columns)
            for cell_idx in range(64):
                cell_id = cell_idx + 1

                # Get labels for this cell (default to all zeros if not annotated)
                if cell_idx in cell_labels:
                    labels = cell_labels[cell_idx]
                else:
                    labels = {'ball': 0, 'bat': 0, 'stump': 0}

                row[f'c{cell_id:02d}_ball'] = int(labels['ball'])
                row[f'c{cell_id:02d}_bat'] = int(labels['bat'])
                row[f'c{cell_id:02d}_stump'] = int(labels['stump'])

            rows.append(row)

        df = pd.DataFrame(rows)

        # Ensure column order: ImageFileName, c01_ball, c01_bat, c01_stump, c02_ball, ...
        columns = ['ImageFileName']
        for cell_id in range(1, 65):
            columns.extend([f'c{cell_id:02d}_ball', f'c{cell_id:02d}_bat', f'c{cell_id:02d}_stump'])

        df = df[columns]

        # Create output directory if needed
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(self.output_csv, index=False)
        print(f"Saved {len(rows)} multi-label annotations to {self.output_csv}")

    def annotate_image(self, image_path: Path):
        """
        Annotate a single image with multi-label grid annotations.

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
            # Initialize all cells with no labels
            cell_labels = {i: {'ball': 0, 'bat': 0, 'stump': 0} for i in range(64)}

        # Create window
        window_name = 'Multi-Label Cricket Object Annotation'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1200, 900)

        # Mouse callback state
        state = {
            'selected_cell': -1,
            'show_dialog': False
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
                    state['show_dialog'] = True

                    print(f"\nSelected cell ({row},{col}) [c{cell_idx+1:02d}]")
                    print("Press 'b' to toggle Ball, 't' to toggle Bat, 's' to toggle Stump")
                    print("Press 'c' to clear all labels, 'd' to close dialog")

        cv2.setMouseCallback(window_name, mouse_callback)

        print("\n" + "="*70)
        print(f"Annotating: {img_name}")
        print("="*70)
        print("Controls:")
        print("  Click on a cell to select it for multi-label annotation")
        print("  When cell selected:")
        print("    b = Toggle Ball (Red)")
        print("    t = Toggle Bat (Brown)")
        print("    s = Toggle Stump (Yellow)")
        print("    c = Clear all labels for selected cell")
        print("    d = Deselect cell / close dialog")
        print("  General:")
        print("    n = Save and next image")
        print("    p = Previous image")
        print("    q = Quit without saving")
        print("    r = Reject image (delete file)")
        print("="*70 + "\n")

        while True:
            # Create display image with grid and labels
            display = image.copy()
            display = draw_grid_on_image(display, self.grid_size, color=(0, 255, 0), thickness=2)

            height, width = display.shape[:2]
            cell_height = height // self.rows
            cell_width = width // self.cols

            # Draw cell labels with color overlays
            for cell_idx in range(64):
                row_idx = cell_idx // self.cols
                col_idx = cell_idx % self.cols

                # Get cell boundaries
                y1 = row_idx * cell_height
                y2 = (row_idx + 1) * cell_height
                x1 = col_idx * cell_width
                x2 = (col_idx + 1) * cell_width

                # Draw cell ID at top-left corner
                cell_id = f"c{cell_idx+1:02d}"
                cv2.putText(display, cell_id, (x1 + 5, y1 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

                # Get labels for this cell
                labels = cell_labels.get(cell_idx, {'ball': 0, 'bat': 0, 'stump': 0})

                # Draw colored overlays for each active label
                overlay = display.copy()
                has_labels = False

                if labels['ball']:
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), self.class_colors['ball'], -1)
                    has_labels = True

                if labels['bat']:
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), self.class_colors['bat'], -1)
                    has_labels = True

                if labels['stump']:
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), self.class_colors['stump'], -1)
                    has_labels = True

                # Blend overlay with display (semi-transparent)
                if has_labels:
                    cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)

                # Draw label indicators as text at cell center
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                label_texts = []
                if labels['ball']:
                    label_texts.append('B')
                if labels['bat']:
                    label_texts.append('T')
                if labels['stump']:
                    label_texts.append('S')

                if label_texts:
                    label_str = ','.join(label_texts)
                    cv2.putText(display, label_str, (cx - 15, cy + 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Highlight selected cell
                if cell_idx == state['selected_cell'] and state['show_dialog']:
                    cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 0), 4)

            # Draw instructions and status
            cv2.putText(display, f"Image: {img_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display, f"Progress: {self.current_idx + 1}/{len(self.image_files)}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Show dialog info if cell selected
            if state['show_dialog'] and state['selected_cell'] >= 0:
                cell_idx = state['selected_cell']
                labels = cell_labels.get(cell_idx, {'ball': 0, 'bat': 0, 'stump': 0})

                dialog_text = f"Cell c{cell_idx+1:02d}: "
                status = []
                if labels['ball']:
                    status.append("Ball")
                if labels['bat']:
                    status.append("Bat")
                if labels['stump']:
                    status.append("Stump")

                if status:
                    dialog_text += ", ".join(status)
                else:
                    dialog_text += "No labels"

                cv2.putText(display, dialog_text, (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                cv2.putText(display, "Press: b=Ball, t=Bat, s=Stump, c=Clear, d=Done", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            cv2.imshow(window_name, display)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            # Cell-specific controls (only when dialog active)
            if state['show_dialog'] and state['selected_cell'] >= 0:
                cell_idx = state['selected_cell']

                if key == ord('b'):
                    # Toggle ball label
                    cell_labels[cell_idx]['ball'] = 1 - cell_labels[cell_idx]['ball']
                    status = "ON" if cell_labels[cell_idx]['ball'] else "OFF"
                    print(f"Cell c{cell_idx+1:02d}: Ball = {status}")

                elif key == ord('t'):
                    # Toggle bat label
                    cell_labels[cell_idx]['bat'] = 1 - cell_labels[cell_idx]['bat']
                    status = "ON" if cell_labels[cell_idx]['bat'] else "OFF"
                    print(f"Cell c{cell_idx+1:02d}: Bat = {status}")

                elif key == ord('s'):
                    # Toggle stump label
                    cell_labels[cell_idx]['stump'] = 1 - cell_labels[cell_idx]['stump']
                    status = "ON" if cell_labels[cell_idx]['stump'] else "OFF"
                    print(f"Cell c{cell_idx+1:02d}: Stump = {status}")

                elif key == ord('c'):
                    # Clear all labels for this cell
                    cell_labels[cell_idx] = {'ball': 0, 'bat': 0, 'stump': 0}
                    print(f"Cell c{cell_idx+1:02d}: Cleared all labels")

                elif key == ord('d'):
                    # Close dialog
                    state['show_dialog'] = False
                    state['selected_cell'] = -1
                    print("Dialog closed")

            # Global controls
            if key == ord('n'):
                # Save and next
                self.annotations[img_name] = cell_labels
                self.save_annotations()
                print(f"Saved annotations for {img_name}")
                break

            elif key == ord('p'):
                # Previous image
                self.annotations[img_name] = cell_labels
                self.save_annotations()
                self.current_idx = max(0, self.current_idx - 2)
                break

            elif key == ord('q'):
                # Quit without saving
                cv2.destroyAllWindows()
                return False

            elif key == ord('r'):
                # Reject image: delete from folder & CSV
                print(f"Rejecting image: {img_name}")

                # 1. Delete the image file
                try:
                    os.remove(image_path)
                    print(f"Deleted image file: {image_path}")
                except Exception as e:
                    print(f"Error deleting image file: {e}")

                # 2. Remove annotation entry
                if img_name in self.annotations:
                    del self.annotations[img_name]
                    self.save_annotations()
                    print(f"Removed annotation record for {img_name}")

                # 3. Remove from image list
                self.image_files.pop(self.current_idx)
                break

        cv2.destroyAllWindows()
        return True

    def run(self):
        """Run the multi-label annotation tool for all images."""
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

        print("\nMulti-label annotation complete!")
        print(f"Total annotated: {len(self.annotations)}/{len(self.image_files)}")


def main():
    """Main function to run the multi-label annotator."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Multi-label annotation tool for cricket images. '
                    'Allows labeling multiple objects per cell (Ball, Bat, Stump).')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Directory containing images to annotate')
    parser.add_argument('--output_csv', type=str, required=True,
                       help='Path to save 196-column multi-label annotations CSV')
    parser.add_argument('--grid_size', type=int, nargs=2, default=[8, 8],
                       help='Grid size (rows cols), default: 8 8')

    args = parser.parse_args()

    annotator = MultiLabelGridAnnotator(
        image_dir=args.image_dir,
        output_csv=args.output_csv,
        grid_size=tuple(args.grid_size)
    )

    annotator.run()


if __name__ == '__main__':
    main()
