import os
import argparse
from PIL import Image, ImageFile
import pillow_avif  # Required to decode AVIF/HEIC formats

ImageFile.LOAD_TRUNCATED_IMAGES = True


def detect_type(path):
    """Detect actual image type using magic numbers."""
    with open(path, "rb") as f:
        header = f.read(16)

    if header.startswith(b"\xff\xd8"):
        return "jpeg"

    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"

    if header.startswith(b"GIF87a") or header.startswith(b"GIF89a"):
        return "gif"

    if header[0:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "webp"

    if header.startswith(b"BM"):
        return "bmp"

    # AVIF / HEIC / MIAF
    if header[4:12] in (
        b"ftypavif",
        b"ftypheic",
        b"ftypmif1",
        b"ftypmsf1",
        b"ftypiso5",
    ):
        return "avif"

    return None


def center_crop_to_aspect(img, aspect=4 / 3):
    """Crop image to target aspect ratio from the center."""
    w, h = img.size
    curr = w / h

    if curr > aspect:  # too wide → crop width
        new_w = int(h * aspect)
        left = (w - new_w) // 2
        return img.crop((left, 0, left + new_w, h))

    else:  # too tall → crop height
        new_h = int(w / aspect)
        top = (h - new_h) // 2
        return img.crop((0, top, w, top + new_h))


def process_folder(input_folder, scaled_folder="scaled"):
    """Process and scale images inside the given folder."""
    os.makedirs(scaled_folder, exist_ok=True)

    for fname in os.listdir(input_folder):
        in_path = os.path.join(input_folder, fname)
        ftype = detect_type(in_path)

        if ftype is None:
            print(f"❌ Not an image: {fname}")
            continue

        try:
            img = Image.open(in_path)
            img.load()

            # Reject images smaller than 800x600
            if img.width < 800 or img.height < 600:
                print(f"❌ Skipped (too small): {fname} — {img.width}x{img.height}")
                continue

            # Convert mode safely
            if img.mode not in ("RGB", "RGBA", "L"):
                img = img.convert("RGB")

            # Center crop 4:3
            cropped = center_crop_to_aspect(img)

            # Resize to 800x600
            resized = cropped.resize((800, 600), Image.LANCZOS)

            # Determine output extension
            if ftype == "avif":
                out_ext = ".png"  # AVIF → PNG
            else:
                out_ext = os.path.splitext(fname)[1]  # original extension

            out_name = os.path.splitext(fname)[0] + out_ext
            out_path = os.path.join(scaled_folder, out_name)

            # Save according to extension
            if out_ext.lower() == ".png":
                resized.save(out_path, "PNG")
            elif out_ext.lower() == ".webp":
                resized.save(out_path, "WEBP", quality=95)
            elif out_ext.lower() in (".jpg", ".jpeg"):
                if resized.mode == "RGBA":
                    resized = resized.convert("RGB")
                resized.save(out_path, "JPEG", quality=95)
            else:
                resized.save(out_path)

            print(f"✔ Saved: {out_path}")

        except Exception as e:
            print(f"⚠️ Could not process {fname}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Crop to 4:3 and scale images to 800x600.")
    parser.add_argument(
        "folder",
        type=str,
        help="Input folder containing images"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="scaled",
        help="Output folder (default: scaled)"
    )

    args = parser.parse_args()
    process_folder(args.folder, args.output)


if __name__ == "__main__":
    main()
