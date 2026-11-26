
import zipfile
from collections import Counter
from pathlib import PurePosixPath

DATASET_PATH = "data/train.zip"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main():
    with zipfile.ZipFile(DATASET_PATH, "r") as archive:
        image_files = [
            name
            for name in archive.namelist()
            if PurePosixPath(name).suffix.lower() in IMAGE_EXTENSIONS
            and len(PurePosixPath(name).parts) >= 2
        ]

        class_counts = Counter(
            PurePosixPath(name).parts[0] for name in image_files
        )

        print("DATASET SUMMARY")
        print("-" * 30)
        print(f"Total images: {len(image_files):,}")

        for label in sorted(class_counts):
            print(f"{label}: {class_counts[label]:,}")

        print("\nSAMPLE IMAGES")
        print("-" * 30)

        for name in image_files[:5]:
            print(name)

        print("\nIMAGE DIMENSIONS")
        print("-" * 30)

        from PIL import Image
        import io

        for name in image_files[:5]:
            with archive.open(name) as image_file:
                image = Image.open(io.BytesIO(image_file.read()))
                print(f"{name}: {image.size}, {image.mode}")


if __name__ == "__main__":
    main()
