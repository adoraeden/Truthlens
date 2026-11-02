
from pathlib import PurePosixPath
from collections import Counter
import zipfile

DATASET_PATH = "data/train.zip"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main():
    with zipfile.ZipFile(DATASET_PATH, "r") as archive:
        image_files = [
            name for name in archive.namelist()
            if PurePosixPath(name).suffix.lower() in IMAGE_EXTENSIONS
            and len(PurePosixPath(name).parts) >= 2
        ]

    counts = Counter(
        PurePosixPath(name).parts[0].lower()
        for name in image_files
    )

    print("EXPERIMENT READINESS CHECK")
    print("=" * 35)
    print(f"Total images: {len(image_files):,}")

    for label, count in sorted(counts.items()):
        print(f"{label}: {count:,}")

    print("\nPlanned evaluation approach:")
    print("- Keep real and fake classes balanced.")
    print("- Separate training, validation and test images.")
    print("- Fit model parameters using training images only.")
    print("- Use validation images for model selection.")
    print("- Reserve test images for final evaluation.")
    print("\nIMPORTANT: This ZIP has no original independent test split.")
    print("Do not report a random split as the official CIFAKE test result.")


if __name__ == "__main__":
    main()
