
import io
import random
import zipfile
from pathlib import PurePosixPath

import matplotlib.pyplot as plt
from PIL import Image

DATASET_PATH = "data/train.zip"


def main():
    with zipfile.ZipFile(DATASET_PATH, "r") as archive:
        names = [
            name for name in archive.namelist()
            if PurePosixPath(name).suffix.lower() in {".jpg", ".jpeg", ".png"}
            and PurePosixPath(name).parts[0] in {"real", "fake"}
        ]

        random.seed(42)
        samples = []

        for label in ("real", "fake"):
            candidates = [
                name for name in names
                if PurePosixPath(name).parts[0] == label
            ]
            samples.extend(random.sample(candidates, 4))

        fig, axes = plt.subplots(2, 4, figsize=(10, 5))

        for ax, name in zip(axes.flat, samples):
            with archive.open(name) as file:
                image = Image.open(io.BytesIO(file.read())).convert("RGB")

            ax.imshow(image)
            ax.set_title(PurePosixPath(name).parts[0])
            ax.axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
