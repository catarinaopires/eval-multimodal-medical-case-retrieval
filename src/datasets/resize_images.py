import os
import pathlib

import pandas as pd
from dotenv import load_dotenv
from PIL import Image
from tqdm.contrib.concurrent import process_map

load_dotenv(dotenv_path=pathlib.Path(__file__).parent.parent / ".env")
DATA_DIR_PATH = os.getenv("DATA_DIR_PATH")
IMAGES_PATH = DATA_DIR_PATH + "figures/images/"


def resize_img(r):
    img_path = os.path.join(IMAGES_PATH, r)
    image = Image.open(img_path)

    image.thumbnail(IMG_SIZE)  # Resize images and keep their aspect ratios

    # Add padding to the images
    background = Image.new("RGB", IMG_SIZE, (255, 255, 255))
    background.paste(
        image,
        (
            int((IMG_SIZE[0] - image.size[0]) / 2),
            int((IMG_SIZE[1] - image.size[1]) / 2),
        ),
    )

    new_path = os.path.join(OUTDIR, r)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    background.save(new_path)


IMG_SIZE = (336, 336)
df = pd.read_csv(DATA_DIR_PATH + "figures/image-paths.csv", dtype=str)
OUTDIR = DATA_DIR_PATH + "figures/images-resized"

process_map(
    resize_img, (r for r in list(df["img_path"])), max_workers=64, total=len(df)
)
