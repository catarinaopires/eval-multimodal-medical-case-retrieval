import argparse
import ast
import os
import pathlib
import time
from datetime import datetime, timedelta

import pandas as pd
import torch
import torchvision.transforms as transforms
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

from datasets.ArticleImagesDataset import ArticleImagesDataset
from encoder.LlavaGenerator import LlavaGenerator

load_dotenv(dotenv_path=pathlib.Path(__file__).parent / ".env")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
DATA_DIR_PATH = os.getenv("DATA_DIR_PATH")
OUTPUT_DIR_PATH = os.getenv("OUTPUT_DIR_PATH")

START_TIME = time.time()


########################## ARTICLES  ###########################
########################## DATALOADER ##########################
IMAGES_PATH = (
    DATA_DIR_PATH + "figures/images-resized/"
)  # TODO: Output directory after running datasets/resize_images.py

START_TIME = time.time()


def collate_fn(batch):
    return {
        "image": torch.stack([sample["image"] for sample in batch]),
        "img_path": [sample["img_path"] for sample in batch],
    }


########################## CHUNKS ##########################


def change_to_generated_captions(article_figures, generator):
    for figure in article_figures:
        path = f"{DATA_DIR_PATH}figures/images/{figure['fig_id']}.jpg"

        if os.path.isfile(path):
            image = Image.open(path)
            figure["generated-caption"] = generator.generate_image_description(image)
        else:
            print(f"Image not found: {path}")
    return article_figures


def image_to_text_articles(generator, df: pd.DataFrame):
    """
    Generate description for article images.
    """

    for article_figures in df["figures"]:
        article_id = df.index[df["figures"] == article_figures].tolist()[0]  # row ID

        article_figures = ast.literal_eval(article_figures)
        df.at[article_id, "figures"] = change_to_generated_captions(
            article_figures, generator
        )

    return df


def image_to_text_articles_processing(
    generator, input_file, output_file, checkpoint_file, chunksize=1000
):
    # Determine the starting chunk index
    start_chunk = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            start_chunk = int(f.read().strip())

    # Process each chunk from the starting point
    for chunk_index, chunk in enumerate(pd.read_csv(input_file, chunksize=chunksize)):
        if chunk_index < start_chunk:
            continue

        # Process the chunk
        processed_chunk = image_to_text_articles(generator, chunk)
        # Save the processed chunk to the output file
        processed_chunk.to_csv(output_file, mode="a", index=False, header=False)

        # Update the checkpoint file
        with open(checkpoint_file, "w") as f:
            f.write(str(chunk_index + 1))

        print(f"Processed chunk {chunk_index + 1}")


########################## TOPICS ##########################


def image_to_text_topics(generator, df: pd.DataFrame):
    """
    Generate description for topic images.
    """

    df["image-descriptions"] = ""

    for topics_images in df["images"]:
        topic_id = df.index[df["images"] == topics_images].tolist()[0]  # row ID

        topics_images = ast.literal_eval(topics_images)
        images = get_images_from_paths(topics_images)

        img_descriptions = [generator.generate_image_description(img) for img in images]

        df.at[topic_id, "image-descriptions"] = img_descriptions

    return df


def get_images_from_paths(images_paths):
    images = []

    for image_path in images_paths:
        path = DATA_DIR_PATH + image_path
        if os.path.isfile(path):
            image = Image.open(path)
            images.append(image)
            print(image_path)
        else:
            print(f"Image not found: {image_path}")
    return images


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "type",
        help="generate descriptions for topics or articles",
        choices=("topics", "articles"),
    )

    parser.add_argument(
        "-p",
        "--path",
        type=str,
        help="path to the CSV file to generate descriptions of images",
    )

    return parser


def main(args):
    print("\nStarted at", datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss\n"))

    if not os.path.isfile(args.path):
        parser.error(f"The path given ({args.path}) is not a file.")

    if not args.path.endswith(".csv"):
        parser.error(f"The file given by the path ({args.path}) is not CSV.")

    # Use the GPU or MPS if available, otherwise stick with the CPU - this is the default for PyTorch
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    print(f"Using device: {DEVICE}\n")

    generator = LlavaGenerator(DEVICE, "llava-hf/llava-1.5-7b-hf")

    START_TIME = time.time()
    current_datetime = datetime.now().strftime("_%Y-%m-%d_%Hh%Mm%Ss")
    output_path = f"{OUTPUT_DIR_PATH}Llava_i2t_{args.type}_{current_datetime}.csv"

    if args.type == "articles":
        # df = image_to_text_articles(generator, df) # Normal
        # image_to_text_articles_processing(generator, args.path, output_path, f"{OUTPUT_DIR_PATH}i2t_checkpoint_{current_datetime}.txt") # Chunks strategy

        # With dataloader
        article_images_dataset = ArticleImagesDataset(
            csv_file=args.path,
            data_dir=IMAGES_PATH,
            transform=transforms.ToTensor(),
        )

        print("Length of the dataset: ", len(article_images_dataset))

        dataloader = torch.utils.data.DataLoader(
            article_images_dataset,
            batch_size=1,
            num_workers=3,
            shuffle=False,
            collate_fn=collate_fn,
        )

        f = open(output_path, "a")
        f.write("image_path, description\n")
        for i_batch, sample_batched in enumerate(tqdm(dataloader)):
            generated_descriptions = generator.generate_image_description(
                sample_batched["image"]
            )

            for filename, description in zip(
                sample_batched["img_path"], generated_descriptions
            ):
                f.write(f"{filename}, {description}\n")

            print(f"Batch {i_batch} done")
            if i_batch % 10 == 0:
                f.flush()
                print("file flushed", flush=True)

        f.close()
    elif args.type == "topics":
        df = pd.read_csv(args.path, index_col=0)

        df = image_to_text_topics(generator, df)

        print(df.head())

        current_datetime = datetime.now().strftime("_%Y-%m-%d_%Hh%Mm%Ss")
        output_path = (
            f"{OUTPUT_DIR_PATH}{args.generator}_i2t_{args.type}_{current_datetime}.csv"
        )

        df.to_csv(output_path, encoding="utf-8")

    print(f"Total time:", str(timedelta(seconds=(time.time() - START_TIME))))

    print(f"i2t image description dataframe written to '{output_path}'.")
    print("\nDone at", current_datetime)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
