import argparse
import ast
import os
import pathlib
import time
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from PIL import Image

from encoder.CLIPencoder import CLIPencoder
from encoder.LlamaEncoder import LlamaEncoder
from encoder.LongCLIPencoder import LongCLIPencoder

load_dotenv(dotenv_path=pathlib.Path(__file__).parent / ".env")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["HF_HOME"] = os.getenv("HF_HOME")
DATA_DIR_PATH = os.getenv("DATA_DIR_PATH")
OUTPUT_DIR_PATH = os.getenv("OUTPUT_DIR_PATH")

START_TIME = time.time()


def encode_text_articles(encoder, df: pd.DataFrame):
    """
    Encode the articles in the CSV file using the given encoder.
    """
    # id, title, authors, abstract, fulltext, figures ({'id': '', 'caption': ''})

    df_embeddings = df.copy()[["title", "abstract", "fulltext", "figures"]]
    # 1. Encode text titles
    df_embeddings["title"] = df_embeddings["title"].apply(
        lambda text: encoder.encode_text(text)
    )
    elapsed_time = time.time() - START_TIME
    temp_time = time.time()
    print("\nJust finished encoding titles:", str(timedelta(seconds=elapsed_time)))

    # 2. Encode text abstracts
    df_embeddings["abstract"] = df_embeddings["abstract"].apply(
        lambda text: encoder.encode_text(str(text))
    )
    elapsed_time = time.time() - temp_time
    temp_time = time.time()
    print("\nJust finished encoding abstracts:", str(timedelta(seconds=elapsed_time)))

    # 3. Encode text fulltexts
    df_embeddings["fulltext"] = df_embeddings["fulltext"].apply(
        lambda text: encoder.encode_text(str(text))
    )
    elapsed_time = time.time() - temp_time
    temp_time = time.time()
    print("\nJust finished encoding fulltexts:", str(timedelta(seconds=elapsed_time)))

    # 4. Encode captions
    # For each row in the dataframe, figures is a list of dictionaries with fig_id and caption
    for article_figures in df_embeddings["figures"]:
        article_id = df.index[df["figures"] == article_figures].tolist()[0]  # row ID

        article_figures = ast.literal_eval(article_figures)
        df_embeddings.at[article_id, "figures"] = [
            encoder.encode_text(fig["caption"]) for fig in article_figures
        ]

    elapsed_time = time.time() - temp_time
    print("\nJust finished encoding captions:", str(timedelta(seconds=elapsed_time)))

    return df_embeddings


def encode_text_articles_chunks(
    encoder, input_file, output_file, checkpoint_file, chunksize=1000
):
    output_dir = OUTPUT_DIR_PATH + "chunks"
    current_datetime = datetime.now().strftime("_%Y-%m-%d_%Hh%Mm%Ss")

    # Create chunks' directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Created directory", output_dir)

    # Determine the starting chunk index
    start_chunk = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            start_chunk = int(f.read().strip())

    print(f"Started at chunk nr {start_chunk}")

    # Process each chunk from the starting point
    for chunk_index, chunk in enumerate(
        pd.read_csv(input_file, chunksize=chunksize, index_col=0)
    ):
        if chunk_index < start_chunk:
            continue

        # Process the chunk
        processed_chunk = encode_text_articles(encoder, chunk)
        # Save the processed chunk to the output file
        processed_chunk.to_pickle(
            f"{output_dir}/chunk_{current_datetime}_{chunk_index}.pkl"
        )

        # Update the checkpoint file
        with open(checkpoint_file, "w") as f:
            f.write(str(chunk_index + 1))

        print(f"Processed chunk (encoding) {chunk_index + 1}")

    combine_chunk_files(output_dir, output_file)


def combine_chunk_files(output_dir, combined_output_file):
    print("Start combining chunk files...")
    files = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith(".pkl")
    ]
    dataframes = [pd.read_pickle(file) for file in sorted(files)]
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df.to_pickle(combined_output_file)
    print(f"Combined all chunks into {combined_output_file}")


def encode_articles(encoder, df: pd.DataFrame):
    """
    Encode the articles in the CSV file using the given encoder.
    """
    # id, title, authors, abstract, fulltext, figures ({'id': '', 'caption': ''})

    df_embeddings = df.copy()[["title", "abstract", "fulltext", "figures"]]
    # 1. Encode text titles
    df_embeddings["title"] = df_embeddings["title"].apply(
        lambda text: encoder.encode_text(text)
    )
    elapsed_time = time.time() - START_TIME
    temp_time = time.time()
    print("\nJust finished encoding titles:", str(timedelta(seconds=elapsed_time)))

    # 2. Encode text abstracts
    df_embeddings["abstract"] = df_embeddings["abstract"].apply(
        lambda text: encoder.encode_text(str(text))
    )
    elapsed_time = time.time() - temp_time
    temp_time = time.time()
    print("\nJust finished encoding abstracts:", str(timedelta(seconds=elapsed_time)))

    # 3. Encode text fulltexts
    df_embeddings["fulltext"] = df_embeddings["fulltext"].apply(
        lambda text: encoder.encode_text(str(text))
    )
    elapsed_time = time.time() - temp_time
    temp_time = time.time()
    print("\nJust finished encoding fulltexts:", str(timedelta(seconds=elapsed_time)))

    # 4. Encode figures (with captions)
    # For each row in the dataframe, figures is a list of dictionaries with fig_id and caption
    for article_figures in df_embeddings["figures"]:
        article_id = df.index[df["figures"] == article_figures].tolist()[0]  # row ID

        article_figures = ast.literal_eval(article_figures)
        figs, caps = get_images_captions(article_figures)

        embeddings = [
            (encoder.encode_image(figs[i]), encoder.encode_text(caps[i]))
            for i in range(len(figs))
        ]
        df_embeddings.at[article_id, "figures"] = embeddings

    elapsed_time = time.time() - temp_time
    print("\nJust finished encoding figures:", str(timedelta(seconds=elapsed_time)))

    return df_embeddings


def encode_text_topics(encoder, df: pd.DataFrame, with_captions: bool = False):
    """
    Encode the topics in the CSV file using the given encoder.
    """
    if with_captions:
        df_embeddings = df.copy()[["description", "image-descriptions"]]
    else:
        df_embeddings = df.copy()[["description"]]

    # 1. Encode text descriptions
    df_embeddings["description"] = df_embeddings["description"].apply(
        lambda text: encoder.encode_text(text)
    )

    # 2. Encode generated image descriptions
    if with_captions:
        for topics_images_description in df_embeddings["image-descriptions"]:
            topic_id = df.index[
                df["image-descriptions"] == topics_images_description
            ].tolist()[
                0
            ]  # row ID

            topics_images_description = ast.literal_eval(topics_images_description)

            embeddings = [
                encoder.encode_text(description)
                for description in topics_images_description
            ]
            df_embeddings.at[topic_id, "image-descriptions"] = embeddings

    return df_embeddings


def encode_topics(encoder, df: pd.DataFrame, with_captions: bool = False):
    """
    Encode the topics in the CSV file using the given encoder.
    """

    if with_captions:
        df_embeddings = df.copy()[["description", "images", "image-descriptions"]]
    else:
        df_embeddings = df.copy()[["description", "images"]]

    # 1. Encode text descriptions
    df_embeddings["description"] = df_embeddings["description"].apply(
        lambda text: encoder.encode_text(text)
    )

    # 2. Encode images (no captions given)
    for topics_images in df_embeddings["images"]:
        topic_id = df.index[df["images"] == topics_images].tolist()[0]  # row ID

        topics_images = ast.literal_eval(topics_images)
        images = get_images_from_paths(topics_images)

        embeddings = [encoder.encode_image(img) for img in images]
        df_embeddings.at[topic_id, "images"] = embeddings

    # 3. Encode generated image descriptions
    if with_captions:
        for topics_images_description in df_embeddings["image-descriptions"]:
            topic_id = df.index[
                df["image-descriptions"] == topics_images_description
            ].tolist()[
                0
            ]  # row ID

            topics_images_description = ast.literal_eval(topics_images_description)

            embeddings = [
                encoder.encode_text(description)
                for description in topics_images_description
            ]
            df_embeddings.at[topic_id, "image-descriptions"] = embeddings

    return df_embeddings


def get_images_captions(article_figures):
    images = []
    captions = []

    for figure in article_figures:
        path = f"{DATA_DIR_PATH}figures/images/{figure['fig_id']}.jpg"

        if os.path.isfile(path):
            image = Image.open(path)
            images.append(image)
            captions.append(figure["caption"])
        else:
            print(f"Image not found: {path}")
    return images, captions


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
        "type", help="encode topics or articles", choices=("topics", "articles")
    )

    parser.add_argument(
        "encoder",
        help="encoder",
        choices=("CLIP", "LongCLIP", "PubMedCLIP", "Llama"),
        default="CLIP",
    )

    parser.add_argument(
        "-c",
        action="store_true",
        help="encode generated topic image descriptions (only for topics)",
    )

    parser.add_argument(
        "-p",
        "--path",
        type=str,
        help="path to the CSV file to encode",
        default="output_parsed.csv",
    )

    return parser


def main(args):
    print("\nStarted at", datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss\n"))

    if not os.path.isfile(args.path):
        parser.error(f"The path given ({args.path}) is not a file.")

    if not args.path.endswith(".csv"):
        parser.error(f"The file given by the path ({args.path}) is not CSV.")

    # Use the GPU or MPS if available, otherwise stick with the CPU - this is the default for PyTorch
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    print(f"Using device: {DEVICE}\n")

    encoder = None
    if args.encoder == "CLIP":
        encoder = CLIPencoder(DEVICE, "openai/clip-vit-base-patch16")
    elif args.encoder == "LongCLIP":
        encoder = LongCLIPencoder(DEVICE)
    elif args.encoder == "PubMedCLIP":
        encoder = CLIPencoder(DEVICE, "flaviagiammarino/pubmed-clip-vit-base-patch32")
    elif args.encoder == "Llama":
        encoder = LlamaEncoder(DEVICE, "meta-llama/Meta-Llama-3-8B")
    else:
        print("Encoder not supported yet.")
        return

    START_TIME = time.time()
    df_embeddings = None

    current_datetime = datetime.now().strftime("_%Y-%m-%d_%Hh%Mm%Ss")
    output_path = (
        f"{OUTPUT_DIR_PATH}{args.encoder}_embeddings_{args.type}_{current_datetime}.pkl"
    )

    if args.encoder == "Llama":  # Text model - only enconde text
        if args.type == "articles":
            encode_text_articles_chunks(
                encoder,
                args.path,
                output_path,
                f"{OUTPUT_DIR_PATH}{args.encoder}_embeddings_{args.type}_checkpoint_{current_datetime}.txt",
            )
            # df_embeddings = encode_text_articles(encoder, df)
        elif args.type == "topics":
            df = pd.read_csv(args.path, index_col=0)
            df_embeddings = encode_text_topics(encoder, df, args.c)
    else:  # Multimodal model
        df = pd.read_csv(args.path, index_col=0)
        if args.type == "articles":
            df_embeddings = encode_articles(encoder, df)
        elif args.type == "topics":
            df_embeddings = encode_topics(encoder, df, args.c)

    if not (args.encoder == "Llama" and args.type == "articles"):
        print(f"Total time:", str(timedelta(seconds=(time.time() - START_TIME))))

        print(df_embeddings.head())

        current_datetime = datetime.now().strftime("_%Y-%m-%d_%Hh%Mm%Ss")
        output_path = f"{OUTPUT_DIR_PATH}{args.encoder}_embeddings_{args.type}_{current_datetime}.pkl"
        df_embeddings.to_pickle(output_path)
    print(f"Embeddings dataframe written to '{output_path}'.")
    print("\nDone at", current_datetime)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
