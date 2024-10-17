import argparse
import os
import pathlib
import time
from datetime import datetime, timedelta

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(dotenv_path=pathlib.Path(__file__).parent.parent / ".env")
INDEXES_DIR_PATH = os.getenv("OUTPUT_DIR_PATH") + "indexes/"

START_TIME = time.time()


def get_faiss_hnsw_index(vectors: pd.DataFrame):
    """
    Implementation of HNSW using the Facebook AI Similarity Search (Faiss) library
    """
    hnsw_config = {
        "d": vectors.shape[1],  # vector size
        "M": 32,  # number of neighbors we add to each vertex on insertion
        "efSearch": 32,  # number of entry points (neighbors) we use on each layer
        "efConstruction": 32,  # number of entry points used on each layer during construction
    }

    index = faiss.IndexHNSWFlat(hnsw_config["d"], hnsw_config["M"])

    index.hnsw.efConstruction = hnsw_config["efConstruction"]
    index.hnsw.efSearch = hnsw_config["efSearch"]

    faiss.normalize_L2(vectors)
    index.add(vectors)

    return index


def store_index(index, filename: str):
    path = INDEXES_DIR_PATH + datetime.now().strftime("_%Y-%m-%d_%Hh%Mm%Ss_") + filename
    faiss.write_index(index, path)
    print(f"Index stored in '{path}'.")


def index_articles(df: pd.DataFrame, output_filename: str, is_text_only: bool = False):
    """
    Index the articles in the dataframe using the Faiss library.
    Store the index in the given filename.
    """

    # Title
    vectors_titles = np.vstack(
        df["title"].values
    )  # Stack all embeddings into a single numpy array
    index_vectors(vectors_titles, "title_" + output_filename)

    elapsed_time = time.time() - START_TIME
    temp_time = time.time()
    print("Just finished indexing titles:", str(timedelta(seconds=elapsed_time)), "\n")

    # Abstract
    vectors_abstracts = np.vstack(df["abstract"].values)
    index_vectors(vectors_abstracts, "abstract_" + output_filename)

    elapsed_time = time.time() - temp_time
    temp_time = time.time()
    print(
        "Just finished indexing abstracts:", str(timedelta(seconds=elapsed_time)), "\n"
    )

    # Fulltext
    vectors_fulltexts = np.vstack(df["fulltext"].values)
    index_vectors(vectors_fulltexts, "fulltext_" + output_filename)

    elapsed_time = time.time() - temp_time
    temp_time = time.time()
    print(
        "Just finished indexing fulltexts:", str(timedelta(seconds=elapsed_time)), "\n"
    )

    # Figures (with captions)
    vectors_images = []
    vectors_captions = []

    for article_figs_cap in df["figures"].to_list():
        if is_text_only:
            vectors_captions += [elem[0] for elem in article_figs_cap]
        else:
            vectors_images += [elem[0] for elem in article_figs_cap]
            vectors_captions += [elem[1] for elem in article_figs_cap]

    # Images
    if is_text_only:
        index_vectors(np.array(vectors_captions), "captions_" + output_filename)
    else:
        index_vectors(np.array(vectors_images)[:, :, 0], "images_" + output_filename)
        index_vectors(
            np.array(vectors_captions)[:, 0, :], "captions_" + output_filename
        )
    elapsed_time = time.time() - temp_time
    print("Just finished indexing figures:", str(timedelta(seconds=elapsed_time)), "\n")


def index_vectors(vectors, output_filename: str):
    index = get_faiss_hnsw_index(vectors)
    store_index(index, output_filename)


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p", "--path", type=str, help="path to the .pkl file to index", required=True
    )

    parser.add_argument("-t", action="store_true", help="text_only")

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="filename to store the index",
        default="stored_index.index",
    )

    return parser


def main(args):
    print("\nStarted at", datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss\n"))

    START_TIME = time.time()

    df = None
    if os.path.isdir(args.path):
        for filename in os.listdir(args.path):
            if filename.endswith(".pkl"):
                single_df = pd.read_pickle(args.path + "/" + filename)

                if df is None:
                    df = single_df
                else:
                    df = pd.concat([df, single_df], ignore_index=True)

        print(f"Loaded {len(df)} embeddings.")
        print(df.head())
    elif os.path.isfile(args.path):
        df = pd.read_pickle(args.path)

    index_articles(df, args.output, args.t)
    print(f"Total time:", str(timedelta(seconds=(time.time() - START_TIME))))


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
