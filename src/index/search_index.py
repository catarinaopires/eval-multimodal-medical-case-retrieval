import argparse
import os
import pathlib

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(dotenv_path=pathlib.Path(__file__).parent.parent / ".env")
DATA_DIR_PATH = os.getenv("DATA_DIR_PATH")
OUTPUT_DIR_PATH = os.getenv("OUTPUT_DIR_PATH")


def search_index(index_filename, query_embedding, k=None):
    index = faiss.read_index(index_filename)

    _vector = np.array(query_embedding)
    faiss.normalize_L2(_vector)

    if k is None or k > index.ntotal:
        k = index.ntotal  # search all neighbors - k set to total

    distances, ann = index.search(_vector, k=k)

    # Sort
    results = pd.DataFrame({"distances": -distances[0], "ann_index": ann[0]})
    results.drop_duplicates(subset="ann_index", inplace=True)
    results = results.sort_values(by="distances", ascending=False)

    return results


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", help="index path", required=True)
    parser.add_argument("-n", help="topic number", required=True)

    parser.add_argument(
        "-t",
        help="topics embeddings .pkl file",
        type=str,
        default="CLIP_embeddings_topics.pkl",
    )

    parser.add_argument(
        "-a",
        help="articles embeddings .pkl file",
        type=str,
        default="CLIP_embeddings_articles.pkl",
    )

    return parser


def main(args):
    index_filename = args.i
    topic_id = args.n

    df_topics = pd.read_pickle(OUTPUT_DIR_PATH + args.t)
    df_articles = pd.read_pickle(OUTPUT_DIR_PATH + args.a)

    results = search_index(
        index_filename, df_topics.at[df_topics.index[topic_id], "description"]
    )  # TODO: adapt to other sections; currently just for topic description section
    results["ann_id"] = results["ann_index"].apply(lambda x: df_articles.index[x])

    print(results.head())

    return results


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
