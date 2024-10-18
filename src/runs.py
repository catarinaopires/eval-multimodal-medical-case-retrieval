import argparse
import os
import pathlib
from datetime import datetime
from itertools import accumulate

import pandas as pd
from dotenv import load_dotenv

from index.search_index import search_index

load_dotenv(dotenv_path=pathlib.Path(__file__).parent / ".env")
DATA_DIR_PATH = os.getenv("DATA_DIR_PATH")
OUTPUT_DIR_PATH = os.getenv("OUTPUT_DIR_PATH")


def get_list_image_counts(df_articles):
    image_counts = [len(figures) for figures in df_articles["figures"]]
    cumulative_counts = [0] + list(accumulate(image_counts))

    list_of_counts = []

    for range_indexes in zip(cumulative_counts, cumulative_counts[1:]):
        start, end = range_indexes
        list_of_counts.append(list(range(start, end)))

    return list_of_counts


def find_article_index(list_of_counts, figure_index):
    for i in range(len(list_of_counts)):
        if figure_index in list_of_counts[i]:
            return i
    return -1


def results_fusion(results, fusion_method, list_of_counts, df_articles_emb):

    if list_of_counts is None:  # Search item is not images or captions
        results["ann_id"] = results["ann_index"].apply(
            lambda x: df_articles_emb.index[x]
        )
    else:  # Search item is images or captions
        # Change index based on number of images/captions per article
        results["ann_id"] = results["ann_index"].apply(
            lambda x: df_articles_emb.index[find_article_index(list_of_counts, x)]
        )

    if fusion_method == "combMNZ":
        results = results.groupby("ann_id", group_keys=False)["distances"].agg(
            ["sum", "count"]
        )
        results["distances"] = results["count"] * results["sum"]
    elif fusion_method == "combSUM":
        results = results.groupby("ann_id", group_keys=False)["distances"].sum()
    elif fusion_method == "combMAX":
        results = results.groupby("ann_id", group_keys=False)["distances"].max()
    else:
        print("Invalid fusion_method method")

    results = results.reset_index()
    results = results.sort_values(by="distances", ascending=False)

    results = results.head(1000)
    return results


def create_submission_file_images(
    with_topic_generated_caption,
    search_object,
    fusion_method,
    df_topic_emb,
    df_articles_emb,
    run_id,
    index_filename,
    output_filename,
):
    list_of_counts = None
    if search_object == "image" or search_object == "caption":
        list_of_counts = get_list_image_counts(df_articles_emb)

    with open(output_filename, "w") as file:
        for topic_nr, row in df_topic_emb.iterrows():
            images_query_embedding = (
                row["image-descriptions"]
                if with_topic_generated_caption
                else row["images"]
            )

            results = None
            # For each image/generated_caption in topic, search for similar section X in articles
            for image_emb in images_query_embedding:

                single_image_results = search_index(
                    index_filename,
                    image_emb if with_topic_generated_caption else image_emb.T,
                )

                # Combine results for all images in topic
                if results is None:
                    results = single_image_results
                else:
                    results = pd.concat([results, single_image_results], join="outer")

            # Results fusion
            results = results_fusion(
                results, fusion_method, list_of_counts, df_articles_emb
            )

            write_submission_file(topic_nr, results, run_id, file)


def create_submission_file_description(
    search_object,
    fusion_method,
    df_topic_emb,
    df_articles_emb,
    run_id,
    index_filename,
    output_filename,
):
    list_of_counts = None
    if search_object == "image" or search_object == "caption":
        list_of_counts = get_list_image_counts(df_articles_emb)

    with open(output_filename, "w") as file:
        for topic_nr, row in df_topic_emb.iterrows():
            query_embedding = row["description"]

            results = search_index(index_filename, query_embedding)

            if search_object == "image" or search_object == "caption":
                results["ann_id"] = results["ann_index"].apply(
                    lambda x: df_articles_emb.index[
                        find_article_index(list_of_counts, x)
                    ]
                )

            # Results fusion
            results = results_fusion(
                results, fusion_method, list_of_counts, df_articles_emb
            )

            write_submission_file(topic_nr, results, run_id, file)


def write_submission_file(topic_nr, results, run_id, file):
    rank = 1
    for _, results_row in results.iterrows():
        article_id = results_row["ann_id"]
        score = results_row["distances"]
        run_row_submission = f"{topic_nr} {1} {article_id} {rank} {score} {run_id}\n"
        file.write(run_row_submission)
        rank += 1


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "type",
        help="compare with description or images or generated_captions",
        choices=("d", "i", "g"),
    )

    parser.add_argument(
        "-s",
        help="title, abstract, fulltext, image or caption",
        type=str,
        required=True,
    )

    parser.add_argument(
        "fusion",
        help="results fusion method",
        choices=("combSUM", "combMAX", "combMNZ"),
    )

    parser.add_argument("-id", help="ID for the run", required=True)

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

    parser.add_argument("-i", help="index filename", type=str, required=True)

    return parser


def main(args):

    if args.s not in ["title", "abstract", "fulltext", "image", "caption"]:
        print("Invalid search object")
        return

    print("Searching for", args.s, "in", args.type, "with ID", args.id)

    current_datetime = datetime.now().strftime("_%Y-%m-%d_%Hh%Mm%Ss.txt")

    submission_filename = OUTPUT_DIR_PATH + "submissions/" + args.id + current_datetime

    df_topics = pd.read_pickle(args.t)
    df_articles = pd.read_pickle(args.a)

    if args.type == "d":
        create_submission_file_description(
            args.s,
            args.fusion,
            df_topics,
            df_articles,
            args.id,
            args.i,
            submission_filename,
        )
    elif args.type == "i" or args.type == "g":
        create_submission_file_images(
            args.type == "g",
            args.s,
            args.fusion,
            df_topics,
            df_articles,
            args.id,
            args.i,
            submission_filename,
        )
    else:
        print("Invalid type")

    print(f"Submission file created: {submission_filename}")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
