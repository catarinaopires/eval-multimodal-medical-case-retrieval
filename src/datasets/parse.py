import argparse
import datetime
import os
import pathlib
import re

import nltk.data
import pandas as pd
import untangle
from dotenv import load_dotenv

load_dotenv(dotenv_path=pathlib.Path(__file__).parent.parent / ".env")
DATA_DIR_PATH = os.getenv("DATA_DIR_PATH")
OUTPUT_DIR_PATH = os.getenv("OUTPUT_DIR_PATH")

# Regex patterns
YEARS_RE = re.compile("(?i)[1-9]?[0-9]?(\-| )(year|yo)")
DIGITS_RE = re.compile("(?i)[1-9]?[0-9]")
MALE_RE = re.compile("(?i)(male|man|boy)")
FEMALE_RE = re.compile("(?i)(female|woman|girl)")


def tokenizeAge(text):
    """
    Extracts the age from the input string.
    """
    years = YEARS_RE.search(text)
    if years is not None:
        age = DIGITS_RE.search(years.group())
        if age is not None:
            return int(age.group())

    return 0


def tokenizeGender(text):
    """
    Extracts the gender from the input string.
    """
    female = FEMALE_RE.search(text)
    male = MALE_RE.search(text)
    if female is not None:
        return "F"
    elif male is not None:
        return "M"
    else:
        return "U"


# Functions to parse XML files
def xml_article_file_to_object(xml_file_path):
    """
    Parse article from the XML file and return a dictionary with the article information.
    """
    data = untangle.parse(str(xml_file_path))

    if data.get_elements(name="article") == []:
        raise ValueError("The XML file does not contain the article tag.")

    article = {}

    article["id"] = data.article.get_attribute("doi")
    article["title"] = data.article.get_elements(name="title")[0].cdata
    article["abstract"] = data.article.get_elements(name="abstract")[0].cdata
    article["fulltext"] = data.article.get_elements(name="fulltext")[0].cdata

    authors = []
    if data.article.authors.get_elements(name="author") != []:
        for author in data.article.authors.author:
            authors.append(author.cdata)
    article["authors"] = authors

    images = []
    for image in data.article.figures.get_elements(name="figure"):
        img = {}
        img["fig_id"] = image.get_attribute("iri")
        img["caption"] = image.get_elements(name="caption")[0].cdata
        images.append(img)
    article["figures"] = images

    return article


def xml_topics_file_to_object(xml_file_path):
    """
    Parse topics from the XML file and return a dictionary with the topics information.
    """
    data = untangle.parse(str(xml_file_path))

    if data.get_elements(name="TOPICS") == []:
        raise ValueError("The XML file does not contain the TOPICS tag.")

    topics = {"id": [], "description": [], "images": []}

    for topic in data.TOPICS.TOPIC:
        topics["id"].append(topic.get_elements(name="ID")[0].cdata)
        topics["description"].append(topic.get_elements(name="EN_DESCRIPTION")[0].cdata)

        images = []
        for image_path in topic.get_elements(name="image"):
            images.append(image_path.cdata)
        topics["images"].append(images)

    return topics


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "type", help="parse topics or articles", choices=("topics", "articles")
    )

    parser.add_argument(
        "-p", "--path", type=str, help="file/folder path to parse", required=True
    )
    parser.add_argument(
        "-r",
        help="if path is folder, parse whole folder XML files (only for articles)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="path to the output CSV file",
        default="output_parsed.csv",
    )

    return parser


def main(args):
    if args.r and args.type != "articles":
        parser.error(
            "The -r argument requires the 'articles' and not 'topics' argument."
        )

    df = None
    path = DATA_DIR_PATH + args.path

    if not args.r and not str(path).endswith(".xml"):
        parser.error("The file given by the path is not XML.")

    if args.type == "articles":
        articles = []
        if args.r:
            if not os.path.isdir(path):
                parser.error(
                    "The -r argument requires a folder path. Path given is not a folder."
                )

            article_paths = list(filter(lambda p: p.endswith(".xml"), os.listdir(path)))
            path = str(path) + "/"

            print("Found %d articles." % len(article_paths))

            articles = []
            for article_path in article_paths:
                try:
                    articles.append(xml_article_file_to_object(path + article_path))
                except ValueError as ve:
                    print("Skipping file %s. %s" % (article_path, ve.args[0]))
        else:
            try:
                articles = [xml_article_file_to_object(path)]
            except ValueError as ve:
                print(ve.args[0])
                return

        df = pd.DataFrame.from_records(
            articles,
            columns=["id", "title", "authors", "abstract", "fulltext", "figures"],
        ).set_index("id")

    elif args.type == "topics":
        if not os.path.isfile(path):
            parser.error("The path given is not a file.")

        try:
            topics = xml_topics_file_to_object(path)
        except ValueError as ve:
            print(ve.args[0])
            return

        print("Found %d topics." % len(topics["id"]))

        df = pd.DataFrame(topics, columns=["id", "description", "images"]).set_index(
            "id"
        )

        # Extract demographics data (age, gender)
        # Pull out the first sentence, since it has the demographics in this data set.
        sentence_detector = nltk.data.load("tokenizers/punkt/english.pickle")
        df["first_sentence"] = df["description"].map(
            lambda t: sentence_detector.tokenize(t)[0]
        )
        df["age"] = df["first_sentence"].map(lambda t: tokenizeAge(t))
        df["gender"] = df["description"].map(lambda t: tokenizeGender(t))
        df["minBirthYear"] = df["age"].map(
            lambda a: int(datetime.datetime.now().year) - a
        )

    if df is not None:
        if not os.path.exists(OUTPUT_DIR_PATH):
            os.makedirs(OUTPUT_DIR_PATH)

        args.output = OUTPUT_DIR_PATH + args.output
        df.to_csv(args.output, encoding="utf-8")
        print("Output written to %s" % args.output)


if __name__ == "__main__":
    """
    usage: parse.py [-h] -p PATH [-r] [-o OUTPUT] {topics,articles}

    positional arguments:
    {topics,articles}     parse topics or articles

    options:
    -h, --help            show this help message and exit
    -p PATH, --path PATH  file/folder path to parse
    -r                    if path is folder, parse whole folder XML files (only for articles)
    -o OUTPUT, --output OUTPUT path to the output CSV file
    """
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
