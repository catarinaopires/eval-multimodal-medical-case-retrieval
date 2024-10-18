#!/bin/sh

# Parse the TOPICS XML files into CSV files
python3 src/datasets/parse.py "topics" -p "case-based-topics.xml" -o "parsed_topics.csv"

# Parse the ARTICLES XML files into CSV files
python3  src/datasets/parse.py "articles" -r -p "meta-xml/one-file-per-article/" -o "parsed_articles.csv"