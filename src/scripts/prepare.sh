#!/usr/bin/env bash

# TODO: Change path according to DATA_DIR_PATH

# Image extraction
cat data/figures/images.tar.gz* > data/figures/images.tar.gz
tar -xzvf data/figures/images.tar.gz -C data/figures

# Metadata extraction - option 2
tar -xzvf data/meta-xml/one-file-per-article.tar.gz -C data/meta-xml