# [ImageCLEFmed 2013] Case-based retrieval task

## Pipeline

### 0. Prepare datasets
#### Extract datasets

Image and metadata extraction from dataset.

Note: change path in script according to `DATA_DIR_PATH`.

```sh
bash scripts/prepare.sh
```

#### Parse datasets

Parse the topics and articles XML files into CSV files.

```sh
bash scripts/parse.sh
```

```
usage: parse.py [-h] -p PATH [-r] [-o OUTPUT] {topics,articles}

positional arguments:
  {topics,articles}     parse topics or articles

options:
  -h, --help            show this help message and exit
  -p PATH, --path PATH  file/folder path to parse
  -r                    if path is folder, parse whole folder XML files (only for articles)
  -o OUTPUT, --output OUTPUT
                        path to the output CSV file
```

#### Image-to-Text

Generate descriptions for topics or articles images using LLaVa model.

```
usage: i2t-descriptions-llava.py [-h] [-p PATH] {topics,articles}

positional arguments:
  {topics,articles}     generate descriptions for topics or articles

options:
  -h, --help            show this help message and exit
  -p PATH, --path PATH  path to the CSV file to encode
```

### 1./3. Encode datasets

Enconde topics or articles and store the embeddings in a .pkl file.

```
usage: encode.py [-h] [-c] [-p PATH] {topics,articles} {CLIP,LongCLIP,PubMedCLIP,Llama}

positional arguments:
  {topics,articles}     encode topics or articles
  {CLIP,LongCLIP,PubMedCLIP,Llama}
                        encoder

options:
  -h, --help            show this help message and exit
  -c                    encode generated topic image descriptions (only for topics)
  -p PATH, --path PATH  path to the CSV file to generate descriptions of images
```

### 2. Index datasets

Index articles' information and store (each section is indexed in a separate index) using FAISS's GPU implementation of HNSW index with squared Euclidean (L2) distance.

```
usage: index.py [-h] -p PATH [-t] [-o OUTPUT]

options:
  -h, --help            show this help message and exit
  -p PATH, --path PATH  path to the .pkl file to index
  -t                    text_only
  -o OUTPUT, --output OUTPUT
                        filename to store the index
```

### 4./5. Submission Runs

Perform similarity search, using FAISS, of topic's embeddings against the pre-computed indexed embeddings of the articles for retrieval. Perform results fusion (CombSUM or CombMAX or CombMNZ), and, create submission run file.

```
usage: runs.py [-h] -s S -id ID [-t T] [-a A] -i I {d,i,g} {combSUM,combMAX,combMNZ}

positional arguments:
  {d,i,g}               compare with description or images or generated_captions
  {combSUM,combMAX,combMNZ}
                        results fusion method

options:
  -h, --help            show this help message and exit
  -s S                  title, abstract, fulltext, image or caption
  -id ID                ID for the run
  -t T                  topics embeddings .pkl file
  -a A                  articles embeddings .pkl file
  -i I                  index filename
```

#### Search

Search index used in this step (Submission Runs).

```
usage: search_index.py [-h] -i I -t T

options:
  -h, --help  show this help message and exit
  -i I        index path
  -t T        topic number
```

### Statistical testing

Perform statistical testing for all metrics on a set of submissions against the corresponding baselines.

Note: `baselines` and `submission_files` lists must be completed with the file names that want to perform the test on. Assuming that baseline and submission files are in the same order so as to perform the test for files at the same index, and that the files are in the `{OUTPUT_DIR_PATH}/submissions/` directory.

```
python test-all-script.py
```

Note: Only performing paired sample t-test as `perform_paired_t_test_only` is set to `True` in `complete_statistical_testing` function in `statistical_testing.py`. Change `perform_paired_t_test_only` to `False` if you want to check normality and perform Wilcoxon signed-rank or paired sample t-test according to data normality.