# Evaluating dense model-based approaches for Multimodal Medical Case retrieval

## Installation and Use

1. Import `imageclefenv` environment: `conda env create --file imageclefenv.yaml`

2. Activate environment: `conda activate imageclefenv`

Then follow the usage instructions [here](src/README.md) to run the code of each step of the pipeline, after setting the environment variables explained below.

![Workflow pipeline of the retrieval system](./docs/pipeline.svg)
**Workflow pipeline of the retrieval system:** dataset collection and article encoding (step 1), storage and indexing of embeddings (step 2), query encoding (step 3), results fusion (step 4), and retrieval of a final ranked list of results (step 5). The query workflow is turquoise, whereas the articles' workflow is black.

Note: To use LongCLIP, download the checkpoint of the model [LongCLIP-B](https://huggingface.co/BeichenZhang/LongCLIP-B) and place it under `./checkpoints`.

## Environment Variables

In order to run the project, you need to set all of the following environment variables in a `.env` file:

- **HF_TOKEN**: HuggingFace token.
- **HF_HOME**: HuggingFace home directory, where data will be locally stored.

- **DATA_DIR_PATH**: Dataset directory (file structure detailed next).
- **OUTPUT_DIR_PATH**: Output directory.


## Dataset

Case-based retrieval task from [ImageCLEFmed 2013](https://www.imageclef.org/2013/medical) Task detailed [here](docs/task.md).

The data directory must follow the following structure:

```
data/
│   case-based-topics.xml   
│
└───CaseQueryImages2013/
│   │   01_1.jpg
│   │   01_2.jpg
|   |   ...
│
└───figures/
|   │   scrt68-3.jpg
|   │   rr11-4.jpg
|   |   ...
|
└───meta-xml/
│   │
│   └───one-file-per-article/
│       │   article_126217.xml
│       │   article_29062.xml
│       │   ...
│
└───qrels/
    │   qrel-2013-case_based.txt
```
