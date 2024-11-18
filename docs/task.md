# [ImageCLEFmed 2013](https://www.imageclef.org/2013/medical): Case-based retrieval task

## Description
This task was first introduced in 2009. This is a more complex task, but one that we believe is closer to the clinical workflow. <b> In this task, a case description, with patient demographics, limited symptoms and test results including imaging studies, is provided (but not the final diagnosis). The goal is to retrieve cases including images that might best suit the provided case description.</b> Unlike the ad-hoc task, the unit of retrieval here is a case, not an image. For the purposes of this task, a "case" is a PubMed ID corresponding to the journal article (case = article). In the results submissions the article DOI should be used as several articles do not have PubMed IDs nor Article URLs.

<u> For each topic, retrieve the cases including images that might best suit the provided case description. </u>

## Data
Database distribution includes an XML file and a compressed file containing the over 300,000 images of 75,000 articles of the biomedical open access literature.

30 case-based topics with images are provided, where the retrieval unit is a case, not an image.

## Submission
- trec_eval format 
- qrels provided

Information requested:
1. What was used for the retrieval: Image, text or mixed (both)
2. Was other training data used?
3. Run type: Automatic, Manual, Interactive
4. Query Language

### trec_eval format

The format for submitting results is based on the trec_eval program (http://trec.nist.gov/trec_eval/) as follows:

```
1 1 27431 1 0.567162 OHSU_text_1
1 1 27982 2 0.441542 OHSU_text_1
.............
1 1 52112 1000 0.045022 OHSU_text_1
2 1 43458 1 0.9475 OHSU_text_1
.............
25 1 28937 995 0.01492 OHSU_text_1
```

where:

- The first column contains the <u> topic number </u>.
- The second column is <u> always 1 </u>.
- The third column is the  <u> full article DOI for the case-based topics </u>.
- The fourth column is the <u> ranking for the topic (1-1000) </u>.
- The fifth column is the <u> score assigned by the system </u>.
- The sixth column is the <u> identifier for the run </u> and should be the same in the entire file.

`{topic_number} 1 {article_DOI} {rank} {score} {run_ID}`

### Several key points for submitted runs are:

- The topic numbers should be consecutive and complete.
- Case-based topics have to be submitted in separate files.
- The score should be in decreasing order (i.e. the case at the top of the list should have a higher score than cases at the bottom of the list).
- Up to (but not necessarily) 1000 images can be submitted for each topic.
- Each topic must have at least one image.
- Each run must be submitted in a single file. Files should be pure text files and not be zipped or otherwise compressed.

## Evaluation

ATTENTION: trec_eval ignores rank. "trec_eval does its own sort of the results by *decreasing score* (which can have ties). trec_eval does not break ties using the rank field, either. A user who wants trec_eval to evaluate a specific ordering of the documents must ensure that the *SCORES* reflect the desired ordering."

```
trec_eval [-h] [-q] [-m measure[.params] [-c] [-n] [-l <num>]
   [-D debug_level] [-N <num>] [-M <num>] [-R rel_format] [-T results_format]
   rel_info_file  results_file
```

Usage example: `./trec_eval QRELS.txt RESULTS.txt`