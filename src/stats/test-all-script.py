import os
import pathlib

import statistical_testing as st
from dotenv import load_dotenv

load_dotenv(dotenv_path=pathlib.Path(__file__).parent / ".env")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
DATA_DIR_PATH = os.getenv("DATA_DIR_PATH")
OUTPUT_DIR_PATH = os.getenv("OUTPUT_DIR_PATH")

metrics = ["map", "bpref", "P.10", "P.30"]
article_sections = ["title", "abstract", "fulltext", "images", "captions"]


SUBMISSION_DIR = OUTPUT_DIR_PATH + "submissions/"
EXPERIMENT_NAME = (
    "exp1"  # TODO: change according with experiment name given in the submission file
)

baselines = []  # TODO: Add baseline files here (derived by runs.py script)
submission_files = []  # TODO: Add submission files here (derived by runs.py script)


i = 0
for submission_file in submission_files:
    submission_path = os.path.join(SUBMISSION_DIR, submission_file)

    # Single experiment at time
    if EXPERIMENT_NAME in submission_file:
        baseline_file = baselines[i]

        print(
            f"\n** - Running statistical testing for {submission_file} against {baseline_file}"
        )

        for metric in metrics:
            baseline_output = os.popen(
                f"../../trec_eval-9.0.7/trec_eval -m {metric} -q -M 1000 {DATA_DIR_PATH}qrels/qrel_2013_case_based.txt {SUBMISSION_DIR}{baseline_file}"
            ).read()

            # Run trec_eval on the submission file
            output = os.popen(
                f"../../trec_eval-9.0.7/trec_eval -m {metric} -q -M 1000 {DATA_DIR_PATH}qrels/qrel_2013_case_based.txt {submission_path}"
            ).read()

            # Run statistical
            st.complete_statistical_testing(
                baseline_output,
                output,
                submission_path.split("/")[-1].split(".")[0] + "_" + metric,
            )

        i += 1
