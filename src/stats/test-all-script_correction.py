import os
import pathlib

import statistical_testing as st
import multiple_testing as mt
from dotenv import load_dotenv

load_dotenv(dotenv_path=pathlib.Path(__file__).parent / ".env")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
DATA_DIR_PATH = os.getenv("DATA_DIR_PATH")
OUTPUT_DIR_PATH = os.getenv("OUTPUT_DIR_PATH")

metrics = ["map", "bpref", "P.10", "P.30"]
article_sections = ["title", "abstract", "fulltext", "images", "captions"]


SUBMISSION_DIR = OUTPUT_DIR_PATH + "submissions/"


def collect_p_value(output1, output2, runid, metric):
    print(f"\n\nComparing {runid} with baseline-{runid} for metric {metric}:")

    # Effect Size
    scores1 = st.parse_scores(output1)
    scores2 = st.parse_scores(output2)

    result = mt.calculate_cohens_dz_with_ci(scores1, scores2)
    print(
        f"Cohen's d_z: {result['d_z']:.4f}, Standard error: {result['se']:.4f}, 95% CI: [{result['ci_low']:.4f}, {result['ci_high']:.4f}]"
    )
    print(f"Interpretation: {mt.interpret_cohens_dz(result['d_z'])}")

    # Collect
    p_value = st.perform_paired_permutation_test(scores1, scores2)

    return p_value


def statistical_testing_all_metrics(baselines, submission_files):
    """
    Perform statistical testing for all metrics on a set of submissions against the corresponding baselines.
    Baseline and submission files are expected to be in the same order in order to perform the test for files at the same index.
    """

    raw_p_values_dict = {}  # {"metric": [p_value1, p_value2, ...]}
    for metric in metrics:
        for baseline_file, submission_file in zip(baselines, submission_files):
            submission_path = os.path.join(SUBMISSION_DIR, submission_file)

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

                id = submission_path.split("/")[-1].split(".")[0] + "_" + metric
                raw_p_values_dict[id] = collect_p_value(
                    output, baseline_output, id, metric
                )

        mt.holmbonferroni_correction(raw_p_values_dict)


# NOTE: Baseline and submission files are expected to be in the same order so as to perform the test for files at the same index.
#       Assuming that the baselines and submissions are in the {OUTPUT_DIR_PATH}/submissions/ directory.
baselines = []  # TODO: Add baseline files here (derived by runs.py script)
submission_files = []  # TODO: Add submission files here (derived by runs.py script)

statistical_testing_all_metrics(baselines, submission_files)
