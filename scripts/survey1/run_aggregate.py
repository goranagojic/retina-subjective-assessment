import click
from pathlib import Path

from aggregate import aggregate
from analyze import calculate_scores


@click.command()
@click.option('--survey-dir',
              type=click.Path(exists=True), required=True,
              help="Path to a directory containing json survey results.")
@click.option('--master-fp',
              type=click.Path(exists=True), required=True,
              help="Path to a master file.")
@click.option('--out-fp',
              type=click.Path(exists=False), required=True,
              help="Where to save a resulting file.")
@click.option('--aggregate-diabetic-retinopathy',
              is_flag=True,
              help="diabetic_retinopathy and background_diabetic_retinopathy will be treated as a same disease in "
                   "the resulting file.")
def run_aggregate(survey_dir, master_fp, out_fp, aggregate_diabetic_retinopathy):
    """
    Merge all survey question answers into one file and associate answers with a grade depicting if the answer is
    correct and what is the confidence a doctor providing the answer.

    :param survey_dir: Path to a directory containing json survey results to be aggregated.
    :param master_fp: Path to a master file with groundtruth answers.
    :param out_fp: Where to save a resulting file.
    :param aggregate_diabetic_retinopathy: Consider diabetic retinopathy and background_diabetic retinopathy as to be
        a same diganosis.
    :return: None
    """
    tmp_dir = Path(out_fp).parent
    tmp_fp = tmp_dir / "aggregated-results.csv"

    # aggregate survey results from separate json files into a single file with groundtruth data
    aggregate(
        survey_dir=survey_dir,
        master_fp=master_fp,
        out_fp=tmp_fp,
        aggregate_diabetic_retinopathy=aggregate_diabetic_retinopathy
    )

    # calculate grading scores using information if a doctor has answered question correctly
    # and how confident is he in his answer
    calculate_scores(
        input_file=tmp_fp,
        output_file=out_fp
    )


if __name__ == "__main__":
    run_aggregate()