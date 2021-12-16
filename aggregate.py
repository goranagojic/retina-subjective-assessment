import json
import re
import sys
from pathlib import Path

import click
import pandas as pd


@click.command()
@click.option('--survey-dir', type=click.Path(exists=True), required=True,
              help="Path to a directory containing json survey results.")
@click.option('--master-fp', type=click.Path(exists=True), required=True, help="Path to a master file.")
@click.option('--out-fp', type=click.Path(exists=False), required=True, help="Where to save a resulting file.")
@click.option('--aggregate-dr', is_flag=True, help="diabetic_retinopathy and background_diabetic_retinopathy will be "
                                                   "treated as a same disease in the resulting file.")
def aggregate(survey_dir, master_fp, out_fp, aggregate_dr):
    """
    Aggregate all survey results into one omnipotent file with groundtruth information.

    :param survey_dir: Where to find all survey result files. It is expected that results are in JSON file format.
    :param master_fp: A path to the master file. Should be in CSV file format.
    :param out_fp: Where to save the aggregated results.
    :param aggregate_dr: Consider diabetic retinopathy and background diabetic retinopathy as same diseases.
    :return:
    """
    input_path: Path = Path(survey_dir)
    output: pd.DataFrame = pd.DataFrame([], [], [
        "surveyID",
        "questionID",
        "doctorID",
        "network",
        "dataset",
        "imageName",
        "answer",
        "correctAnswer",
        "correct",
        "certainty"
    ])

    master = pd.read_csv(master_fp)

    for file in input_path.glob("*.json"):
        with open(file) as fd:
            alterations = []
            loaded = json.load(fd)
            res_count, entries = loaded["ResultCount"], loaded["Data"]
            for entry in entries:
                doctor = entry["doctorID"]
                for key in entry.keys():
                    m = re.match(r's(\d+)-q(\d+)-choice', key)
                    if m:
                        s = m.group(1)
                        q = m.group(2)
                        certainty = entry[f's{s}-q{q}-certainty']
                        img = master[
                            (master['survey_id'] == int(s)) & (master['question_id'] == int(q))].image_filename.array[0]
                        mm = re.match(r'\d+-(\w+)-(\w+).*', img)
                        if not mm:
                            print(f'Cannot decode filename {img}')
                            sys.exit()
                        network = mm.group(1)
                        dataset = mm.group(2)
                        correct_answer = \
                        master[(master['survey_id'] == int(s)) & (master['question_id'] == int(q))].disease_token.array[
                            0]
                        our_answer = entry[key]
                        answer_correct = correct_answer == our_answer
                        if aggregate_dr:
                            if our_answer == "diabetic_retinopathy" and \
                                    correct_answer == "background_diabetic_retinopathy":
                                answer_correct = True
                            if our_answer == "background_diabetic_retinopathy" and \
                                    correct_answer == "diabetic_retinopathy":
                                answer_correct = True

                        alterations.append({
                            "surveyID": int(s),
                            "questionID": int(q),
                            "doctorID": int(doctor),
                            "network": network,
                            "dataset": dataset,
                            "imageName": img,
                            "answer": our_answer,
                            "correctAnswer": correct_answer,
                            "correct": answer_correct,
                            "certainty": int(certainty)
                        })

                df = pd.DataFrame(alterations)
                output = pd.concat([output, df])

    output.to_csv(out_fp, index=False)
    print(output)


if __name__ == '__main__':
    aggregate()
