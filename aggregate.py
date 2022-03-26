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
    :param aggregate_dr: Consider diabetic retinopathy and background diabetic retinopathy as same diseases. If this
        option is active, every occurrence of `background_diabetic_retinopathy` will be replaced with
        `diabetic retinopathy`.
    :return: None
    """
    print(">> Aggregating results data...")

    if aggregate_dr:
        print(">> An option to treat equally 'background_diabetic_retinopathy' and 'diabetic_retionopathy' is ENABLED.")

    input_path: Path = Path(survey_dir)
    output: pd.DataFrame = pd.DataFrame([], [], [
        "surveyID",
        "surveyType",
        "questionID",
        "doctorID",
        "network",
        "dataset",
        "imageName",
        "answer",
        "correctAnswers",
        "correct",
        "certainty"
    ])

    master = pd.read_csv(master_fp)

    for file in input_path.glob("*.json"):

        # name of the survey indicates if it is regular or control (name examples: "regular survey 1.csv",
        # "control survey 1.json", etc.
        if "regular" in str(file):
            survey_type = "regular"
        else:
            survey_type = "control"

        with open(file) as fd:
            alterations = []
            loaded = json.load(fd)
            res_count, entries = loaded["ResultCount"], loaded["Data"]  # ResCount - n entries, Data - entry content
            for entry in entries:
                doctor = entry["doctorID"]
                for key in entry.keys():
                    # parse surveyID and questionID
                    # string example used for parsing: '"s1-q39-choice": "arteriosclerotic_retinopathy"'
                    m = re.match(r's(\d+)-q(\d+)-choice', key)
                    if m:
                        s = m.group(1)
                        q = m.group(2)

                        # parse certainty, string example: `"s1-q39-certainty": 3`
                        certainty = entry[f's{s}-q{q}-certainty']

                        # each question in the survey is associated with one segmentation mask - parse mask info from
                        # master file, string example: "001041-vgan-stare.png" (imgid-network-dataset)
                        img = master[
                            (master['survey_id'] == int(s)) & (master['question_id'] == int(q))].image_filename.array[0]
                        mm = re.match(r'\d+-(\w+)-(\w+).*', img)
                        if not mm:
                            print(f'Cannot decode filename {img}')
                            sys.exit()
                        network = mm.group(1)
                        dataset = mm.group(2)

                        # get correct labels from master file for a segmentation mask, one image can have multiple
                        # diseases associated with it
                        correct_answers = \
                        list(master[(master['survey_id'] == int(s)) & (master['question_id'] == int(q))].disease_token)
                        our_answer = entry[key]

                        # it seems that background_diabetic_retinopathy is subtype or a synonym for diabetic_retinopathy
                        # this is not verified at the moment, but this code piece is here to account on possibility that
                        # these two disease types should be treated as one
                        if aggregate_dr:
                            correct_answers = ["diabetic_retinopathy" if a == "background_diabetic_retinopathy" else a
                                               for a in correct_answers]
                            our_answer = "diabetic_retinopathy" \
                                if our_answer == "background_diabetic_retinopathy" else our_answer

                        # see if survey answer is a correct one
                        # if there are multiple diseases associated with an image, if a doctor has selected one of the
                        # diseases as his answer, the answer is considered to be correct (value 1), otherwise it can be
                        # 'not_applicable' when an image is of insufficient quality (value 0) or incorrect (value 1)
                        if our_answer == 'not_applicable':
                            answer_correct = 0
                        else:
                            answer_correct = 1 if our_answer in correct_answers else -1

                        alterations.append({
                            "surveyID": int(s),
                            "surveyType": survey_type,
                            "questionID": int(q),
                            "doctorID": int(doctor),
                            "network": network,
                            "dataset": dataset,
                            "imageName": img,
                            "answer": our_answer,
                            "correctAnswers": ','.join(correct_answers),
                            "correct": answer_correct,
                            "certainty": int(certainty)
                        })

                df = pd.DataFrame(alterations)
                output = pd.concat([output, df])

    output.to_csv(out_fp, index=False)
    print(output)
    print(">> Done.")


if __name__ == '__main__':
    aggregate()
