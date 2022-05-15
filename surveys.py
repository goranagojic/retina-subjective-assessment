import click
import itertools
import json
import re

import numpy as np
import requests
import pandas as pd

from string import Template
from pathlib import Path


get_survey_result_template = Template("https://api.surveyjs.io/private/Surveys/getSurveyResults/$id?accessKey=$access_key")

survey_ids = [
    "06c972bd-b950-49b1-a2eb-3ea27cec8b98",     # 1
    "1eecbcfa-d6d7-4617-aae3-6f042fac806f",     # 2
    "314ddd1a-b747-469d-b7c4-8622ba8b02d2",     # 3
    "32b6452b-ec02-461c-95da-89e54c69a44f",     # 4
    "a45594ef-5e39-447f-9f3d-05961eef0c20",     # 5
    "f4d5e5c9-bee4-4dac-8bca-5737a007b53d",     # 6
    "57ddee5f-789d-42c7-ae91-18828ee8620d",     # 7
    "1e571430-c722-46c3-b736-8abaa68908a4",     # 8
    "2bc01724-ce97-41f2-80c6-08ddcad3a57c",     # 9
    "69d24998-6e20-4ca0-ba97-cc1513b247f8",     # 10
    "be2e5123-099f-4591-b952-4f91d076ec36",     # 11
    "f459f8df-b059-43fe-8979-dff4fcbbc2db",     # 12
    "26b0ae8d-e0fe-4906-a5b1-4ab188232aef",     # 13
    "8d9926b3-7497-47cb-af84-d3d666ef0ef8",     # 14
    "e00e6a04-3e83-4774-bee0-18cd7003b674",     # 15
    "3deeff18-c6d7-47ac-b074-0f352eeb0771",     # 16
    "6446e6a5-c1e5-42e9-9453-efc07b943d7e",     # 17
    "8b386623-85f0-4b83-8db3-6eda0580b236",     # 18, 1
    "e20b5187-6be4-4a0b-962c-0d424816e98d",     # 19, 2
    "2d3c688c-e767-423c-bb4e-d860f27d787e",     # 20
    "419e1db1-f43f-40eb-af2e-f8292b64aa6e",     # 21
    "e900f4da-4919-476e-8fef-09d46ca6c2e0",     # 22
    "4f1759d6-bb4c-422d-90be-62e1a5c497bc",     # 23
    "89f567f4-2c4c-4a7c-9378-e44bee902471",     # 24
    "5db72be1-275f-4468-bdb8-e8b6a413d31c",     # 25
    "53e4752a-3096-4a83-9b87-3f254a56973c",     # 26
    "c4e6a943-297c-439f-ad4b-32a0ffeaad91",     # 27
    "c9782492-80dc-421a-97e3-f9badbc8049f",     # 28
    "1d681d01-c7ed-44bd-97bf-14197da6e800",     # 29, 12
    "b94b51b9-e7d3-4ad4-844d-b7942981e88b",     # 30
    "534fa0ef-01c2-4eb1-80ca-5def3b83f19c",     # 31
    "171d3fad-18c1-42d4-a906-be8ffaaeb812",     # 32
    "5225702b-edc8-4484-85e1-cbd09051f348",     # 33
    "502fd2cb-e63d-4a8d-bca2-73001c04aea0",     # 34
    "d053852a-da06-4fac-bd67-fc85a016cfc9",     # 35
    "aa291849-ba47-445c-a3a5-9ca81e0c53b4",     # 36
    "6812832d-145c-49ea-b8be-c51458d0dc34",     # 37
    "ddd99628-cda5-4c0c-89bb-1271df6209b9",     # 38
    "0cf14db1-14f6-4566-a7e6-6f1ce71cab8a",     # 39
    "b0ae9537-bd9d-4058-9c16-256d4d2eb683",     # 40
    "51f37db5-da52-4e54-b7f1-803ec14bbeb6",     # 41
    "ed4de120-79c4-4789-8f12-098e97a54a8c",     # 42
    "c3faa4f1-51b9-4f77-9556-29a653981f6c",     # 43
    "9cfda37d-4b27-46f5-9476-762d9b6cd2e5",     # 44
    "193a3449-ff9a-4951-82f5-734abec4303a",     # 45
    "c8bb6a74-698e-4505-ba76-32d399664ae1",     # 46
    "4b545d4f-a2fd-4403-bfeb-d838018c17b1",     # 47
    "04a9e353-2a24-4d5c-ade7-afe559b52216",     # 48
    "a4a183f5-9d7d-4104-a1c0-b8e277a1e885",     # 49
    "0d5d48ef-b6e3-44c7-aa74-f7df87e2916c",     # 50
    "864a3246-d18a-408f-830d-e6420f49f820",     # 51
    "ebcde336-b4f7-49b9-bbb3-126a37a10b8d",     # 52
    "f08922be-9a8a-4ae2-a6ee-83eb34835d48",     # 53
    "33fcc311-19f3-4139-8356-d022f1b33118",     # 54
    "706b2e98-5489-47c0-ba41-69a09d2ecd76",     # 55
    "f8be8f20-5f57-4645-9e14-fcb1bc8dfd7c",     # 56
    "f3213878-8cf0-49df-821a-9b304b18def0",     # 57
    "3fe1cd3f-359c-48bc-8523-fc6f9ae4d9a0",     # 58
    "849c092b-e4b7-4b94-844d-5b84b72fb9ea",     # 59
    "467f9c6c-f183-47e0-b872-eac675010cf1",     # 60
    "b23bb450-fbf9-445c-b6b1-943ecd059cda",     # 61
    "1a1a5ce7-7704-4c5b-aa43-697319412397",     # 62
    "f7907df7-0f95-4984-9851-29939279828c",     # 63
    "eae25f18-3772-45eb-b2d6-f64e3c7fedcf",     # 64
    "a36eedeb-bd02-48b8-880d-27df2c1ef120",     # 65
    "bcaa3e09-d1cd-44d2-a96a-b21e4bd16e0f",     # 66
    "1558b4d9-0f00-48be-bfc1-3490079d9af2",     # 67
    "dab8b02b-a973-429f-9973-e33b0312d38d",     # 68
    "dce2f0e3-ef6a-4101-9dc7-6bcbb4b15a1a",     # 69
    "9362b518-a4e7-4968-856f-5c3046a1de9e",     # 70
    "4e3de4f2-e9e7-4b84-8d28-eed0fcb982db",     # 71
    "390b0164-46db-48d1-98c7-aae04c909657",     # 72
    "7caabd04-d3ce-4e51-bf80-fa45bb6e0bdf",     # 73
    "e1424354-262f-4eb9-9dc4-7794487712a0",     # 74
    "e4fb433b-b4f0-4ce0-9330-b6802b2f0533",     # 75
]


def collect_results(survey_type, access_key, fix, limit=None, out_dir=None):
    """
    Uses SurveyJS web API to fetch multiple survey results in JSON file format.

    :param survey_type: Valid values are 1 and 2.
    :param access_key: SurveyJS access key.
    :param fix: Question headers in some surveys contain invalid symbols that are corrected if this flag is set to True.
    :param limit: Download first `limit` survey results.
    :param out_dir: Where to save surveys. If does not exist, it will be created.
    :return: None
    """
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Surveys will be saved on the path {out_dir}.")

        metadata = dict()   # saves survey ids the user have filled for each user
        for i, survey_id in enumerate(survey_ids):
            if limit == i+1:
                break

            content = collect_result(survey_id=survey_id, access_key=access_key)

            # if there is a bug in survey column names, fix that by replacing a leftover placeholder sequence ^_^ with
            # a survey number
            responses = content["Data"]     # list of dictionaries

            fixed_responses = list()
            for response in responses:
                # fill metadata dictionary
                key = response["doctorID"]
                if not metadata.get(key):
                    metadata[key] = list()
                metadata[key].append(survey_id)

                # fix question identifiers if needed (replace placeholder ^_^ with survey id)
                if fix:
                    fixed_responses.append({k.replace("^_^", str(i+1)): v for k, v, in response.items()})
            if fix:
                content["Data"] = fixed_responses
            survey_filepath = out_dir / f"{i+1}_{survey_id}.json"
            with open(survey_filepath, "w") as f:
                json.dump(content, f)
                print(f"Saved survey {i+1} with id {survey_id}.")
        print(f"Doctor/survey metadata: ")
        for uid, sids in metadata.items():
            print(f"Doctor {uid} filled {len(sids)} surveys.")
    else:
        for i, survey_id in enumerate(survey_ids):
            print(collect_result(survey_id=survey_id, access_key=access_key))


def collect_result(survey_id, access_key):
    """
    Uses SurveyJS web API to fetch single survey result.

    :param survey_id: Unique survey string that is to be found under ID field in the survey description.
    :param access_key: A private access key.
    :return: Survey result in
    """
    url = get_survey_result_template.substitute({
        "id": survey_id,
        "access_key": access_key
    })
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None


@click.command()
@click.option("--results-dir", type=click.Path(exists=False), required=True,
              help="")
@click.option("--out-dir", type=click.Path(exists=False), required=True,
              help="")
def to_csv(results_dir, out_dir):
    """
    Converts survey results from JSON to CSV file format identical to the CSV downloaded from the SurveyJS website.

    :param results_dir: A directory with survey results in JSON format.
    :param out_dir: A directory where to save results converted into CSV format. Each new file will be named identically
    as a source JSON file.

    :return: None
    """
    results_dir, out_dir = Path(results_dir), Path(out_dir)

    # create output dir if does not exist
    out_dir.mkdir(parents=True, exist_ok=True)

    for r in results_dir.glob("*.json"):
        with open(r, "r") as f:
            content = json.load(f)
            survey_filepath = out_dir / (r.stem + ".csv")
            df = pd.DataFrame.from_dict(content["Data"])

            # column modifications so that saved data is in the same format as the data downloaded from the surveyjs
            # website

            # 1. remove InstanceID column (last in the dataframe)
            df = df.drop(columns=["InstanceId"])

            # 2. column HappenedAt is named as Submitted in the csv downloaded from surveyjs site
            df = df.rename(columns={"HappendAt": "Submitted"}, inplace=False)

            # 3. column Submitted is supposed to be the first column in the file, but in the dataframe it is the last
            cols = df.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            df = df[cols]

            # 4. dates should be in format like 01/16/2022 19:04
            df["Submitted"] = pd.to_datetime(df["Submitted"]).dt.strftime('%m/%d/%Y %H:%M')

            # save to csv file
            df.to_csv(str(survey_filepath), index=False)


def get_network_name(fname):
    networks = ["laddernet", "iternet_uni", "iternet", "saunet", "vgan", "eswanet", "vesselunet", "unet"]
    for network in networks:
        if network in fname:
            return network
    return "UNKNOWN"


def get_dataset_name(fname):
    m = re.match(
        r"(\d+)-.*(laddernet|iternet|saunet|vgan|iternet_uni|eswanet|vesselunet|unet)-(chase|drive|stare).png",
        str(fname))
    if m is None:
        return "UNKNOWN"
    else:
        return m.group(3)


def aggregate(survey_dir, final, images_filepath, observer, output_dir=None):
    aggregated_results = None
    survey_files = Path(survey_dir).glob("*.json")

    all_observers = set()
    for survey_file in survey_files:
        print(f"Processing file {survey_file}...")
        with open(survey_file, "r") as f:
            responses = json.load(f)["Data"]
        for response in responses:
            data = {
                "img_pair": list(),
                "question": list(),
                "img1": list(),
                "img2": list(),
                "answer": list(),
                "survey": list(),
                "observer": list(),
                "is_redundant": list()
            }

            # alternative key is a key that has im1 and im2 substrings switched
            # some of the redundant questions compare same two images but in a different order
            # in this case, it is ok to have different question numbers
            alternative_key = lambda sid, quid, im1id, im2id: f"s{sid}-*-im{im2id}-im{im1id}-impicker"

            # here will be all identifiers corresponding to questions already inserted
            # into data dictionary of a dataframe
            question_lookup = set()
            for key, answer in response.items():
                # if key corresponds to a question id (e.g. s1-q17-im57-im64-impicker)
                m = re.match(r"s(\d+)-q(\d+)-im(\d+)-im(\d+)-impicker", key)
                if m:
                    # just replace substring q<NUMBER> with *, since keys might differ by question number and still
                    # represent the same image combination
                    anonimized_key = re.sub("q[0-9]+", "*", key)

                    survey, question, img1, img2 = m.group(1), m.group(2), m.group(3), m.group(4)
                    data["img_pair"].append(key)
                    data["survey"].append(survey)
                    data["question"].append(question)
                    data["img1"].append(img1)
                    data["img2"].append(img2)
                    data["answer"].append(answer[2:])   # removes string im from image name
                    if anonimized_key in question_lookup:
                        data["is_redundant"].append(True)
                    else:
                        data["is_redundant"].append(False)
                    question_lookup.add(anonimized_key)
                    question_lookup.add(alternative_key(survey, question, img1, img2))
                elif key == "doctorID":
                    obs = answer
                else:
                    pass
            data["observer"] = [obs] * len(data["question"])
            all_observers.add(obs)

            # create dataframe for loaded survey data and specify column types
            df = pd.DataFrame.from_dict(data)
            df = df.astype({
                'img_pair': 'string',
                'survey': int,
                'question': int,
                'img1': int,
                'img2': int,
                'answer': int,
                'observer': int,
                'is_redundant': bool
            })

            if aggregated_results is None:
                aggregated_results = df
            else:
                aggregated_results = pd.concat(
                    [aggregated_results, df],
                    ignore_index=True
                )

    # keep only data for the observers specified
    # aggregated_results["network"] = aggregated_results["answer_fname"].apply(get_network_name)
    if observer[0] == "":
        excluded_observers = list()
    else:
        excluded_observers = [int(obs) for obs in set(all_observers) if obs not in observer]
        print(f"Data associated with {excluded_observers} will be removed from further processing...")

    # remove all entries not associated by specified observers
    aggregated_results = aggregated_results.drop(
        aggregated_results[aggregated_results["observer"].isin(excluded_observers)].index
    )

    if output_dir is None:
        output_dir = survey_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load image data into dataframe and specify column types
    images = pd.read_csv(images_filepath)

    # images.csv might have listed both original images and segmentation masks.
    # only segmentation masks are used in surveys, so leave just that entry type
    images = images[images["type"] == "segmap"]

    aggregated_results = aggregated_results.merge(
        images[["id", "filename"]], how="left", left_on="img1", right_on="id"
    )
    aggregated_results = aggregated_results.merge(
        images[["id", "filename"]], how="left", left_on="img2", right_on="id", suffixes=("_img1", "_img2")
    )
    aggregated_results = aggregated_results.merge(
        images[["id", "filename"]], how="left", left_on="answer", right_on="id"
    )
    aggregated_results = aggregated_results.rename(columns={
        "filename_img1": "img1_fname",
        "filename_img2": "img2_fname",
        "filename": "answer_fname"
    })

    # drop columns used to join dataframes, since they are duplicates of img1 and img2
    # columns from a starting dataframe
    aggregated_results = aggregated_results.drop(["id_img1", "id_img2", "id"], axis=1)

    aggregated_results["network"] = aggregated_results["answer_fname"].apply(get_network_name)
    aggregated_results["dataset"] = aggregated_results["answer_fname"].apply(get_dataset_name)

    survey_filename = "survey_data"
    if final:
        aggregated_results.drop([
            "img1_fname", "img2_fname", "answer_fname"
        ], axis=1, inplace=True)
        survey_filename += "-minimal"

    survey_filename += ".csv"

    aggregated_results.to_csv(output_dir / survey_filename)
    print(f"Resulting file save to {output_dir / survey_filename}")


def rank(input_file,  output_dir, images_file=None):

    results = pd.read_csv(input_file)

    # redundant entries are intended to be used for inter-consistency check, and are not supposed to be included
    # in ballot counting
    results = results[results["is_redundant"] == False]

    rankings = pd.DataFrame(columns=["candidate", "copeland_score", "image", "network", "dataset", "survey"])
    rankings = rankings.astype({
        "candidate": np.int16,
        "copeland_score": np.float16,
        "survey": np.int16
    })

    # https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html
    for survey_id, group in results.groupby("survey"):

        # identifiers of all images involved in comparison are in img1 and img2 columns
        images = pd.concat([group["img1"], group["img2"]], axis=0)

        # image identifiers repeats in img1 and img2 and for voting it is important to have all this identifiers appear
        # just once
        candidates = np.unique(images.values)
        assert len(candidates) == 8            # one image for each of the networks

        ranking = pd.DataFrame({
            "candidate": candidates,
            "copeland_score": np.zeros(len(candidates), dtype=np.float16)
        })
        ranking = ranking.set_index(ranking["candidate"])

        # create empty pairwise comparison table
        candidate_pairs = [i for i in itertools.combinations(iterable=candidates, r=2)]
        xs = [x for x, _ in candidate_pairs]
        ys = [y for _, y in candidate_pairs]
        pairwise_comparison = pd.DataFrame({
            "candidate1": xs,
            "candidate2": ys,
            "candidate1_wins": np.zeros(len(xs), dtype=np.int16),
            "candidate2_wins": np.zeros(len(ys), dtype=np.int16)
        })

        # populate pairwise table
        for c1, c2 in candidate_pairs:
            # print(f"Calculating Copeland score for image pair {c1} and {c2}.")

            # extract from the group all questions where candidate1 and candidate2 compete
            # as img1 and/or img2
            pairs = group[
                np.logical_or(
                    np.logical_and(group["img1"] == c1, group["img2"] == c2),
                    np.logical_and(group["img1"] == c2, group["img2"] == c1)
                )
            ]
            c1_wins = len(pairs[pairs["answer"] == c1])
            c2_wins = len(pairs[pairs["answer"] == c2])

            # write win count for candidate 1
            pairwise_comparison.at[
                np.logical_and(
                    pairwise_comparison["candidate1"] == c1,
                    pairwise_comparison["candidate2"] == c2
                ), 'candidate1_wins'
            ] = c1_wins

            # write win count for candidate 2
            pairwise_comparison.at[
                np.logical_and(
                    pairwise_comparison["candidate1"] == c1,
                    pairwise_comparison["candidate2"] == c2
                ), 'candidate2_wins'
            ] = c2_wins

            # populate table with copeland ranking score
            if c1_wins > c2_wins:
                ranking.at[c1, "copeland_score"] += 1
            elif c2_wins > c1_wins:
                ranking.at[c2, "copeland_score"] += 1
            else:
                ranking.at[c1, "copeland_score"] += 0.5
                ranking.at[c2, "copeland_score"] += 0.5
                # TODO what now?!
                pass

        # if some image is selected by none of the doctors, then it will appear in ranking with score
        # 0 and no other cells populated (such as network name etc.)
        # luckily, there are just a few entries like this so it is easy to manually correct this once ranking file is
        # produced
        uniques = group.drop_duplicates("answer")
        ranking = ranking.reset_index(drop=True)
        ranking = ranking.merge(
            uniques[["answer", "answer_fname", "network", "dataset", "survey"]],
            how="left", left_on="candidate", right_on="answer"
        )
        ranking = ranking.rename(columns={"answer_fname": "image"})
        ranking = ranking.drop(['answer'], axis=1)
        ranking = ranking.sort_values(by="copeland_score")

        rankings = pd.concat([rankings, ranking], ignore_index=True)

        # just some things to write to the console so that user can figure out what is happening
        try:
            image_basename = ranking["image"].iloc[1].split('-')[0]
        except:
            image_basename = ""
        dataset = ranking['dataset'].iloc[1]
        total_votes = ranking['copeland_score'].sum()
        network_ranking = ranking[['copeland_score', 'network']]

        if total_votes != 28:
            print("ALARM GÉNÉRALE!!!")
        print(f"Ranking for image group ({image_basename} - {dataset}) (total votes = {total_votes})")
        print(f"Network ranking:\n{network_ranking}.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = str(output_dir / "ranking.csv")

    rankings.to_csv(output_file)

    print(f"Rankings saved on path {output_file}...")


def fix_aggregate(survey_data_s1, survey_data_s2, out=None):
    """
    This function exists because I mistakenly assumed that I will have all data necessary to compute results in a single
    master file. However, because my surveys were split into two series, there were two master files. That means that
    for the first series of surveys (the first 17) I have to use one master file to merge question/image ids with, and
    for the second series (from 18 till the end) I have to use the other master file. Consequently, I will have two
    output files with results that have to be merged latter in order to compute Copeland score. So, this script does
    exactly that - it merges two survey data files.

    :param survey_data_s1: The path to the merged results for surveys of series 1.
    :param survey_data_s2: The path to the merged results for surveys of series 2.
    :param out: The path to the file produced by merging survey_data_s1 and survey_data_s2.
    :return: Pandas DataFrame of the merged input files.
    """
    data1, data2 = pd.read_csv(survey_data_s1), pd.read_csv(survey_data_s2)
    result = pd.concat([data2, data1], ignore_index=True)

    if out is not None:
        out = Path(out)
        if out.parent.exists():
            result.to_csv(out)
        else:
            print(f"The output will not be saved to path {out}, because it seems that directory {out.parent} "
                  f"does not exist.")

    return result
