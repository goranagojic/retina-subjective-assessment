import click

from surveys import aggregate


@click.command()
@click.option("--survey-dir", type=click.Path(exists=False),
              help="Where to find surveys that are to be aggregated. Surveys must be in json format.")
@click.option("-o", "--output-dir", type=click.Path(exists=False), required=False,
              help="Where to save aggregated results. The output file is named survey_data.csv.")
@click.option("-i", "--images-filepath", type=click.Path(exists=False), required=False, default="images.csv",
              help="A path to the file with data about segmentation masks, one mask per line. Each line must have id,"
                   "mask filename, dataset, type (segmask in this case), and disease token.")
@click.option("--final/--no-final", is_flag=True, default=False,
              help="Remove/keep some image related columns.")
@click.option("--observer", type=str, multiple=True, default=[""], required=False,
              help="Specify observers whose results you wish to include in the output file.")
def run_aggregate(survey_dir, final, images_filepath, observer, output_dir=None):
    aggregate(
        survey_dir=survey_dir,
        final=final,
        images_filepath=images_filepath,
        observer=observer,
        output_dir=output_dir
    )


if __name__ == "__main__":
    run_aggregate()