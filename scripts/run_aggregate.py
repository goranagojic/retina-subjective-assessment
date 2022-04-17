import click

from surveys import aggregate


@click.command()
@click.option("--survey-dir", type=click.Path(exists=False))
@click.option("-o", "--output-dir", type=click.Path(exists=False), required=False)
@click.option("-i", "--images-filepath", type=click.Path(exists=False), required=False, default="images.csv")
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