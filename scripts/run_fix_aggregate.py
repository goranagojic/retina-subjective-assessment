import click

from surveys import fix_aggregate


@click.command()
@click.option("-s1", "--data-s1", type=click.Path(exists=False), required=True,
              help="Where is the file with aggregated results for the first series.")
@click.option("-s2", "--data-s2", type=click.Path(exists=False), required=True,
              help="Where is the file with aggregated results for the second series.")
@click.option("--output", type=click.Path(exists=False), required=True,
              help="A filepath where merged result will be saved.")
def run_fix_aggregate(data_s1, data_s2, output):
    fix_aggregate(
        survey_data_s1=data_s1,
        survey_data_s2=data_s2,
        out=output
    )


if __name__ == "__main__":
    run_fix_aggregate()