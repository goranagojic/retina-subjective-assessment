import click

from surveys import to_csv


@click.command()
@click.option("--results-dir", type=click.Path(exists=False), required=True,
              help="")
@click.option("--out-dir", type=click.Path(exists=False), required=True,
              help="")
def run_to_csv(results_dir, out_dir):
    to_csv(
        results_dir=results_dir,
        out_dir=out_dir
    )