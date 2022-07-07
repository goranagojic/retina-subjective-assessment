import click

from surveys import rank


@click.command()
@click.option("-i", "--input-file", type=click.Path(exists=False),
              help="A path to the file produced by 'aggregate' method.")
@click.option("--images-file", type=click.Path(exists=False), required=False)
@click.option("-o", "--output-dir", type=click.Path(exists=False), required=False)
def run_rank(input_file, output_dir, images_file=None):
    rank(input_file=input_file, images_file=images_file, output_dir=output_dir)


if __name__ == "__main__":
    run_rank()