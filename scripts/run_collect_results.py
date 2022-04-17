import click

from surveys import collect_results


@click.command()
@click.option("--survey-type", type=int, required=False,
              help="A survey type to collect the results for. Can have values 1 or 2. If not specified, both types are "
                   "collected. ")
@click.option("--access_key", type=str, required=True,
              help="SurveyJS access key")
@click.option("--out-dir", type=click.Path(exists=False), required=False,
              help="")
@click.option("--fix/--no-fix", is_flag=True, default=False,
              help="Some surveys exibit bug in column names where survey id placeholder (^_^) is not replaced with an "
                   "actual survey number. If the flag is set to true, placeholders are substituted with an appropriate "
                   "survey number.")
@click.option("--limit", type=int, required=None, help="How many surveys will be downloaded. If not specified, all "
                                                       "surveys with identifiers stated in `survey_ids` dictionary are "
                                                       "downloaded.")
def run_collect_results(survey_type, access_key, fix, limit=None, out_dir=None):
    collect_results(
        survey_type=survey_type,
        access_key=access_key,
        fix=fix,
        limit=limit,
        out_dir=out_dir
    )


if __name__ == "__main__":
    run_collect_results()