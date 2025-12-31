import typer
from rich import print
from ga_trader.commands.universe import universe_app
from ga_trader.commands.data import data_app
from ga_trader.commands.ga_cmd import ga_app
from ga_trader.commands.export_cmd import export_app
from ga_trader.commands.validate import validate_app
from ga_trader.commands.indicators_cmd import indicators_app

app = typer.Typer(add_completion=False, help="GA Trader Engine CLI")
app.add_typer(universe_app, name="universe")
app.add_typer(data_app, name="data")
app.add_typer(ga_app, name="ga")
app.add_typer(export_app, name="export")
app.add_typer(validate_app, name="validate")
app.add_typer(indicators_app, name="indicators")


def main() -> None:
    """CLI entrypoint.

    This module is intended to be executed via:
      - python -m ga_trader.cli
    """
    app()


if __name__ == "__main__":
    main()
