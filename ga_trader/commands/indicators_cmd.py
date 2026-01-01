from __future__ import annotations
import typer
from rich import print
from pathlib import Path
from ga_trader.indicators.registry import IndicatorRegistry
from ga_trader.indicators.pine_harvest import harvest_pine_to_spec

indicators_app = typer.Typer(add_completion=False, help="Open-source Pine indicator management")

@indicators_app.command("init")
def init_indicators(
    root: str = typer.Option("indicators", "--root", help="Indicator root folder (contains specs/ and pine_sources/)"),
):
    rootp = Path(root)
    (rootp / "specs").mkdir(parents=True, exist_ok=True)
    (rootp / "pine_sources").mkdir(parents=True, exist_ok=True)
    (rootp / "python_impl").mkdir(parents=True, exist_ok=True)
    print(f"[green]Initialized indicator folders:[/green] {rootp}")

@indicators_app.command("list")
def list_indicators(
    root: str = typer.Option("indicators", "--root"),
):
    reg = IndicatorRegistry.from_root(Path(root))
    items = reg.list()
    if not items:
        print("[yellow]No indicator specs found.[/yellow]")
        return
    for it in items:
        print(f"- {it.indicator_id}: {it.name} (outputs={list(it.outputs.keys())}, params={list(it.parameters.keys())})")

@indicators_app.command("harvest")
def harvest(
    pine_file: str = typer.Option(..., "--pine", help="Path to an open-source Pine script (.pine)"),
    out_spec: str = typer.Option(None, "--out_spec", help="Output YAML spec path (default: indicators/specs/<id>.yaml)"),
    indicator_id: str = typer.Option(None, "--id", help="Indicator id (default inferred from file name)"),
    root: str = typer.Option("indicators", "--root", help="Indicator root folder"),
):
    pine_path = Path(pine_file)
    if not pine_path.exists():
        raise typer.BadParameter(f"File not found: {pine_path}")
    rootp = Path(root)
    rootp.mkdir(parents=True, exist_ok=True)
    spec = harvest_pine_to_spec(pine_path, indicator_id=indicator_id)
    if out_spec is None:
        out_spec_path = rootp / "specs" / f"{spec['indicator_id']}.yaml"
    else:
        out_spec_path = Path(out_spec)
    out_spec_path.parent.mkdir(parents=True, exist_ok=True)
    import yaml
    out_spec_path.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"[green]Wrote draft spec:[/green] {out_spec_path}")
    print("[yellow]Next step:[/yellow] fill 'outputs' and implement python_impl + pine_snippet mapping as needed.")


@indicators_app.command("import-finrl")
def import_finrl(
    root: str = typer.Option("indicators", "--root", help="Indicator root folder"),
):
    """Generate a set of FinRL-inspired indicator specs (no external deps required to generate specs)."""
    rootp = Path(root)
    rootp.mkdir(parents=True, exist_ok=True)
    from ga_trader.providers.finrl import generate_indicator_specs
    n = generate_indicator_specs(rootp)
    print(f"[green]Wrote {n} indicator specs to[/green] {rootp / 'specs'}")

