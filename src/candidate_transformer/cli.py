"""Typer CLI for the candidate transformer."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from .pipeline import CandidateTransformer
from .projection import ProjectionConfig, ProjectionEngine

app = typer.Typer(add_completion=False, help="Transform multi-source candidate data into a canonical profile.")


@app.command()
def transform(
    input_path = typer.Argument(..., exists=True, readable=True, help="Input file"),
    projection = typer.Option(None, "--projection", exists=True, readable=True, help="YAML projection config"),
    output = typer.Option(None, "--output", help="Write JSON output to a file"),
) -> None:
    """Run the transformation pipeline for one or more inputs."""

    transformer = CandidateTransformer()
    input_path = Path(str(input_path))
    result = transformer.transform_paths([input_path])
    payload: dict[str, object]

    if projection is not None:
        projection_config = ProjectionConfig.from_yaml(projection)
        projection_result = ProjectionEngine(projection_config).project(result.profile)
        payload = {
            "profile": result.profile.model_dump(mode="json"),
            "projection": projection_result.model_dump(mode="json"),
            "graph_stats": result.graph_stats,
        }
    else:
        payload = {
            "profile": result.profile.model_dump(mode="json"),
            "graph_stats": result.graph_stats,
        }

    if output is not None:
        output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        typer.echo(f"Wrote transformation result to {output}")
        return

    typer.echo(json.dumps(payload, indent=2, default=str))


@app.command()
def inspect_graph(input_path = typer.Argument(..., exists=True, readable=True)) -> None:
    """Print graph statistics for a single input."""

    transformer = CandidateTransformer()
    result = transformer.transform_paths([Path(str(input_path))])
    typer.echo(result.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
