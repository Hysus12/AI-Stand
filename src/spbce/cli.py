from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from spbce.inference.comparison import compare_project_to_reference, export_comparison_report
from spbce.inference.mvp import MvpInferenceEngine
from spbce.schema.project import ProjectOutput

app = typer.Typer(help="SPBCE Pilot MVP CLI")


@app.command("run-project")
def run_project(
    input_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")],
    synthetic_count: Annotated[
        int | None,
        typer.Option(
            "--synthetic-count",
            help="Override synthetic respondent count from the project spec.",
        ),
    ] = None,
    env_file: Annotated[
        str | None,
        typer.Option(
            "--env-file",
            help="Optional dotenv file for DeepSeek / OpenAI-compatible credentials.",
        ),
    ] = None,
    disable_remote_llm: Annotated[
        bool,
        typer.Option(
            "--disable-remote-llm",
            help="Skip remote DeepSeek calls and force immediate heuristic fallback.",
        ),
    ] = False,
) -> None:
    engine = MvpInferenceEngine(env_file=env_file, disable_remote_llm=disable_remote_llm)
    run_result = engine.run_project_file(
        input_path=input_path,
        output_dir=output_dir,
        synthetic_respondent_count=synthetic_count,
    )
    typer.echo(f"project_id={run_result.project_output.project_id}")
    for artifact_name, artifact_path in sorted(run_result.export_manifest.items()):
        typer.echo(f"{artifact_name}={artifact_path}")


@app.command("compare-project")
def compare_project(
    project_spec: Annotated[Path, typer.Argument(exists=True, readable=True)],
    project_result: Annotated[Path, typer.Argument(exists=True, readable=True)],
    reference_result: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")],
) -> None:
    engine = MvpInferenceEngine()
    project = engine.load_project_input(project_spec)
    project_output = ProjectOutput.model_validate_json(project_result.read_text(encoding="utf-8"))
    reference_payload = json.loads(reference_result.read_text(encoding="utf-8"))
    comparison_report = compare_project_to_reference(
        project=project,
        project_output=project_output,
        reference_payload=reference_payload,
    )
    manifest = export_comparison_report(comparison_report, output_dir=output_dir)
    for artifact_name, artifact_path in sorted(manifest.items()):
        typer.echo(f"{artifact_name}={artifact_path}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
