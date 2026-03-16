from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.evaluation_metrics import compute_note_metrics, compute_tab_metrics
from backend.src.models import NoteEvent, TabEvent


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate result payloads against GuitarSet-style references.")
    parser.add_argument("results_dir", type=Path, help="Directory containing JSON result payloads.")
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=None,
        help="Directory containing reference JSON payloads keyed by the same filename stem.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the aggregated report as JSON.",
    )
    args = parser.parse_args()

    result_files = sorted(args.results_dir.glob("*.json"))
    if not result_files:
        raise SystemExit("No JSON results found.")

    report = evaluate_directory(result_files, args.reference_dir)
    output = json.dumps(report, indent=2)
    if args.output_json is not None:
        args.output_json.write_text(output, encoding="utf-8")

    print(output)


def evaluate_directory(result_files: list[Path], reference_dir: Path | None) -> dict[str, object]:
    per_file: list[dict[str, object]] = []
    note_counts = []
    avg_confidences = []
    artifact_counts = []
    quantization_errors = []
    engraving_successes = 0
    measure_fill_valid = 0
    note_f1_scores = []
    onset_f1_scores = []
    pitch_accuracies = []
    string_accuracies = []
    fret_accuracies = []
    invalid_rates = []
    density_peaks = []

    for path in result_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload.get("confidence_summary", {})
        note_counts.append(summary.get("note_count", 0))
        avg_confidences.append(summary.get("average_confidence", 0.0))
        artifact_counts.append(len(payload.get("artifacts", [])))
        score_part = payload.get("score_part") or {}
        if payload.get("engraved_output"):
            engraving_successes += 1
        if score_part.get("measure_fill_valid"):
            measure_fill_valid += 1
        quantization_errors.append(score_part.get("quantization_error_sec", 0.0))

        result_entry: dict[str, object] = {
            "file": path.name,
            "note_count": summary.get("note_count", 0),
            "average_confidence": summary.get("average_confidence", 0.0),
            "quantization_error_sec": score_part.get("quantization_error_sec", 0.0),
            "engraved_output": payload.get("engraved_output", False),
        }

        if reference_dir is not None:
            reference_path = reference_dir / path.name
            if reference_path.exists():
                reference_payload = json.loads(reference_path.read_text(encoding="utf-8"))
                reference_notes = load_note_events(reference_payload, "reference_note_events", "note_events")
                predicted_notes = load_note_events(payload, "note_events", "raw_note_events")
                note_metrics = compute_note_metrics(reference_notes, predicted_notes)
                result_entry["note_metrics"] = note_metrics
                note_f1_scores.append(note_metrics["note_f1"])
                onset_f1_scores.append(note_metrics["onset_f1"])
                pitch_accuracies.append(note_metrics["pitch_accuracy"])

                reference_tabs = load_tab_events(reference_payload, "reference_tab_events", "tab_events")
                predicted_tabs = load_tab_events(payload, "tab_events", "reference_tab_events")
                if reference_tabs and predicted_tabs:
                    tab_metrics = compute_tab_metrics(reference_tabs, predicted_tabs)
                    result_entry["tab_metrics"] = tab_metrics
                    string_accuracies.append(tab_metrics["string_accuracy"])
                    fret_accuracies.append(tab_metrics["fret_accuracy"])
                    invalid_rates.append(tab_metrics["invalid_fingering_rate"])
                    density_peaks.append(tab_metrics["voice_density_peak"])
            else:
                result_entry["reference_missing"] = True

        per_file.append(result_entry)

    summary_report: dict[str, object] = {
        "files_evaluated": len(result_files),
        "average_notes": safe_mean(note_counts),
        "average_confidence": safe_mean(avg_confidences),
        "average_artifact_count": safe_mean(artifact_counts),
        "average_quantization_error_sec": safe_mean(quantization_errors),
        "engraving_success_rate": engraving_successes / len(result_files),
        "measure_fill_valid_rate": measure_fill_valid / len(result_files),
        "average_note_f1": safe_mean(note_f1_scores),
        "average_onset_f1": safe_mean(onset_f1_scores),
        "average_pitch_accuracy": safe_mean(pitch_accuracies),
        "average_string_accuracy": safe_mean(string_accuracies),
        "average_fret_accuracy": safe_mean(fret_accuracies),
        "average_invalid_fingering_rate": safe_mean(invalid_rates),
        "average_voice_density_peak": safe_mean(density_peaks),
    }
    return {"summary": summary_report, "files": per_file}


def load_note_events(payload: dict[str, object], primary_key: str, fallback_key: str) -> list[NoteEvent]:
    raw = payload.get(primary_key) or payload.get(fallback_key) or []
    return [NoteEvent.model_validate(item) for item in raw]


def load_tab_events(payload: dict[str, object], primary_key: str, fallback_key: str) -> list[TabEvent]:
    raw = payload.get(primary_key) or payload.get(fallback_key) or []
    return [TabEvent.model_validate(item) for item in raw]


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


if __name__ == "__main__":
    main()
