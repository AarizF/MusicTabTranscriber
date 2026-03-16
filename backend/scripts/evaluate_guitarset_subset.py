from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.evaluation_metrics import compute_note_metrics
from backend.src.models import JobOptions, NoteEvent
from backend.src.transcription import transcribe_audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate raw GuitarSet zip assets directly against the current transcription pipeline."
    )
    parser.add_argument("audio_zip", type=Path, help="Path to GuitarSet audio_mono-mic.zip")
    parser.add_argument("annotation_zip", type=Path, help="Path to GuitarSet annotation.zip")
    parser.add_argument(
        "--sample",
        action="append",
        dest="samples",
        default=[],
        help="Sample base name without suffixes, e.g. 00_Rock1-90-C#_solo",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="If --sample is omitted, evaluate the first N solo mic files alphabetically.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=PROJECT_ROOT / "backend" / "benchmarks" / "guitarset" / "extracted",
        help="Directory used to extract temporary sample assets.",
    )
    parser.add_argument(
        "--profile",
        choices=["fast", "accurate"],
        default="accurate",
        help="Transcription profile to use.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the evaluation report JSON.",
    )
    return parser.parse_args()


def discover_default_samples(audio_zip: Path, limit: int) -> list[str]:
    with zipfile.ZipFile(audio_zip) as archive:
        names = sorted(name for name in archive.namelist() if name.endswith("_solo_mic.wav"))
    return [name.removesuffix("_mic.wav") for name in names[:limit]]


def extract_pair(audio_zip: Path, annotation_zip: Path, sample: str, work_dir: Path) -> tuple[Path, Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    audio_name = f"{sample}_mic.wav"
    annotation_name = f"{sample}.jams"

    with zipfile.ZipFile(audio_zip) as archive:
        archive.extract(audio_name, path=work_dir)
    with zipfile.ZipFile(annotation_zip) as archive:
        archive.extract(annotation_name, path=work_dir)

    return work_dir / audio_name, work_dir / annotation_name


def load_reference_notes(jams_path: Path) -> list[NoteEvent]:
    payload = json.loads(jams_path.read_text(encoding="utf-8"))
    notes: list[NoteEvent] = []
    for annotation in payload.get("annotations", []):
        if annotation.get("namespace") != "note_midi":
            continue
        for item in annotation.get("data", []):
            onset = float(item["time"])
            duration = float(item.get("duration", 0.0))
            midi_pitch = int(round(float(item["value"])))
            notes.append(
                NoteEvent(
                    onset_sec=onset,
                    offset_sec=onset + duration,
                    midi_pitch=midi_pitch,
                    confidence=1.0,
                )
            )
    return sorted(notes, key=lambda event: (event.onset_sec, event.midi_pitch, event.offset_sec))


def evaluate_sample(audio_path: Path, jams_path: Path, profile: str) -> dict[str, object]:
    options = JobOptions(separation_mode="off", transcription_profile=profile)
    branch = transcribe_audio(str(audio_path), "original", options)
    reference_notes = load_reference_notes(jams_path)
    metrics = compute_note_metrics(reference_notes, branch.note_events)
    return {
        "sample": audio_path.stem.removesuffix("_mic"),
        "reference_note_count": len(reference_notes),
        "predicted_note_count": len(branch.note_events),
        "raw_note_count": len(branch.raw_note_events),
        "used_fallback": branch.used_fallback,
        "average_confidence": branch.average_confidence,
        "warnings": branch.warnings,
        "cleanup_warnings": branch.cleanup_warnings,
        "metrics": metrics,
    }


def summarize(results: list[dict[str, object]]) -> dict[str, float]:
    if not results:
        return {}
    def avg(key: str) -> float:
        return mean(float(item["metrics"][key]) for item in results)
    return {
        "samples_evaluated": len(results),
        "average_onset_f1": avg("onset_f1"),
        "average_note_f1": avg("note_f1"),
        "average_pitch_accuracy": avg("pitch_accuracy"),
        "average_onset_precision": avg("onset_precision"),
        "average_onset_recall": avg("onset_recall"),
        "average_note_precision": avg("note_precision"),
        "average_note_recall": avg("note_recall"),
    }


def main() -> None:
    args = parse_args()
    samples = args.samples or discover_default_samples(args.audio_zip, args.limit)
    results: list[dict[str, object]] = []

    for sample in samples:
        audio_path, jams_path = extract_pair(args.audio_zip, args.annotation_zip, sample, args.work_dir)
        result = evaluate_sample(audio_path, jams_path, args.profile)
        results.append(result)
        metrics = result["metrics"]
        print(
            f"{sample}: onset_f1={metrics['onset_f1']:.3f} note_f1={metrics['note_f1']:.3f} "
            f"pitch_acc={metrics['pitch_accuracy']:.3f} fallback={result['used_fallback']}"
        )

    report = {"summary": summarize(results), "results": results}
    output = json.dumps(report, indent=2)
    if args.output_json is not None:
        args.output_json.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
