from __future__ import annotations

from pathlib import Path


def separate_guitar(input_path: str, output_dir: str) -> str:
    """
    Best-effort guitar stem separation.
    Returns the path to the stem when Demucs is available; otherwise raises.
    """
    try:
        import torchaudio
        from demucs import pretrained
        from demucs.apply import apply_model
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("Demucs is not available in this environment.") from exc

    input_file = Path(input_path).resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    model = pretrained.get_model(name="htdemucs_6s")
    model.cpu()

    wav, sample_rate = torchaudio.load(str(input_file))
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    sources = apply_model(model, wav.unsqueeze(0), device="cpu", split=True)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    stem_names = ["drums", "bass", "other_a", "vocals", "guitar", "other_b"]

    guitar_path = output_root / "guitar_stem.wav"
    for index, stem in enumerate(sources[0]):
        stem_path = output_root / f"{stem_names[index]}.wav"
        torchaudio.save(str(stem_path), stem.cpu(), sample_rate)
        if stem_names[index] == "guitar":
            guitar_path = stem_path

    return str(guitar_path)
