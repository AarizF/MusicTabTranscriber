import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import "./styles.css";

type JobStatus =
  | "queued"
  | "preprocessing"
  | "separating"
  | "transcribing"
  | "tabulating"
  | "exporting"
  | "completed"
  | "failed";

type Artifact = {
  kind: "mid" | "gp5" | "pdf" | "musicxml";
  path: string;
  download_url: string;
};

type NoteEvent = {
  onset_sec: number;
  offset_sec: number;
  midi_pitch: number;
  confidence: number;
  technique_hint: string;
  source_branch: string;
};

type TabEvent = NoteEvent & {
  string: number;
  fret: number;
};

type JobPayload = {
  id: string;
  status: JobStatus;
  progress: number;
  message: string;
  error?: string | null;
};

type ResultPayload = {
  warnings: string[];
  cleanup_warnings: string[];
  engraving_warnings: string[];
  engraved_output: boolean;
  inferred_tempo_bpm: number | null;
  inferred_time_signature: string | null;
  measure_count: number;
  beat_confidence: number | null;
  note_level_fusion_used: boolean;
  artifacts: Artifact[];
  branch_scores: Array<{
    branch_name: string;
    raw_note_count: number;
    cleaned_note_count: number;
    used_fallback: boolean;
    score: {
      average_confidence: number;
      onset_alignment: number;
      playable_ratio: number;
      density_score: number;
      duplicate_penalty: number;
      octave_penalty: number;
      overall: number;
    };
  }>;
  confidence_summary: {
    average_confidence: number;
    note_count: number;
    low_confidence_count: number;
    branch_used: string;
  };
  low_confidence_spans: Array<{
    start_sec: number;
    end_sec: number;
    average_confidence: number;
  }>;
  note_events: NoteEvent[];
  tab_events: TabEvent[];
  preprocess: {
    duration_sec: number;
    sample_rate: number;
    used_separation: boolean;
  };
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

function formatSeconds(value: number): string {
  return `${value.toFixed(2)}s`;
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [transcriptionProfile, setTranscriptionProfile] = useState<"fast" | "accurate">("accurate");
  const [separationMode, setSeparationMode] = useState<"auto" | "off" | "demucs">("auto");
  const [preferLowerPositions, setPreferLowerPositions] = useState(true);
  const [leadGuitarMode, setLeadGuitarMode] = useState(true);
  const [tempoOverride, setTempoOverride] = useState("");
  const [timeSignatureOverride, setTimeSignatureOverride] = useState("4/4");
  const [job, setJob] = useState<JobPayload | null>(null);
  const [result, setResult] = useState<ResultPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [musicXml, setMusicXml] = useState<string | null>(null);
  const [previewSvg, setPreviewSvg] = useState<string | null>(null);
  const [pageCount, setPageCount] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);
  const [zoom, setZoom] = useState(48);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [rendererReady, setRendererReady] = useState(false);
  const toolkitRef = useRef<any>(null);

  useEffect(() => {
    if (!job || job.status === "completed" || job.status === "failed") {
      return;
    }

    const interval = window.setInterval(async () => {
      const response = await fetch(`${API_BASE}/jobs/${job.id}`);
      const payload = (await response.json()) as JobPayload;
      setJob(payload);
      if (payload.status === "completed") {
        const resultResponse = await fetch(`${API_BASE}/jobs/${job.id}/result`);
        const resultPayload = (await resultResponse.json()) as ResultPayload;
        setResult(resultPayload);
      }
    }, 1500);

    return () => window.clearInterval(interval);
  }, [job]);

  const accentLabel = useMemo(() => {
    if (!job) {
      return "Ready for lead-guitar extraction";
    }
    if (job.status === "completed") {
      return "Transcription complete";
    }
    if (job.status === "failed") {
      return "Pipeline needs attention";
    }
    return `Pipeline stage: ${job.message}`;
  }, [job]);

  useEffect(() => {
    let active = true;

    async function loadVerovio() {
      const [{ default: createVerovioModule }, { VerovioToolkit }] = await Promise.all([
        import("verovio/wasm"),
        import("verovio/esm"),
      ]);
      const module = await createVerovioModule();
      if (!active) {
        return;
      }
      toolkitRef.current = new VerovioToolkit(module);
      setRendererReady(true);
    }

    void loadVerovio();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    const musicXmlArtifact = result?.artifacts.find((artifact) => artifact.kind === "musicxml");
    if (!musicXmlArtifact) {
      setMusicXml(null);
      setPreviewSvg(null);
      setPageCount(0);
      setCurrentPage(1);
      return;
    }

    let active = true;
    setPreviewError(null);
    fetch(`${API_BASE}${musicXmlArtifact.download_url}`)
      .then((response) => response.text())
      .then((xml) => {
        if (!active) {
          return;
        }
        setMusicXml(xml);
        setCurrentPage(1);
      })
      .catch((fetchError: unknown) => {
        if (!active) {
          return;
        }
        setPreviewError(fetchError instanceof Error ? fetchError.message : "Could not load engraved preview.");
      });

    return () => {
      active = false;
    };
  }, [result]);

  useEffect(() => {
    const toolkit = toolkitRef.current;
    if (!toolkit || !musicXml) {
      return;
    }

    try {
      toolkit.setOptions({
        inputFrom: "xml",
        pageWidth: 1800,
        pageHeight: 2200,
        adjustPageHeight: true,
        scale: zoom,
        header: "none",
        footer: "none",
      });
      toolkit.loadData(musicXml);
      const totalPages = toolkit.getPageCount();
      const safePage = Math.max(1, Math.min(currentPage, totalPages || 1));
      if (safePage !== currentPage) {
        setCurrentPage(safePage);
        return;
      }
      setPageCount(totalPages);
      setPreviewSvg(toolkit.renderToSVG(safePage));
    } catch (renderError) {
      setPreviewError(renderError instanceof Error ? renderError.message : "Could not render engraved preview.");
    }
  }, [currentPage, musicXml, rendererReady, zoom]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file) {
      setError("Choose an audio file first.");
      return;
    }

    setIsSubmitting(true);
    setError(null);
    setJob(null);
    setResult(null);
    setMusicXml(null);
    setPreviewSvg(null);
    setPreviewError(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("transcription_profile", transcriptionProfile);
    formData.append("separation_mode", separationMode);
    formData.append("prefer_lower_positions", String(preferLowerPositions));
    formData.append("lead_guitar_mode", String(leadGuitarMode));
    formData.append("time_signature_override", timeSignatureOverride);
    if (tempoOverride) {
      formData.append("tempo_override_bpm", tempoOverride);
    }
    formData.append("quantization_level", "16th");
    formData.append("engraved_layout", "linked_notation_tab");
    formData.append("export_formats", "mid,gp5,pdf,musicxml");

    try {
      const response = await fetch(`${API_BASE}/jobs`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error("The backend rejected the upload.");
      }
      const payload = (await response.json()) as JobPayload;
      setJob(payload);
    } catch (submissionError) {
      setError(submissionError instanceof Error ? submissionError.message : "Upload failed.");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <main className="shell">
      <section className="hero">
        <p className="eyebrow">{accentLabel}</p>
        <h1>Transcribe the guitar, not the whole mess.</h1>
        <p className="lede">
          Upload a song, quantize the riff into measures, and preview the engraved score as linked notation
          and tab before exporting it.
        </p>
      </section>

      <section className="panel upload-panel">
        <form className="upload-form" onSubmit={handleSubmit}>
          <label className="dropzone">
            <input
              accept="audio/*"
              type="file"
              onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            />
            <span>{file ? file.name : "Choose audio or drop it here"}</span>
            <small>MP3, WAV, or other ffmpeg-readable formats</small>
          </label>

          <div className="controls">
            <label>
              Transcription profile
              <select
                value={transcriptionProfile}
                onChange={(event) => setTranscriptionProfile(event.target.value as "fast" | "accurate")}
              >
                <option value="accurate">Accurate</option>
                <option value="fast">Fast</option>
              </select>
            </label>

            <label>
              Separation mode
              <select value={separationMode} onChange={(event) => setSeparationMode(event.target.value as "auto" | "off" | "demucs")}>
                <option value="auto">Auto</option>
                <option value="off">Off</option>
                <option value="demucs">Demucs</option>
              </select>
            </label>

            <label className="toggle">
              <input
                checked={preferLowerPositions}
                type="checkbox"
                onChange={(event) => setPreferLowerPositions(event.target.checked)}
              />
              Prefer lower fret positions
            </label>

            <label className="toggle">
              <input
                checked={leadGuitarMode}
                type="checkbox"
                onChange={(event) => setLeadGuitarMode(event.target.checked)}
              />
              Lead guitar cleanup
            </label>

            <label>
              Tempo override
              <input
                inputMode="decimal"
                placeholder="Auto"
                type="text"
                value={tempoOverride}
                onChange={(event) => setTempoOverride(event.target.value)}
              />
            </label>

            <label>
              Meter
              <select value={timeSignatureOverride} onChange={(event) => setTimeSignatureOverride(event.target.value)}>
                <option value="4/4">4/4</option>
              </select>
            </label>
          </div>

          <button className="primary-button" disabled={isSubmitting}>
            {isSubmitting ? "Uploading..." : "Start transcription"}
          </button>
        </form>
        {error ? <p className="error-text">{error}</p> : null}
      </section>

      <section className="grid">
        <article className="panel status-panel">
          <h2>Job status</h2>
          {job ? (
            <>
              <div className="status-row">
                <span>{job.status}</span>
                <span>{Math.round((job.progress ?? 0) * 100)}%</span>
              </div>
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${Math.round((job.progress ?? 0) * 100)}%` }} />
              </div>
              <p>{job.message}</p>
              {job.error ? <p className="error-text">{job.error}</p> : null}
            </>
          ) : (
            <p>No active job yet.</p>
          )}
        </article>

        <article className="panel summary-panel">
          <h2>Result snapshot</h2>
          {result ? (
            <>
              <div className="summary-metrics">
                <div>
                  <strong>{result.confidence_summary.note_count}</strong>
                  <span>notes</span>
                </div>
                <div>
                  <strong>{result.confidence_summary.average_confidence.toFixed(2)}</strong>
                  <span>avg confidence</span>
                </div>
                <div>
                  <strong>{result.preprocess.used_separation ? "Yes" : "No"}</strong>
                  <span>used separation</span>
                </div>
                <div>
                  <strong>{result.inferred_tempo_bpm ? Math.round(result.inferred_tempo_bpm) : "--"}</strong>
                  <span>bpm</span>
                </div>
                <div>
                  <strong>{result.measure_count || "--"}</strong>
                  <span>measures</span>
                </div>
                <div>
                  <strong>{result.beat_confidence !== null ? `${Math.round(result.beat_confidence * 100)}%` : "--"}</strong>
                  <span>beat confidence</span>
                </div>
              </div>

              <p className="muted-copy">
                Selected branch: {result.confidence_summary.branch_used}
                {result.note_level_fusion_used ? " with note-level fusion" : ""}
              </p>

              <div className="artifact-list">
                {result.artifacts.map((artifact) => (
                  <a
                    className="artifact-link"
                    href={`${API_BASE}${artifact.download_url}`}
                    key={artifact.kind}
                    target="_blank"
                    rel="noreferrer"
                  >
                    Download {artifact.kind.toUpperCase()}
                  </a>
                ))}
              </div>

              {result.warnings.length > 0 ? (
                <div className="warning-list">
                  {result.warnings.map((warning) => (
                    <p key={warning}>{warning}</p>
                  ))}
                </div>
              ) : null}

              {result.cleanup_warnings.length > 0 ? (
                <div className="warning-list">
                  {result.cleanup_warnings.map((warning) => (
                    <p key={warning}>{warning}</p>
                  ))}
                </div>
              ) : null}

              {result.engraving_warnings.length > 0 ? (
                <div className="warning-list">
                  {result.engraving_warnings.map((warning) => (
                    <p key={warning}>{warning}</p>
                  ))}
                </div>
              ) : null}

              {result.branch_scores.length > 0 ? (
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Branch</th>
                        <th>Overall</th>
                        <th>Confidence</th>
                        <th>Onset</th>
                        <th>Playable</th>
                        <th>Cleaned notes</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.branch_scores.map((branch) => (
                        <tr key={branch.branch_name}>
                          <td>{branch.branch_name}</td>
                          <td>{branch.score.overall.toFixed(2)}</td>
                          <td>{branch.score.average_confidence.toFixed(2)}</td>
                          <td>{branch.score.onset_alignment.toFixed(2)}</td>
                          <td>{branch.score.playable_ratio.toFixed(2)}</td>
                          <td>{branch.cleaned_note_count}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : null}
            </>
          ) : (
            <p>When a job completes, the main artifacts and confidence summary will show up here.</p>
          )}
        </article>
      </section>

      {result ? (
        <section className="panel details-panel">
          <div className="details-header">
            <h2>Tab assignments</h2>
            <p>
              Duration {formatSeconds(result.preprocess.duration_sec)} at {result.preprocess.sample_rate}Hz
            </p>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Time</th>
                  <th>MIDI</th>
                  <th>String</th>
                  <th>Fret</th>
                  <th>Technique</th>
                  <th>Confidence</th>
                </tr>
              </thead>
              <tbody>
                {result.tab_events.slice(0, 24).map((event, index) => (
                  <tr key={`${event.onset_sec}-${index}`}>
                    <td>{formatSeconds(event.onset_sec)}</td>
                    <td>{event.midi_pitch}</td>
                    <td>{event.string}</td>
                    <td>{event.fret}</td>
                    <td>{event.technique_hint}</td>
                    <td>{event.confidence.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      {result ? (
        <section className="panel engraving-panel">
          <div className="details-header">
            <div>
              <h2>Engraved score preview</h2>
              <p>
                {result.engraved_output
                  ? `${result.inferred_time_signature ?? "4/4"} at ${Math.round(result.inferred_tempo_bpm ?? 0)} BPM`
                  : "Engraved output was skipped for this run."}
              </p>
            </div>
            <div className="engraving-controls">
              <button
                className="secondary-button"
                disabled={currentPage <= 1}
                onClick={() => setCurrentPage((page) => Math.max(1, page - 1))}
                type="button"
              >
                Prev
              </button>
              <span>
                Page {pageCount === 0 ? 0 : currentPage} / {pageCount}
              </span>
              <button
                className="secondary-button"
                disabled={pageCount === 0 || currentPage >= pageCount}
                onClick={() => setCurrentPage((page) => Math.min(pageCount, page + 1))}
                type="button"
              >
                Next
              </button>
              <label className="zoom-control">
                Zoom
                <input
                  max="72"
                  min="28"
                  onChange={(event) => setZoom(Number(event.target.value))}
                  type="range"
                  value={zoom}
                />
              </label>
              <button className="secondary-button" onClick={() => window.print()} type="button">
                Print
              </button>
            </div>
          </div>
          {previewError ? <p className="error-text">{previewError}</p> : null}
          {previewSvg && result.engraved_output ? (
            <div className="engraving-canvas" dangerouslySetInnerHTML={{ __html: previewSvg }} />
          ) : (
            <p className="muted-copy">
              {result.engraved_output
                ? "Preparing engraved preview..."
                : "No preview was generated because the beat grid was not stable enough for trustworthy notation."}
            </p>
          )}
        </section>
      ) : null}
    </main>
  );
}

export default App;
