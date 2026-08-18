# Tamil Nadu Folk Song Dataset Pipeline (CPU-only, resumable)

Local-first, **CPU-only**, resumable pipeline that scrapes Tamil folk songs from
YouTube category-by-category, runs each candidate through a fixed chain of
quality/relevance gates, and — **only for candidates that reach ACCEPT** —
transcribes the full audio and uploads audio + lyrics.

> Reference state: **Tamil Nadu**. The engine is state-agnostic; point it at a
> different `folk_categories.yaml` to add a new state.

## Pipeline order (fixed — do not reorder)

```
 1  YouTube Search                (yt-dlp, rotated Tamil/English/Tanglish queries)
 2  Metadata Relevance Gate       (RapidFuzz; pre-download)
 3  Download Full Audio           (yt-dlp + FFmpeg -> full-track mono WAV)
 4  Integrity Gate                (ffprobe + soundfile)
 5  Strict Audio Quality Gate     (pure DSP, no model; highest priority)
 6  Music/Speech/Mixed Classifier (PANNs Cnn14, CPU)
 7  Voice Activity Analysis       (Silero VAD; signal only, never sole reject)
 8  Folk-Relevance Gate           (sentence-transformers + CLAP fallback; NON_FOLK only)
 9  Duplicate Detection           (Chromaprint / fpcalc)
10  Weighted Final Decision       (ACCEPT / REVIEW / REJECT; quality hard-override)
11  Full-Audio Transcription      (faster-whisper Tamil, int8, ACCEPT only)
12  Generate lyrics .txt
13  Upload                        (ACCEPT + REVIEW; REJECT never stored)
```

There is **no NSFW gate** and **no NSFW rejection code** anywhere by design.

## CPU-only design

- `device: "cpu"` everywhere (Intel iGPU, no CUDA).
- All models are **loaded once and reused** across files (`ModelManager`).
- GPU-heavy models run behind **`workers.inference: 1`** and
  **`workers.transcription: 1`** so PANNs, sentence-transformers, CLAP and
  faster-whisper never contend for CPU threads simultaneously.
- Whisper runs with **`compute_type: int8`** (required on CPU; no float16/32).
- CLAP is a **secondary** signal, invoked **only** for candidates in the
  uncertain folk-relevance band — not on every file — to keep CPU load down.

## Resume — no database

- **Primary:** before processing a candidate, its output files are checked at the
  target storage path; if present, it is **skipped**.
- **Audit + skip-set:** a JSONL run-log (`state/run_log.jsonl`, one line per
  processed video id + decision) is written for auditing and read once at startup
  so already-processed ids (including REJECTs, which are never stored) aren't
  redone. This is a plain append-only file, **not** a database.
- Duplicate fingerprints persist to `state/fingerprints.jsonl` (also not a DB).

## Storage / upload

Layout (exact, case-sensitive):

```
<root>/<state_name>/<audio_category>/<recording_id>.wav
<root>/<state_name>/<audio_category>/<recording_id>.txt
REVIEW: <root>/<state_name>/review/<audio_category>/<recording_id>.wav[.txt]
```

- `state_name` = `tamil-nadu`; `<audio_category>` = exact `folder_slug`;
  `recording_id` = YouTube video id.
- `storage.backend`: `local` (default, easiest for testing) or `gcs`. For `gcs`
  the first segment of `root` (`kl-workspace`) is the bucket, uploaded via the
  `gcloud storage` CLI to
  `gs://kl-workspace/Data_scraping_version1/Song/<state>/<category>/`.
- Only **ACCEPT** (Gold Data) and **REVIEW** are stored; **REJECT** is never stored.

## Decision buckets & rejection codes

- **ACCEPT** — passes every gate, no hard-fail → transcribed + stored.
- **REVIEW** — borderline anywhere → stored separately, never auto-promoted.
- **REJECT** — standardized codes only: `INTERVIEW, SPEECH, PODCAST, NEWS,
  LECTURE, DISCUSSION, SERMON, STORY_ONLY, NON_SONG, NON_FOLK, CORRUPT_AUDIO,
  TOO_SHORT, MOSTLY_SILENT, EXCESSIVE_CLIPPING, SEVERE_DISTORTION, VERY_LOW_SNR,
  EXCESSIVE_NOISE, AUDIO_DROPOUT, LOW_BANDWIDTH, DUPLICATE_RECORDING,
  DOWNLOAD_FAILED`.

The **quality hard override** is explicit: if the composite quality score is
below `quality_gate.minimum_quality_score`, the candidate is REJECTED regardless
of folk relevance (e.g. folk=0.98, quality=0.21 → REJECT).

## Install

### 1. Native binaries on PATH

- **ffmpeg** + **ffprobe** — download/convert + integrity checks
- **fpcalc** (Chromaprint) — duplicate-detection fingerprinting (optional; if
  missing, dedupe is skipped with a warning, not a crash)

Windows: `winget install Gyan.FFmpeg`; Chromaprint from
<https://acoustid.org/chromaprint>. Ensure the `.exe`s are on `PATH`.

### 2. Python deps

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> **Windows / Tamil output:** `$env:PYTHONUTF8=1` so Tamil script logs correctly.

### 3. Tamil Whisper model (auto-converted on first ACCEPT)

Uses [`inspiredclone101/whisper-tamil-ft-v1`](https://huggingface.co/inspiredclone101/whisper-tamil-ft-v1)
(a transformers checkpoint). faster-whisper needs CTranslate2, so it is converted
once (int8) and cached to `models/whisper-tamil-ft-v1-ct2`. Manual equivalent:

```bash
ct2-transformers-converter --model inspiredclone101/whisper-tamil-ft-v1 \
    --output_dir models/whisper-tamil-ft-v1-ct2 \
    --copy_files tokenizer.json preprocessor_config.json --quantization int8
```

## Configuration

- `config/folk_categories.yaml` — 84-category taxonomy (regenerate with
  `python scripts/gen_taxonomy.py`; per-category gate overrides live here).
- `config/pipeline_config.yaml` — device, storage backend, worker caps, retry
  policy, all thresholds. Nothing quality/decision-related is hard-coded.

## Running

```powershell
$env:PYTHONUTF8=1

# process specific categories (a few candidates each while testing):
python -m tnfolk.orchestrator --categories kummi,oppari --limit 5

# process every category:
python -m tnfolk.orchestrator

# test explicit YouTube URLs under one category (bypasses search):
python -m tnfolk.orchestrator --slug kummi --urls "https://www.youtube.com/watch?v=XXXXXXXXXXX,https://youtu.be/YYYYYYYYYYY"
```

### Test a single gate offline

Gate 1 (metadata relevance) needs no models/network:

```powershell
python tests\test_metadata_gate.py
python tests\test_metadata_gate.py --slug villupattu --title "..." --desc "..."
```

## Project structure

```
config/   folk_categories.yaml, pipeline_config.yaml
scripts/  gen_taxonomy.py
tnfolk/
  config.py, models.py, logging_utils.py, device.py, retry.py, runlog.py
  query_generator.py, youtube_search.py, downloader.py, transcription.py, storage.py
  orchestrator.py
  gates/  metadata_relevance, integrity, audio_quality, music_speech, vad,
          folk_relevance, duplicate, final_scoring
tests/    test_metadata_gate.py
state/    run_log.jsonl, fingerprints.jsonl, staging/   (created at runtime)
```
