"""Retry transcription for audio files missing .txt transcriptions.

Works on both:
1. Local directories (e.g. --folder "path/to/tamil-nadu/ammanai" or any folder)
2. GCS storage buckets (e.g. --gcs --category ammanai or --gcs --all)

Usage examples:
    # 1. Process a specific local folder:
    python scripts/retry_transcription.py --folder "tamil-nadu/ammanai"

    # 2. Process an entire local directory recursively:
    python scripts/retry_transcription.py --folder "C:/Users/Lenovo/DataScraping-V1-Audio/tamil-nadu" --recursive

    # 3. Process a specific category directly in GCS:
    python scripts/retry_transcription.py --gcs --category ammanai

    # 4. Process all categories in GCS:
    python scripts/retry_transcription.py --gcs --all --workers 4
"""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
import tempfile
from typing import List, Tuple, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tnfolk.config import PipelineConfig, load_pipeline_config, load_taxonomy
from tnfolk.logging_utils import get_logger, setup_logging
from tnfolk.storage import Storage, format_lyrics_file
from tnfolk.transcription import _transcribe_api

_log = get_logger("retry_transcription")
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus"}


def process_local_file(
    audio_path: str,
    cfg: PipelineConfig,
    category_slug: str = "",
    force: bool = False,
) -> bool:
    """Transcribe a single local audio file and write its .txt transcript."""
    base_stem = os.path.splitext(audio_path)[0]
    txt_path = base_stem + ".txt"

    if os.path.exists(txt_path) and not force:
        try:
            if os.path.getsize(txt_path) > 20:
                _log.debug("Already exists and non-empty: %s", txt_path)
                return True
        except OSError:
            pass

    recording_id = os.path.basename(base_stem)
    if not category_slug:
        # Infer category from parent directory name if possible
        category_slug = os.path.basename(os.path.dirname(audio_path))

    _log.info("[Local] Transcribing: %s ...", os.path.basename(audio_path))
    try:
        raw_text = _transcribe_api(audio_path, cfg)
        content = format_lyrics_file(
            title=f"Tamil Folk Song ({recording_id})",
            channel="Unknown Artist",
            category_slug=category_slug,
            source_url=f"https://www.youtube.com/watch?v={recording_id}",
            raw_text=raw_text,
        )
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(content)
        _log.info("[Local] [OK] Saved transcript -> %s (%d chars)", txt_path, len(raw_text))
        return True
    except Exception as exc:
        _log.error("[Local] [FAIL] Failed transcribing %s: %s", os.path.basename(audio_path), exc)
        return False


def run_local_folder(
    folder_path: str,
    cfg: PipelineConfig,
    recursive: bool = True,
    category: str = "",
    force: bool = False,
    workers: int = 2,
) -> Tuple[int, int, int]:
    """Scan local directory for audio files missing .txt and transcribe them."""
    if not os.path.exists(folder_path):
        _log.error("Folder does not exist: %s", folder_path)
        return 0, 0, 0

    audio_files: List[str] = []
    if recursive:
        for root, _, files in os.walk(folder_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in AUDIO_EXTS:
                    audio_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(folder_path):
            ext = os.path.splitext(file)[1].lower()
            if ext in AUDIO_EXTS:
                audio_files.append(os.path.join(folder_path, file))

    total = len(audio_files)
    if total == 0:
        _log.info("No audio files found in %s", folder_path)
        return 0, 0, 0

    to_process: List[str] = []
    already_done = 0

    for a in audio_files:
        txt = os.path.splitext(a)[0] + ".txt"
        if os.path.exists(txt) and not force and os.path.getsize(txt) > 20:
            already_done += 1
        else:
            to_process.append(a)

    _log.info(
        "Found %d total audio files in %s (%d already have .txt, %d to transcribe)",
        total, folder_path, already_done, len(to_process)
    )

    if not to_process:
        _log.info("All audio files already have transcripts! Nothing to do.")
        return total, already_done, 0

    success_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(process_local_file, a, cfg, category, force): a
            for a in to_process
        }
        for fut in concurrent.futures.as_completed(futures):
            if fut.result():
                success_count += 1

    return total, already_done, success_count


def _gcs_list(gcloud: str, gcs_dir_uri: str) -> List[str]:
    """List objects under a GCS URI using gcloud storage ls."""
    uri = gcs_dir_uri.rstrip("/") + "/**"
    proc = subprocess.run([gcloud, "storage", "ls", uri],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        return []
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return lines


def process_gcs_category(
    category_slug: str,
    cfg: PipelineConfig,
    storage: Storage,
    force: bool = False,
    workers: int = 2,
) -> Tuple[int, int, int]:
    """Check a GCS category folder and retry missing transcriptions."""
    gcloud = storage.gcloud
    wav_base_uri = storage._gcs_uri(category_slug, review=False, filename="")
    _log.info("[GCS] Listing objects in %s ...", wav_base_uri)
    all_uris = _gcs_list(gcloud, wav_base_uri)

    wav_uris = [u for u in all_uris if u.lower().endswith(".wav")]
    txt_uris = {u for u in all_uris if u.lower().endswith(".txt")}

    total = len(wav_uris)
    if total == 0:
        _log.info("[GCS] No .wav files found under %s", wav_base_uri)
        return 0, 0, 0

    missing_wavs: List[str] = []
    already_done = 0

    for w_uri in wav_uris:
        expected_txt_uri = os.path.splitext(w_uri)[0] + ".txt"
        if expected_txt_uri in txt_uris and not force:
            already_done += 1
        else:
            missing_wavs.append(w_uri)

    _log.info(
        "[GCS] Category '%s': %d WAVs total (%d already have .txt, %d missing)",
        category_slug, total, already_done, len(missing_wavs)
    )

    if not missing_wavs:
        return total, already_done, 0

    def _process_one_gcs_wav(wav_uri: str) -> bool:
        recording_id = os.path.splitext(os.path.basename(wav_uri))[0]
        expected_txt_uri = os.path.splitext(wav_uri)[0] + ".txt"

        with tempfile.TemporaryDirectory() as tmp_dir:
            local_wav = os.path.join(tmp_dir, f"{recording_id}.wav")
            local_txt = os.path.join(tmp_dir, f"{recording_id}.txt")

            # 1. Download WAV from GCS
            dl_proc = subprocess.run(
                [gcloud, "storage", "cp", wav_uri, local_wav],
                capture_output=True, text=True
            )
            if dl_proc.returncode != 0:
                _log.error("[GCS] Failed to download %s: %s", wav_uri, dl_proc.stderr.strip()[:200])
                return False

            # 2. Transcribe
            try:
                _log.info("[GCS] Transcribing %s (%s) ...", recording_id, category_slug)
                raw_text = _transcribe_api(local_wav, cfg)
                content = format_lyrics_file(
                    title=f"Tamil Folk Song ({recording_id})",
                    channel="Unknown Artist",
                    category_slug=category_slug,
                    source_url=f"https://www.youtube.com/watch?v={recording_id}",
                    raw_text=raw_text,
                )
                with open(local_txt, "w", encoding="utf-8") as f:
                    f.write(content)

                # 3. Upload .txt to GCS
                up_proc = subprocess.run(
                    [gcloud, "storage", "cp", local_txt, expected_txt_uri],
                    capture_output=True, text=True
                )
                if up_proc.returncode != 0:
                    _log.error("[GCS] Failed to upload %s: %s", expected_txt_uri, up_proc.stderr.strip()[:200])
                    return False

                _log.info("[GCS] [OK] Successfully uploaded transcript -> %s", expected_txt_uri)
                return True
            except Exception as exc:
                _log.error("[GCS] [FAIL] Transcription failed for %s: %s", recording_id, exc)
                return False

    success_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_process_one_gcs_wav, w): w for w in missing_wavs}
        for fut in concurrent.futures.as_completed(futures):
            if fut.result():
                success_count += 1

    return total, already_done, success_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retry transcription for audio files that do not have .txt files."
    )
    parser.add_argument(
        "--folder", "-f",
        type=str,
        default="",
        help="Path to local folder containing audio files (e.g. 'tamil-nadu/ammanai')",
    )
    parser.add_argument(
        "--gcs",
        action="store_true",
        help="Scan and upload directly to GCS storage bucket",
    )
    parser.add_argument(
        "--category", "-c",
        type=str,
        default="",
        help="Category slug to process (e.g. 'ammanai')",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all categories from taxonomy",
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        default=True,
        help="Scan directories recursively (default: True)",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=2,
        help="Number of concurrent transcription workers (default: 2)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-transcribing even if .txt exists",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to pipeline_config.yaml",
    )

    args = parser.parse_args()

    cfg = load_pipeline_config(args.config)
    log_cfg = cfg.raw.get("logging", {})
    setup_logging(
        level=str(log_cfg.get("level", "INFO")),
        log_dir=cfg.abspath(str(log_cfg.get("dir", "logs"))) if log_cfg.get("dir") else None,
        log_file=str(log_cfg.get("file", "pipeline.log")),
    )

    _log.info("=" * 65)
    _log.info("  RETRY TRANSCRIPTION TOOL")
    _log.info("  Primary API URL: %s", cfg.raw["transcription"].get("api_url"))
    _log.info("  Fallback API URL: %s", cfg.raw["transcription"].get("fallback_api_url"))
    _log.info("=" * 65)

    if args.folder:
        total, done, success = run_local_folder(
            folder_path=args.folder,
            cfg=cfg,
            recursive=args.recursive,
            category=args.category,
            force=args.force,
            workers=args.workers,
        )
        print("\n" + "=" * 50)
        print(f"Summary for folder: {args.folder}")
        print(f"Total audio files:     {total}")
        print(f"Already transcribed:   {done}")
        print(f"Newly transcribed:     {success}")
        print(f"Failed / remaining:    {max(0, (total - done) - success)}")
        print("=" * 50)
        return

    if args.gcs or args.category or args.all:
        storage = Storage(cfg)
        taxonomy = load_taxonomy()

        categories: List[str] = []
        if args.category:
            categories = [args.category]
        elif args.all:
            categories = [c.folder_slug for c in taxonomy.categories]
        else:
            parser.error("Specify --folder <path>, --category <slug>, or --all")

        total_all, done_all, success_all = 0, 0, 0
        for cat in categories:
            t, d, s = process_gcs_category(
                category_slug=cat,
                cfg=cfg,
                storage=storage,
                force=args.force,
                workers=args.workers,
            )
            total_all += t
            done_all += d
            success_all += s

        print("\n" + "=" * 50)
        print("GCS Summary across categories:")
        print(f"Total WAV files:       {total_all}")
        print(f"Already had .txt:      {done_all}")
        print(f"Newly transcribed:     {success_all}")
        print(f"Failed / remaining:    {max(0, (total_all - done_all) - success_all)}")
        print("=" * 50)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
