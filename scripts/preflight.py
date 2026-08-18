"""Pre-flight check for the pipeline."""
import os
import sys
import shutil
import subprocess
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print("=" * 60)
print("  PRE-FLIGHT CHECK")
print("=" * 60)

issues = []

# 1. Config loads OK
try:
    with open("config/pipeline_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    t = cfg["transcription"]
    print(f"[OK] Config loads successfully")
    print(f"     Transcription backend: {t['backend']}")
    print(f"     API URL: {t.get('api_url', 'N/A')}")
    print(f"     Language: {t.get('language', 'N/A')}")
    print(f"     Storage backend: {cfg['storage']['backend']}")
    print(f"     GCS root: {cfg['storage']['root']}")
except Exception as e:
    issues.append(f"Config error: {e}")
    print(f"[FAIL] Config: {e}")

# 2. Check ffmpeg
if shutil.which("ffmpeg"):
    print("[OK] ffmpeg found on PATH")
else:
    issues.append("ffmpeg not found on PATH")
    print("[FAIL] ffmpeg not found on PATH")

# 3. Check ffprobe
if shutil.which("ffprobe"):
    print("[OK] ffprobe found on PATH")
else:
    issues.append("ffprobe not found on PATH")
    print("[FAIL] ffprobe not found on PATH")

# 4. Check fpcalc (Chromaprint)
if shutil.which("fpcalc"):
    print("[OK] fpcalc found on PATH")
else:
    issues.append("fpcalc not found on PATH (needed for duplicate detection)")
    print("[WARN] fpcalc not found on PATH (duplicate detection may fail)")

# 5. Check gcloud (for GCS upload)
gcloud = shutil.which("gcloud") or shutil.which("gcloud.cmd")
if gcloud:
    print(f"[OK] gcloud found: {gcloud}")
else:
    if cfg["storage"]["backend"] == "gcs":
        issues.append("gcloud not found but storage backend is 'gcs'")
        print("[FAIL] gcloud not found on PATH (needed for GCS upload)")
    else:
        print("[OK] gcloud not needed (storage backend is local)")

# 6. State dirs exist
os.makedirs("state", exist_ok=True)
print("[OK] state/ directory ready")

# 7. Test API reachability (URL read from config, not hardcoded)
try:
    import requests
    _api_base = t.get("api_url", "").rsplit("/api/", 1)[0] + "/"
    r = requests.get(_api_base, timeout=10)
    print(f"[OK] Transcription API reachable at {_api_base} (status {r.status_code})")
except requests.exceptions.ConnectionError:
    _api_base = t.get("api_url", "N/A")
    issues.append(f"Cannot reach transcription API at {_api_base}")
    print(f"[FAIL] Cannot reach transcription API at {_api_base}")
except Exception as e:
    print(f"[WARN] API check inconclusive: {e}")

# 8. Check yt-dlp
if shutil.which("yt-dlp"):
    print("[OK] yt-dlp found on PATH")
else:
    try:
        import yt_dlp
        print("[OK] yt-dlp available as Python module")
    except ImportError:
        issues.append("yt-dlp not found")
        print("[FAIL] yt-dlp not found")

# 9. Pipeline module loads
try:
    from tnfolk.orchestrator import build_pipeline
    print("[OK] tnfolk.orchestrator imports successfully")
except Exception as e:
    issues.append(f"Import error: {e}")
    print(f"[FAIL] Import error: {e}")

print("=" * 60)
if issues:
    print(f"  {len(issues)} ISSUE(S) FOUND:")
    for i, issue in enumerate(issues, 1):
        print(f"    {i}. {issue}")
    print("  FIX THESE BEFORE RUNNING THE PIPELINE")
else:
    print("  ALL CHECKS PASSED - READY TO RUN!")
print("=" * 60)
