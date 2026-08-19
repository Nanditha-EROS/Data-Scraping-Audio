"""Check and display the live count of songs uploaded to GCS."""
import os
import shutil
import subprocess
import yaml
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "pipeline_config.yaml")

def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    storage_cfg = cfg.get("storage", {})
    root = storage_cfg.get("root", "kl-workspace/Data_scraping_version1/Song").strip("/")
    state = cfg.get("state_name", "tamil-nadu")
    gcloud = storage_cfg.get("gcloud_path") or shutil.which("gcloud") or shutil.which("gcloud.cmd") or "gcloud"

    base_prefix = f"{root}/{state}/"
    uri = f"gs://{base_prefix}**"

    print("=" * 65)
    print(f"  GCS LIVE COUNT: gs://{base_prefix}")
    print("=" * 65)

    proc = subprocess.run([gcloud, "storage", "ls", uri], capture_output=True, text=True)
    if proc.returncode != 0:
        print("Error connecting to GCS:", proc.stderr.strip()[:300])
        return

    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    wavs = [l for l in lines if l.lower().endswith(".wav")]
    txts = [l for l in lines if l.lower().endswith(".txt")]

    prefix_strip = f"gs://{base_prefix}"
    by_cat_wav = defaultdict(int)
    by_cat_txt = defaultdict(int)
    review_wav = defaultdict(int)

    for w in wavs:
        rel = w.replace(prefix_strip, "").split("/")
        if rel[0] == "review":
            cat = rel[1] if len(rel) > 1 else "unknown"
            review_wav[cat] += 1
        else:
            cat = rel[0]
            by_cat_wav[cat] += 1

    for t in txts:
        rel = t.replace(prefix_strip, "").split("/")
        if rel[0] != "review":
            cat = rel[0]
            by_cat_txt[cat] += 1

    all_cats = sorted(set(list(by_cat_wav.keys()) + list(review_wav.keys())))

    print(f"\nTotal Audio (.wav):     {len(wavs)}")
    print(f"Total Transcripts (.txt): {len(txts)}")
    print(f"  - Accepted .wav:      {sum(by_cat_wav.values())}")
    print(f"  - Review .wav:        {sum(review_wav.values())}\n")

    print("-" * 65)
    print(f"{'Category':<30} | {'Accepted WAV':<12} | {'TXT':<6} | {'Review WAV':<10}")
    print("-" * 65)
    for c in all_cats:
        w_count = by_cat_wav[c]
        t_count = by_cat_txt[c]
        r_count = review_wav[c]
        print(f"{c:<30} | {w_count:<12} | {t_count:<6} | {r_count:<10}")
    print("-" * 65)


if __name__ == "__main__":
    main()
