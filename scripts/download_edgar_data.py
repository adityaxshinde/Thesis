"""
Downloads SEC EDGAR Financial Statement Data Sets (quarterly zips) for
2012 Q1 through 2026 Q1, saves them to data/raw/, unzips each into its
own subfolder, then prints a summary table with success/size/row-count
and flags quarters with abnormally low sub.txt row counts.

Usage: python scripts/download_edgar_data.py
"""

import os
import time
import zipfile
import statistics
from datetime import datetime

import requests

BASE_URL = "https://www.sec.gov/files/dera/data/financial-statement-data-sets"
USER_AGENT = "Thesis Research thesis@research.com"
HEADERS = {"User-Agent": USER_AGENT}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, "..", "data", "raw")
RAW_DIR = os.path.abspath(RAW_DIR)

START_YEAR, START_Q = 2012, 1
END_YEAR, END_Q = 2026, 1

MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 5
REQUEST_DELAY_SEC = 0.5  # be polite to SEC servers between requests


def generate_quarters(start_year, start_q, end_year, end_q):
    quarters = []
    y, q = start_year, start_q
    while (y, q) <= (end_year, end_q):
        quarters.append(f"{y}q{q}")
        q += 1
        if q > 4:
            q = 1
            y += 1
    return quarters


def download_quarter(quarter, dest_zip):
    """Downloads one quarterly zip with retries. Returns (success, size_mb, error_msg)."""
    url = f"{BASE_URL}/{quarter}.zip"

    if os.path.exists(dest_zip) and os.path.getsize(dest_zip) > 0:
        size_mb = os.path.getsize(dest_zip) / (1024 * 1024)
        return True, size_mb, None

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, stream=True, timeout=60)
            if resp.status_code == 200:
                tmp_path = dest_zip + ".partial"
                with open(tmp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                os.replace(tmp_path, dest_zip)
                size_mb = os.path.getsize(dest_zip) / (1024 * 1024)
                return True, size_mb, None
            else:
                last_error = f"HTTP {resp.status_code}"
                if resp.status_code == 404:
                    break  # no point retrying a missing file
        except requests.RequestException as e:
            last_error = str(e)

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF_SEC * attempt)

    return False, 0.0, last_error


def unzip_quarter(quarter, dest_zip, extract_dir):
    try:
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(dest_zip, "r") as zf:
            zf.extractall(extract_dir)
        return True, None
    except zipfile.BadZipFile as e:
        return False, str(e)


def count_sub_rows(extract_dir):
    """Counts data rows (excluding header) in that quarter's sub.txt."""
    sub_path = os.path.join(extract_dir, "sub.txt")
    if not os.path.exists(sub_path):
        return None
    try:
        with open(sub_path, "r", encoding="utf-8", errors="replace") as f:
            row_count = sum(1 for _ in f) - 1  # minus header
        return max(row_count, 0)
    except OSError:
        return None


def flag_abnormal_rows(results):
    """Flags quarters whose sub.txt row count is far below the median of
    their nearest neighbours (within a window of 2 quarters on each side
    that have valid counts)."""
    counts = [r["row_count"] for r in results]
    n = len(results)
    for i in range(n):
        if counts[i] is None:
            results[i]["flag"] = "NO DATA"
            continue
        neighbours = []
        for j in range(max(0, i - 2), min(n, i + 3)):
            if j != i and counts[j] is not None:
                neighbours.append(counts[j])
        if len(neighbours) < 2:
            results[i]["flag"] = ""
            continue
        neighbour_median = statistics.median(neighbours)
        if neighbour_median > 0 and counts[i] < 0.5 * neighbour_median:
            pct = 100 * counts[i] / neighbour_median
            results[i]["flag"] = f"LOW ({pct:.0f}% of neighbour median)"
        else:
            results[i]["flag"] = ""


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    quarters = generate_quarters(START_YEAR, START_Q, END_YEAR, END_Q)
    print(f"Planning to fetch {len(quarters)} quarters: {quarters[0]} .. {quarters[-1]}")
    print(f"Saving to: {RAW_DIR}\n")

    results = []
    for idx, quarter in enumerate(quarters, 1):
        dest_zip = os.path.join(RAW_DIR, f"{quarter}.zip")
        extract_dir = os.path.join(RAW_DIR, quarter)

        print(f"[{idx}/{len(quarters)}] {quarter} ... ", end="", flush=True)
        success, size_mb, error = download_quarter(quarter, dest_zip)

        row_count = None
        if success:
            unzip_ok, unzip_error = unzip_quarter(quarter, dest_zip, extract_dir)
            if unzip_ok:
                row_count = count_sub_rows(extract_dir)
                print(f"OK ({size_mb:.1f} MB, sub.txt rows={row_count})")
            else:
                success = False
                error = f"unzip failed: {unzip_error}"
                print(f"FAILED ({error})")
        else:
            print(f"FAILED ({error})")

        results.append(
            {
                "quarter": quarter,
                "success": success,
                "size_mb": size_mb if success else 0.0,
                "row_count": row_count,
                "error": error,
            }
        )

        time.sleep(REQUEST_DELAY_SEC)

    flag_abnormal_rows(results)

    # ---- summary table ----
    print("\n" + "=" * 90)
    print(f"SUMMARY  ({datetime.now().isoformat(timespec='seconds')})")
    print("=" * 90)
    header = f"{'Quarter':<10}{'Success':<10}{'Size (MB)':<12}{'sub.txt rows':<15}{'Flag':<30}"
    print(header)
    print("-" * 90)

    for r in results:
        size_str = f"{r['size_mb']:.1f}" if r["success"] else "-"
        rows_str = str(r["row_count"]) if r["row_count"] is not None else "-"
        flag_str = r["flag"] if r["success"] else (r["error"] or "FAILED")
        print(f"{r['quarter']:<10}{str(r['success']):<10}{size_str:<12}{rows_str:<15}{flag_str:<30}")

    n_success = sum(1 for r in results if r["success"])
    n_failed = len(results) - n_success
    flagged = [r for r in results if r["success"] and r["flag"] and r["flag"] != ""]
    print("-" * 90)
    print(f"Total: {len(results)}  |  Succeeded: {n_success}  |  Failed: {n_failed}")
    if flagged:
        print(f"Flagged quarters (abnormally low sub.txt row count): {', '.join(r['quarter'] for r in flagged)}")
    else:
        print("No quarters flagged for abnormal sub.txt row counts.")


if __name__ == "__main__":
    main()
