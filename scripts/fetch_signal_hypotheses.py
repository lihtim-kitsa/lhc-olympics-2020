"""Fetch selected supplementary LHCO signal samples from their Zenodo archive."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import time
import tarfile
import urllib.request

import yaml


def digest(path, algorithm="md5"):
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download_ranges(url, temporary, *, workers=2, chunk_size=128 * 1024 * 1024):
    """Download a large archive in resumable byte ranges, then assemble it."""
    headers = {"User-Agent": "lhco-anomaly-benchmark/1.0"}
    head = urllib.request.Request(url, method="HEAD", headers=headers)
    while True:
        try:
            with urllib.request.urlopen(head, timeout=180) as response:
                total = int(response.headers["Content-Length"])
            break
        except (OSError, TimeoutError) as error:
            print(f"archive size lookup failed: {error}; retrying", flush=True)
            time.sleep(5)
    parts_dir = temporary + ".parts"
    os.makedirs(parts_dir, exist_ok=True)
    ranges = [(i, start, min(total - 1, start + chunk_size - 1))
              for i, start in enumerate(range(0, total, chunk_size))]

    def fetch(part):
        index, start, end = part
        path = os.path.join(parts_dir, f"{index:04d}.part")
        expected = end - start + 1
        while True:
            current = os.path.getsize(path) if os.path.exists(path) else 0
            if current == expected:
                return path
            if current > expected:
                os.remove(path)
                current = 0
            request_headers = dict(headers)
            request_headers["Range"] = f"bytes={start + current}-{end}"
            request = urllib.request.Request(url, headers=request_headers)
            try:
                with urllib.request.urlopen(request, timeout=45) as response:
                    if response.status != 206:
                        raise OSError(f"server ignored byte range for chunk {index}")
                    content_range = response.headers.get("Content-Range")
                    expected_range = f"bytes {start + current}-{end}/{total}"
                    if content_range != expected_range:
                        raise OSError(f"unexpected byte range for chunk {index}: {content_range}")
                    with open(path, "ab" if current else "wb") as out:
                        shutil.copyfileobj(response, out, length=8 * 1024 * 1024)
                if os.path.getsize(path) != expected:
                    raise OSError(f"incomplete chunk {index}: {os.path.getsize(path)} of {expected} bytes")
                print(f"chunk {index} complete ({expected} bytes)", flush=True)
                time.sleep(.15)  # respect Zenodo's per-client request pacing
            except (OSError, TimeoutError) as error:
                print(f"chunk {index} interrupted at {current} bytes: {error}; resuming", flush=True)
                time.sleep(3)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, part) for part in ranges]
        for future in as_completed(futures):
            future.result()

    with open(temporary, "wb") as output:
        for index, _, _ in ranges:
            with open(os.path.join(parts_dir, f"{index:04d}.part"), "rb") as chunk:
                shutil.copyfileobj(chunk, output, length=8 * 1024 * 1024)
    if os.path.getsize(temporary) != total:
        raise ValueError(f"assembled archive size mismatch: expected {total}, got {os.path.getsize(temporary)}")
    for name in os.listdir(parts_dir):
        os.remove(os.path.join(parts_dir, name))
    os.rmdir(parts_dir)


def fetch_one(item, root):
    archive_path = os.path.join(root, os.path.basename(item["archive"]))
    os.makedirs(root, exist_ok=True)
    if not os.path.exists(archive_path):
        temporary = archive_path + ".part"
        download_ranges(item["url"], temporary)
        os.replace(temporary, archive_path)
    actual = digest(archive_path)
    if actual.lower() != item["archive_md5"].lower():
        raise ValueError(f"MD5 mismatch for {archive_path}: expected {item['archive_md5']}, got {actual}")

    target_dir = os.path.join(root, item["id"])
    output = os.path.join(target_dir, "events.h5")
    if os.path.exists(output):
        return output
    os.makedirs(target_dir, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        matches = [m for m in archive.getmembers() if m.isfile() and m.name.endswith("/events.h5")]
        if len(matches) != 1:
            raise ValueError(f"Expected one events.h5 in {archive_path}; found {len(matches)}")
        source = archive.extractfile(matches[0])
        if source is None:
            raise ValueError(f"Could not read {matches[0].name} from {archive_path}")
        temp_output = output + ".part"
        try:
            with source, open(temp_output, "wb") as out:
                while True:
                    block = source.read(8 * 1024 * 1024)
                    if not block:
                        break
                    out.write(block)
            os.replace(temp_output, output)
        finally:
            if os.path.exists(temp_output):
                os.remove(temp_output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hypothesis", nargs="+", help="IDs from configs/signal_hypotheses.yaml")
    parser.add_argument("--config", default="configs/signal_hypotheses.yaml")
    parser.add_argument("--directory", default="data/external_signals")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    items = {item["id"]: item for item in config["hypotheses"] if item.get("archive")}
    fetched = []
    for hypothesis in args.hypothesis:
        if hypothesis not in items:
            parser.error(f"unknown external hypothesis {hypothesis!r}; choose from {sorted(items)}")
        item = items[hypothesis]
        path = fetch_one(item, args.directory)
        fetched.append({"id": hypothesis, "events_path": os.path.relpath(path, args.directory),
                        "source_record": item["source_record"], "archive_url": item["url"],
                        "archive_md5": item["archive_md5"], "events_sha256": digest(path, "sha256"),
                        "events_bytes": os.path.getsize(path)})
        print(f"{hypothesis}: {path}")
    manifest_path = os.path.join(args.directory, "manifest.json")
    existing = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            existing = {row["id"]: row for row in json.load(f).get("hypotheses", [])}
    existing.update({row["id"]: row for row in fetched})
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"recorded_utc": datetime.now(timezone.utc).isoformat(),
                   "hypotheses": list(existing.values())}, f, indent=2)


if __name__ == "__main__":
    main()
