import os
import urllib.request
import hashlib
import json
import datetime
import yaml

ZENODO_RECORD = '6466204'
ZENODO_API_URL = f'https://zenodo.org/api/records/{ZENODO_RECORD}'

TARGET_FILES = [
    'events_anomalydetection_v2.h5',
    'events_anomalydetection_v2.features.h5',
    'events_anomalydetection_Z_XY_qqq.h5',
    'events_anomalydetection_Z_XY_qqq.features.h5'
]

DATA_DIR = os.path.join('data', 'raw')
MANIFEST_PATH = os.path.join('data', 'manifest.yaml')

def download_and_verify(url, filename, expected_checksum, expected_size):
    filepath = os.path.join(DATA_DIR, filename)
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(filepath):
        print(f"File {filename} already exists. Verifying checksum...")
        if verify_checksum(filepath, expected_checksum):
            print(f"Checksum verified for {filename}.")
            return True
        else:
            print(f"Checksum mismatch for {filename}. Redownloading...")

    print(f"Downloading {filename}...")
    # Add simple progress reporter
    last_bucket = [-10]
    def reporthook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        bucket = min(100, (percent // 10) * 10)
        if bucket > last_bucket[0]:
            last_bucket[0] = bucket
            print(f"\rDownloading {filename}: {bucket}%", end="", flush=True)

    # Download beside the destination and only replace it after size/checksum
    # validation. Interrupted transfers therefore cannot corrupt a usable file.
    partial_path = filepath + '.part'
    urllib.request.urlretrieve(url, partial_path, reporthook=reporthook)
    print()
    if os.path.getsize(partial_path) == expected_size and verify_checksum(partial_path, expected_checksum):
        os.replace(partial_path, filepath)
        print(f"Checksum verified for {filename}.")
        return True
    else:
        print(f"ERROR: Checksum mismatch after download for {filename}.")
        return False

def verify_checksum(filepath, expected_checksum):
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    file_hash = f"md5:{hash_md5.hexdigest()}"
    return file_hash == expected_checksum

def sha256_file(filepath):
    digest = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    req = urllib.request.Request(ZENODO_API_URL)
    with urllib.request.urlopen(req) as response:
        record_data = json.loads(response.read().decode())
    
    manifest = {
        'dataset': 'LHC Olympics 2020 R&D dataset',
        'zenodo_doi': '10.5281/zenodo.6466204',
        'record_version': record_data.get('metadata', {}).get('version', 'v5'),
        'download_date': datetime.datetime.now().isoformat(),
        'files': [],
        'feature_schema_version': '1.1.0'
    }
    
    for f in record_data['files']:
        filename = f['key']
        if filename in TARGET_FILES:
            success = download_and_verify(
                url=f['links']['self'], 
                filename=filename, 
                expected_checksum=f['checksum'], 
                expected_size=f['size']
            )
            if success:
                manifest['files'].append({
                    'filename': filename,
                    'byte_size': f['size'],
                    'checksum': f['checksum'],
                    'sha256': sha256_file(os.path.join(DATA_DIR, filename))
                })
    
    with open(MANIFEST_PATH, 'w') as mf:
        yaml.dump(manifest, mf, sort_keys=False)
    
    print(f"Manifest written to {MANIFEST_PATH}")

if __name__ == '__main__':
    main()
