# Dataset files

The benchmark downloads the four official LHC Olympics 2020 R&D release files (Zenodo record v5, DOI 10.5281/zenodo.6466204) into `data/raw/` on first use. The files are intentionally excluded from version control. `data/manifest.yaml` records Zenodo-provided MD5 checksums, locally verified SHA-256 digests, and byte sizes. `python scripts/download_data.py` verifies the files; `python scripts/reproduce.py` runs the full pipeline.
