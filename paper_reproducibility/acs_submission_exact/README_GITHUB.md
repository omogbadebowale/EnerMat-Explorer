# Exact ACS submission snapshot

This folder preserves the **exact text files** used in the final ACS Applied Energy Materials supplementary ZIP.

The large CSV datasets are not duplicated here. Their canonical repository copies are in `../data/` and have been checked against the final ZIP; all S1–S7b CSV files are byte-for-byte identical to the submitted package.

- `README.txt` and `SHA256SUMS.txt` are copied from the final flat supplementary package.
- The three Python files in this folder are the exact submission versions.
- The repository-native scripts one directory above are retained because they write outputs into the repository's `data/` and `figures/` layout.

For the original flat-package checksum list, see `SHA256SUMS.txt`. When checking the CSVs in this repository, use the corresponding files under `../data/`.
