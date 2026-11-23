"""Single source of truth for the published RNAglib data releases.

Bump these values (and nothing else) when a new database release is published on Zenodo.
Every download helper, dataset and task defaults to ``DATA_VERSION`` / ``ZENODO_RECORD``.

This module must stay free of rnaglib imports so it can be imported from anywhere
without creating import cycles.
"""

# --- RNA structure database (graphs built by ``rnaglib_prepare_data``) ---
# Version tag of the current database release. Archives are named
# ``rnaglib-{all,nr,debug}-<DATA_VERSION>[-annotated].tar.gz``.
DATA_VERSION = "3.1.0"

# Zenodo record holding the current release. Each new Zenodo version gets a new record id;
# new versions inherit the previous files, so this record also serves older data versions.
ZENODO_RECORD = "24476342"
ZENODO_BASE_URL = "https://zenodo.org"

# --- Precomputed task datasets ---
# Database version the benchmark tasks are built from. Kept separate from DATA_VERSION so a new
# database release does not silently change benchmark data; bump it deliberately.
TASKS_DATA_VERSION = "3.0.0"
TASKS_ZENODO_RECORD = "189027"
TASKS_ZENODO_BASE_URL = "https://sandbox.zenodo.org"


def zenodo_file_url(filename: str, record: str = ZENODO_RECORD, base_url: str = ZENODO_BASE_URL) -> str:
    """Direct download URL for a file in a Zenodo record."""
    return f"{base_url}/records/{record}/files/{filename}?download=1"


def tasks_zenodo_file_url(filename: str) -> str:
    """Download URL for a precomputed task archive (no query string: the downloader names the file
    after the last URL segment)."""
    return f"{TASKS_ZENODO_BASE_URL}/records/{TASKS_ZENODO_RECORD}/files/{filename}"
