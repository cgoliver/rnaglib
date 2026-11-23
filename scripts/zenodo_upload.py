#!/usr/bin/env python3
"""Upload a database release to Zenodo as a new version of the current record.

Starts from ``ZENODO_RECORD`` in ``rnaglib/config/data_release.py``, opens a new-version draft
(which inherits the previous version's files), uploads the given archives, sets the version
metadata and, with ``--publish``, publishes it. Prints the new record id to put into
``data_release.py``.

Token: ``ZENODO_TOKEN`` env var, or ``--token-file``.

    python scripts/zenodo_upload.py --version 3.1.0 release-3.1.0/*.tar.gz            # draft only
    python scripts/zenodo_upload.py --version 3.1.0 --draft <id> --publish            # publish draft
"""

import argparse
import datetime
import os
import sys
from pathlib import Path

import requests

from rnaglib.config.data_release import ZENODO_BASE_URL, ZENODO_RECORD

API = f"{ZENODO_BASE_URL}/api/deposit/depositions"


def check(r: requests.Response) -> dict:
    if not r.ok:
        sys.exit(f"Zenodo error {r.status_code} on {r.request.method} {r.url}:\n{r.text}")
    return r.json() if r.content else {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path, help="archives to upload")
    ap.add_argument("--version", required=True, help="data version, e.g. 3.1.0")
    ap.add_argument("--record", default=ZENODO_RECORD, help="record to create a new version of")
    ap.add_argument("--draft", help="reuse an existing draft id instead of creating a new version")
    ap.add_argument("--token-file", type=Path)
    ap.add_argument("--replace", action="store_true", help="replace files of the same name inherited from the previous version")
    ap.add_argument("--publish", action="store_true", help="publish the draft (irreversible: mints a DOI)")
    args = ap.parse_args()

    token = args.token_file.read_text().strip() if args.token_file else os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("Set ZENODO_TOKEN or pass --token-file")
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"

    if args.draft:
        draft = check(s.get(f"{API}/{args.draft}"))
    else:
        resp = check(s.post(f"{API}/{args.record}/actions/newversion"))
        draft = check(s.get(resp["links"]["latest_draft"]))
    did = draft["id"]
    print(f"Draft {did}: {draft['links']['html']}")

    existing = {f["filename"]: f for f in draft.get("files", [])}
    bucket = draft["links"]["bucket"]
    for path in args.files:
        if path.name in existing:
            if not args.replace:
                sys.exit(f"{path.name} already in draft {did}; pass --replace to overwrite it")
            print(f"Removing inherited {path.name} from draft...")
            check(s.delete(f"{API}/{did}/files/{existing[path.name]['id']}"))
        size = path.stat().st_size
        print(f"Uploading {path.name} ({size / 1e6:.1f} MB)...", flush=True)
        with open(path, "rb") as fh:
            check(s.put(f"{bucket}/{path.name}", data=fh, timeout=None))

    meta = draft["metadata"]
    meta["version"] = args.version
    meta["publication_date"] = datetime.date.today().isoformat()
    meta.pop("doi", None)
    check(s.put(f"{API}/{did}", json={"metadata": meta}))

    files = check(s.get(f"{API}/{did}/files"))
    print("Files in draft:")
    for f in files:
        print(f"  {f['filename']:45s} {f['filesize'] / 1e6:10.1f} MB  md5:{f['checksum']}")

    if args.publish:
        pub = check(s.post(f"{API}/{did}/actions/publish"))
        print(f"Published record {pub['id']} (DOI {pub.get('doi')}): {pub['links']['html']}")
        print(f'Now set ZENODO_RECORD = "{pub["id"]}" and DATA_VERSION = "{args.version}" in data_release.py')
    else:
        print(f"Not published. Review the draft, then rerun with --draft {did} --publish")


if __name__ == "__main__":
    main()
