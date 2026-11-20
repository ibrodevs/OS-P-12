"""Download the public educational stroke dataset and verify its Git blob hash."""
from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import urlopen

URL = 'https://raw.githubusercontent.com/chandanverma07/DataSets/master/healthcare-dataset-stroke-data.csv'
EXPECTED_GIT_BLOB_SHA1 = '763aacc5a9a5e62e735fe4aab8acead059a18cd1'
DESTINATION = Path(__file__).parent / 'data' / 'healthcare-dataset-stroke-data.csv'


def download() -> Path:
    with urlopen(URL, timeout=30) as response:
        data = response.read()
    git_hash = hashlib.sha1(b'blob ' + str(len(data)).encode('ascii') + b'\0' + data).hexdigest()
    if git_hash != EXPECTED_GIT_BLOB_SHA1:
        raise ValueError('Source CSV was changed: its Git blob SHA-1 does not match the pinned value')
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_bytes(data)
    print(f'Downloaded verified dataset: {DESTINATION} ({len(data)} bytes)')
    return DESTINATION


if __name__ == '__main__':
    download()
