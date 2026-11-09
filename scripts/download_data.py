"""Download the UCI Bank Marketing dataset into data/.

Source: UCI ML Repository, dataset 222 (Bank Marketing).
The download is a zip containing a second zip; the file we want is
bank-additional-full.csv inside the inner zip.
"""
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
OUT = Path("data/bank-additional-full.csv")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL} ...")
    with urllib.request.urlopen(URL) as response:
        outer = zipfile.ZipFile(io.BytesIO(response.read()))
    inner_name = [n for n in outer.namelist() if n.endswith("bank-additional.zip")][0]
    inner = zipfile.ZipFile(io.BytesIO(outer.read(inner_name)))
    csv_name = [n for n in inner.namelist() if n.endswith("bank-additional-full.csv")][0]
    OUT.write_bytes(inner.read(csv_name))
    print(f"Wrote {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
