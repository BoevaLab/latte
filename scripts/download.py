"""Script for downloading the datasets.

Currently supported:
    - dsprites
"""
import pathlib
import requests

RAW_DATA_DIR = pathlib.Path("data/raw")

DSPRITES_URL = "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
FILE_LOCATION = RAW_DATA_DIR / "dsprites.npz"


def main() -> None:
    req = requests.get(DSPRITES_URL)

    RAW_DATA_DIR.mkdir(exist_ok=True, parents=True)

    with open(FILE_LOCATION, "wb") as f:
        f.write(req.content)


if __name__ == "__main__":
    main()
