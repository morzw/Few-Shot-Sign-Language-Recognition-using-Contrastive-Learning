import argparse
import shutil
import zipfile
from pathlib import Path

import gdown


MESL_FILE_ID = "1DavK1m4cVOOkSqXcRm1ZJw6JEWeGSPZf"
FESL_FILE_ID = "1zxg3Yafon_v94gGI_zPXYKbK6l_6cjZH"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download MESL/FESL datasets from Google Drive and place them in project-local paths."
    )
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Standalone project root (defaults to current community_project).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing dataset folders.",
    )
    return parser.parse_args()


def _clear_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _extract_zip_normalized(zip_path: Path, target_dir: Path):
    temp_dir = target_dir.parent / f".tmp_extract_{target_dir.name}"
    _clear_dir(temp_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(temp_dir)

    items = list(temp_dir.iterdir())
    # If archive has a single root directory, flatten it.
    if len(items) == 1 and items[0].is_dir():
        source_root = items[0]
    else:
        source_root = temp_dir

    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.move(str(source_root), str(target_dir))

    if temp_dir.exists():
        shutil.rmtree(temp_dir)


def _download_file(file_id: str, output_zip: Path):
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url=url, output=str(output_zip), quiet=False, fuzzy=True)


def main():
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    datasets_root = project_root / "datasets"
    raw_root = datasets_root / "raw"
    mesl_zip = raw_root / "mesl.zip"
    fesl_zip = raw_root / "fesl.zip"
    mesl_dir = datasets_root / "mesl"
    fesl_dir = datasets_root / "fesl"

    if (mesl_dir.exists() or fesl_dir.exists()) and not args.force:
        raise RuntimeError(
            "Dataset folders already exist. Use --force to overwrite: "
            f"{mesl_dir}, {fesl_dir}"
        )

    print("Downloading MESL...")
    _download_file(MESL_FILE_ID, mesl_zip)
    print("Downloading FESL...")
    _download_file(FESL_FILE_ID, fesl_zip)

    print("Extracting MESL...")
    _extract_zip_normalized(mesl_zip, mesl_dir)
    print("Extracting FESL...")
    _extract_zip_normalized(fesl_zip, fesl_dir)

    print(f"MESL ready: {mesl_dir}")
    print(f"FESL ready: {fesl_dir}")
    print("Config paths expected:")
    print("  - configs/mesl_train.yaml -> dataset_root: datasets/mesl")
    print("  - configs/fesl_eval.yaml -> fesl_root: datasets/fesl")


if __name__ == "__main__":
    main()

