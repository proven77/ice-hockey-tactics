import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path, PurePosixPath
from random import Random
from zipfile import ZipFile


NAMES = ["centerIce", "faceoff", "goal", "goaltender", "player", "puck", "referee"]


def prepare(archive_path, output):
    with ZipFile(archive_path) as archive:
        names = archive.namelist()
        frames = {PurePosixPath(name).stem: name for name in names
                  if name.startswith("SHL/frames/") and name.endswith(".jpg")}
        labels = {PurePosixPath(name).stem: name for name in names
                  if name.startswith("SHL/annotations/") and name.endswith(".txt")}
        if not frames or frames.keys() != labels.keys():
            raise ValueError("Expected one SHL annotation per image in the HockeyAI ZIP")

        identical = defaultdict(list)
        for image_id, member in frames.items():
            signature = hashlib.sha256(archive.read(member)).digest()
            identical[signature].append(image_id)
        # Repeated images in this archive can carry different annotations.
        ids = sorted(group[0] for group in identical.values() if len(group) == 1)
        Random(42).shuffle(ids)
        n_val = len(ids) // 10
        if n_val == 0:
            raise ValueError("At least ten unique images are needed for train, validation, and test")
        split = {"test": ids[:n_val], "val": ids[n_val:2 * n_val], "train": ids[2 * n_val:]}
        output.mkdir(parents=True, exist_ok=False)
        for subset, members in split.items():
            images = output / "images" / subset
            annotations = output / "labels" / subset
            images.mkdir(parents=True)
            annotations.mkdir(parents=True)
            for image_id in members:
                (images / f"{image_id}.jpg").write_bytes(archive.read(frames[image_id]))
                (annotations / f"{image_id}.txt").write_bytes(archive.read(labels[image_id]))

    config = {
        "path": output.resolve().as_posix(), "train": "images/train", "val": "images/val", "test": "images/test",
        "names": NAMES,
    }
    (output / "data.yaml").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print({subset: len(members) for subset, members in split.items()}, f"excluded repeated images: {len(frames) - len(ids)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert the official HockeyAI archive to an Ultralytics dataset.")
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path, help="New dataset directory")
    args = parser.parse_args()
    prepare(args.archive, args.output)
