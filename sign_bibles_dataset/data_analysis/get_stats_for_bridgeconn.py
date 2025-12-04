import logging
import argparse
from pathlib import Path
import tempfile

import webdataset as wds
from torch.utils.data import DataLoader
import numpy as np
import json
import cv2

from pose_format import Pose
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process WebDataset samples.")
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=None,
        help="Directory to use for temporary downloads (defaults to system temp)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "validation", "test"],
        help="Dataset split to process (default: all splits)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    download_dir = args.download_dir or Path(tempfile.gettempdir())
    download_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Using download directory: %s", download_dir.resolve())

    lang = "ins"
    project = "indian_sign_language_version_islv"
    splits = [args.split] if args.split else ["train", "validation", "test"]
    buffer_size = 1024

    for split in tqdm(splits, desc="Processing splits"):
        logging.info("Processing split: %s", split)

        dataset_url = (
            f"https://huggingface.co/datasets/bridgeconn/sign-bibles-isl/resolve/main/"
            f"{lang}/{project}/shard_{{00001..00002}}-{split}.tar"
        )

        dataset = (
            wds.WebDataset(dataset_url, shardshuffle=False)
            .shuffle(buffer_size)
            .decode()
        )

        for sample in dataset:
            try:
                json_data = sample["json"]
                logging.info("Total frames: %d", json_data["total_frames"])
                logging.info("Bible ref: %s", json_data["bible-ref"])
                logging.info("BibleNLP ref: %s", json_data["biblenlp-vref"])
                logging.info("Signer: %s", json_data["signer"])

                text_json = sample["transcripts.json"]
                logging.info("Text: %s", text_json[0]["text"])
                logging.info("Language: %s", text_json[0]["language"]["name"])

                process_video(sample["mp4"], download_dir)
                process_poseformat(sample["pose-mediapipe.pose"], download_dir)
                break  # Remove this line to process all samples in each split
            except KeyError as e:
                logging.warning("Missing expected key in sample: %s", e)
            except (OSError, ValueError) as e:
                logging.error("Error processing sample: %s", e)


def process_video(mp4_data: bytes, tmp_dir: Path) -> None:
    temp_path = tmp_dir / "sample.mp4"
    try:
        temp_path.write_bytes(mp4_data)
        cap = cv2.VideoCapture(str(temp_path))

        if not cap.isOpened():
            logging.error("Failed to open video: %s", temp_path)
            return

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logging.info("Video Info: %d frames, %.2f FPS, %dx%d", frame_count, fps, width, height)

        ret, frame = cap.read()
        if ret:
            logging.info("First frame shape: %s, dtype: %s", frame.shape, frame.dtype)
            # Optional preview
            import matplotlib.pyplot as plt
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.title("First Frame")
            plt.axis("off")
            plt.show()
        else:
            logging.warning("Could not read first frame")

    finally:
        if temp_path.exists():
            temp_path.unlink()


def process_poseformat(pose_format_data: bytes, tmp_dir: Path) -> None:
    temp_path = tmp_dir / "sample.pose"
    try:
        temp_path.write_bytes(pose_format_data)
        pose = Pose.read(temp_path.read_bytes())
        logging.info("Mediapipe pose shape: %s", pose.body.data.shape)
    except (OSError, ValueError) as e:
        logging.error("Error processing pose-format: %s", e)
    finally:
        if temp_path.exists():
            temp_path.unlink()


if __name__ == "__main__":
    main()
