import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv12 with Ultralytics.")
    # Pretrained model or checkpoint to start from.
    parser.add_argument("--model", default="yolo12s.pt", help="Model or weights path")
    # Dataset definition YAML with train/val/test paths and class names.
    parser.add_argument(
        "--data",
        default="/home/ubu/Desktop/Yolo-Train/data/SeaTrekker_Dataset/data.yaml",
        help="Dataset YAML path",
    )
    # Total training epochs.
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    # Early stopping patience (epochs without improvement).
    parser.add_argument(
        "--patience", type=int, default=30, help="Early stopping patience"
    )
    # Input image size (square).
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    # Samples per batch.
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    # Root directory for run artifacts.
    parser.add_argument("--project", default="output", help="Output root directory")
    # Subfolder name under project for this run.
    parser.add_argument("--name", default="train", help="Run name under project dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.project).mkdir(parents=True, exist_ok=True)
    # Keep Ultralytics config in a writable workspace path.
    yolo_config_dir = Path(args.project) / ".yolo-config"
    yolo_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_config_dir.resolve()))

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is not installed. Run: pip install ultralytics"
        ) from exc

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
