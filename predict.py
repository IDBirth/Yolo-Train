import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with YOLOv12.")
    parser.add_argument(
        "--weights",
        default="output/train/weights/best.pt",
        help="Weights path",
    )
    parser.add_argument(
        "--source",
        default="data",
        help="Image/dir/video source",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--project", default="output", help="Output root directory")
    parser.add_argument("--name", default="predict", help="Run name under project dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.project).mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is not installed. Run: pip install ultralytics"
        ) from exc

    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        project=args.project,
        name=args.name,
        save=True,
    )


if __name__ == "__main__":
    main()
