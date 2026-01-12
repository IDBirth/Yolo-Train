# Yolo-Train

Train and run inference with YOLOv12 using the Ultralytics API.

## Requirements
- Python 3.10+
- PyTorch (install the build that matches your CPU/GPU)
- Python packages in `requirements.txt`

## Environment Setup
```bash
python -m venv Yolo_env
source Yolo_env/bin/activate
pip install --upgrade pip
# Install PyTorch first (CPU/GPU build)
# https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

## Data Setup
Update the dataset YAML path if needed:
```bash
python train.py --data data/SeaTrekker_Dataset/data.yaml
```
The default in `train.py` is an absolute path, so you may want to change it
or always pass `--data`.

## Train
```bash
python train.py --model yolo12s.pt --data data/SeaTrekker_Dataset/data.yaml
```

## Predict
```bash
python predict.py --weights output/train/weights/best.pt --source data
```

## Outputs
Runs are written under `output/` by default, and Ultralytics config files are
kept in `output/.yolo-config/`.
