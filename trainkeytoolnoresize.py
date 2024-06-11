import argparse
from ultralytics import YOLO
import yaml

# Argumentparser für --mode-train und --mode-val
parser = argparse.ArgumentParser(description="Train or validate YOLO model")
parser.add_argument('--mode-train', action='store_true', help='Train the model')
parser.add_argument('--mode-val', action='store_true', help='Validate the model')
args = parser.parse_args()

# Überprüfen der Argumente
if not args.mode_train and not args.mode_val:
    mode = input("Please specify the mode: 'train' or 'val': ").strip().lower()
    if mode == 'train':
        args.mode_train = True
    elif mode == 'val':
        args.mode_val = True
    else:
        raise ValueError("Invalid mode specified")

# Modell laden und konfigurieren
model = YOLO('yolov8n-pose.pt')

if args.mode_train:
    model.train(
        data='config.yaml', 
        epochs=10, 
        imgsz=256, 
        patience=10, 
        batch=-1, 
        optimizer='Adam', 
        lr0=0.01, 
        warmup_epochs=10
    )
elif args.mode_val:
    model.val(data='config.yaml')
