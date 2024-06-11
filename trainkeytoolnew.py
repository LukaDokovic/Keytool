import argparse
from ultralytics import YOLO
import cv2
import os
import yaml

def resize_images(image_dir, output_size=(256, 256)):
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith(('png', 'jpg', 'jpeg')):
                
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                print(img.shape)
                resized_img = cv2.resize(img, output_size)
                cv2.imwrite(img_path, resized_img)
                print(resized_img.shape)


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

# Bildergröße ändern
data_config = 'config.yaml'
data = yaml.safe_load(open(data_config))

# Debug-Ausgabe für Keypoints
print("Keypoint shape from config:", data['kpt_shape'])

# Bildergröße ändern
resize_images(data['path'])

# Model laden und konfigurieren
model = YOLO('yolov8n-pose.pt')

# Debug-Ausgabe für den Validierungsmodus
if args.mode_val:
    print("Validation mode selected")

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
    results = model.val(data='config.yaml')
    # Debug-Ausgabe der Ergebnisse
    print(results)


