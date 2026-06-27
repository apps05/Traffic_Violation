

import os, shutil, re, cv2, yaml, json, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
from ultralytics import YOLO
import easyocr

os.makedirs('./models/easyocr', exist_ok=True)
print('models/ directory ready')

#phase 1:  Downloading yolov8s
model = YOLO('yolov8s.pt')
if not os.path.exists('./models/yolov8s.pt'):
    shutil.copy('yolov8s.pt', './models/yolov8s.pt')
size = os.path.getsize('./models/yolov8s.pt') / 1024**2
print(f' yolov8s.pt  →  {size:.1f} MB')

# Phase 2: Helmet Dataset & Train

import os
import zipfile
import yaml


# Extracting datasets
zip_filename = '/content/Motorcycle Helmet and License plate detection.yolov8.zip'
extract_dir = './my_dataset'

if os.path.exists(zip_filename):
    print(f"Extracting {zip_filename}...")
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction complete.")
else:
    print(f" Error: {zip_filename} not found. Please upload it to the sidebar.")

base_dir = os.path.abspath(extract_dir)
yaml_path = os.path.join(base_dir, 'data.yaml')

if not os.path.exists(yaml_path):
    # Search for data.yaml if it's nested
    for root, dirs, files in os.walk(extract_dir):
        if 'data.yaml' in files:
            yaml_path = os.path.join(root, 'data.yaml')
            break

if not os.path.exists(yaml_path):
    print(f"Error: Could not find data.yaml in {extract_dir}")
else:
    print(f" Scanning structure for: {base_dir}")

    # Read the current YAML
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Map Train images
    if os.path.exists(os.path.join(base_dir, 'train', 'images')):
        cfg['train'] = os.path.join(base_dir, 'train', 'images')
    else:
        cfg['train'] = os.path.join(base_dir, 'train')

    # Map Validation images  with some fallbacks
    if os.path.exists(os.path.join(base_dir, 'valid', 'images')):
        cfg['val'] = os.path.join(base_dir, 'valid', 'images')
    elif os.path.exists(os.path.join(base_dir, 'val', 'images')):
        cfg['val'] = os.path.join(base_dir, 'val', 'images')
    elif os.path.exists(os.path.join(base_dir, 'test', 'images')):
        print("No validation set found. Using 'test' set for validation.")
        cfg['val'] = os.path.join(base_dir, 'test', 'images')
    else:
        print("No validation or test set found. Using 'train' set for validation.")
        cfg['val'] = cfg['train']

    # Remove the 'path' key to force YOLO to use the absolute paths we just set
    if 'path' in cfg:
        del cfg['path']

    # Save the fixed YAML
    with open(yaml_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print("\nHelmet Dataset YAML paths have been mapped!")
    print(f" Train: {cfg['train']}")
    print(f" Val:   {cfg['val']}")
    print('Classes:', cfg.get('names'))

import os
import torch
import shutil
from ultralytics import YOLO

data_yaml = './my_dataset/data.yaml'

if not os.path.exists(data_yaml):
    print(f"Could not find data.yaml at '{data_yaml}'.")
else:
    print(f" YAML found at '{data_yaml}'. Booting up YOLO")

    #RUN TRAINING
    helmet_model = YOLO('yolov8s.pt')
    helmet_model.train(
        data     = data_yaml,
        epochs   = 50,
        imgsz    = 640,
        batch    = 16,
        name     = 'helmet_finetune',
        exist_ok = True,
        patience = 15,
        device   = 0 if torch.cuda.is_available() else 'cpu',
        verbose  = True,
    )

    #sAVE THE MODEL
    os.makedirs('./models', exist_ok=True) # Ensure the models folder exists
    best = 'runs/detect/helmet_finetune/weights/best.pt'
    shutil.copy(best, './models/helmet_model.pt')
    size = os.path.getsize('./models/helmet_model.pt') / 1024**2
    print(f'helmet_model.pt saved  →  {size:.1f} MB')

#Phase 3: Plate Dataset & Train

import os
import zipfile
import yaml

zip_filename = '/content/Vehicle Registration Plates.yolov8.zip'
extract_dir = './vehicle_plates_dataset'

# Extract the file
if os.path.exists(zip_filename):
    print(f"Extracting {zip_filename}...")
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(" Extraction complete.")
else:
    print(f" Error: {zip_filename} not found.")

# Locate and load the data.yaml
yaml_path = None
for root, dirs, files in os.walk(extract_dir):
    if 'data.yaml' in files:
        yaml_path = os.path.join(root, 'data.yaml')
        break

if yaml_path:
    with open(yaml_path) as f:
        cfg2 = yaml.safe_load(f)

    print('Classes:', cfg2['names'])
    print('Dataset at:', extract_dir)
else:
    print(" Could not find data.yaml inside the extracted folder.")

import os
import yaml

base_dir = os.path.abspath('./vehicle_plates_dataset')
yaml_path = os.path.join(base_dir, 'data.yaml')

if not os.path.exists(yaml_path):
    print(f"Error: Could not find {yaml_path}.")
else:
    print(f"Scanning structure for: {base_dir}")

    #  Read the current YAML
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Map Train images
    # Checking for common Roboflow sub-folder structures
    if os.path.exists(os.path.join(base_dir, 'train', 'images')):
        cfg['train'] = os.path.join(base_dir, 'train', 'images')
    else:
        cfg['train'] = os.path.join(base_dir, 'train')

    # Map Validation images (with fallbacks)
    if os.path.exists(os.path.join(base_dir, 'valid', 'images')):
        cfg['val'] = os.path.join(base_dir, 'valid', 'images')
    elif os.path.exists(os.path.join(base_dir, 'val', 'images')):
        cfg['val'] = os.path.join(base_dir, 'val', 'images')
    elif os.path.exists(os.path.join(base_dir, 'test', 'images')):
        print(" No validation set found. Using 'test' set for validation.")
        cfg['val'] = os.path.join(base_dir, 'test', 'images')
    else:
        print("No validation or test set found. Using 'train' set for validation.")
        cfg['val'] = cfg['train']

    # Remove the 'path' key to ensure absolute paths are used
    if 'path' in cfg:
        del cfg['path']

    # Save the fixed YAML
    with open(yaml_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print("\n License Plate YAML paths have been mapped!")
    print(f" Train: {cfg['train']}")
    print(f" Val:   {cfg['val']}")

import os
import shutil
from ultralytics import YOLO

data_yaml2 = './vehicle_plates_dataset/data.yaml'

if not os.path.exists(data_yaml2):
    print(f" Error: {data_yaml2} not found.")
else:
    print(f" Starting training for Dataset 2 (License Plates) using {data_yaml2}...")

    #  Initialize and Train
    plate_model = YOLO('yolov8s.pt')
    plate_model.train(
        data     = data_yaml2,
        epochs   = 50,
        imgsz    = 640,
        batch    = 16,
        name     = 'plate_finetune',
        exist_ok = True,
        patience = 15,
        device   = 0 if torch.cuda.is_available() else 'cpu',
        verbose  = True,
    )

    os.makedirs('./models', exist_ok=True)

    best_plate_weights = 'runs/detect/plate_finetune/weights/best.pt'
    if os.path.exists(best_plate_weights):
        shutil.copy(best_plate_weights, './models/plate_model.pt')
        size = os.path.getsize('./models/plate_model.pt') / 1024**2
        print(f' plate_model.pt saved successfully  →  {size:.1f} MB')
    else:
        print("Training folder not found. Check if the training finished successfully.")

reader = easyocr.Reader(
    ['en'],
    gpu=False,
    model_storage_directory='./models/easyocr',
    download_enabled=True,
    verbose=True,
)
print('EasyOCR weights downloaded')

def dir_mb(path):
    total = 0
    for dp, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(dp, f))
    return total / 1024**2

items = [
    ('yolov8s.pt',             './models/yolov8s.pt'),
    ('helmet_model.pt',        './models/helmet_model.pt'),
    ('plate_model.pt',         './models/plate_model.pt'),
    ('easyocr/ (all weights)', './models/easyocr'),
]
total = 0
print('─' * 52)
for label, path in items:
    mb = dir_mb(path) if os.path.isdir(path) else (
         os.path.getsize(path)/1024**2 if os.path.isfile(path) else 0)
    total += mb
    print(f'  {label:35s}  {mb:6.1f} MB')
print('─' * 52)
status = ' Within limit' if total < 250 else ' OVER LIMIT'
print(f'  {"TOTAL":35s}  {total:6.1f} MB / 250 MB  {status}')
