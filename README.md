# Traffic Rule Violation Detection

## Overview

A computer vision pipeline that detects traffic rule violations on two-wheelers from street images. The system identifies:
- More than two riders on a single two-wheeler
- Riders not wearing helmets
- Extracts and reads the license plate of any violating vehicle

---

## System Architecture

```
Input Image
    │
    ▼
┌─────────────────────────┐
│  YOLOv8s (COCO)         │  Detects two-wheelers (class 3 = motorcycle)
└─────────────────────────┘
    │  padded crop per bike
    ▼
┌─────────────────────────┐
│  Helmet Model (YOLOv8s) │  Counts riders & helmets → computes violations
└─────────────────────────┘
    │  if violation found
    ▼
┌─────────────────────────┐
│  Plate Model (YOLOv8s)  │  Localises license plate region (ensemble + NMS)
└─────────────────────────┘
    │  plate crop
    ▼
┌─────────────────────────┐
│  EasyOCR                │  3-stage preprocessing → best OCR result
└─────────────────────────┘
    │
    ▼
Output: { "violations": [ { "num_riders", "helmet_violations", "license_plate" } ] }
```

---

## Models Used

| File | Purpose | Source |
|---|---|---|
| `yolov8s.pt` | General object detection (two-wheeler localisation) | Ultralytics pretrained (COCO) |
| `helmet_model.pt` | Helmet & rider detection | YOLOv8s fine-tuned on *Motorcycle Helmet and License Plate Detection* dataset (Roboflow) |
| `plate_model.pt` | License plate localisation | YOLOv8s fine-tuned on *Vehicle Registration Plates* dataset (Roboflow) |
| `easyocr/` | OCR weights | EasyOCR English model (downloaded once in `train.py`) |

All model files are stored in `./models/` and the total size stays within the **250 MB** limit.

---

## Directory Structure

```
<BT2024256_BT2024258>/
├── solution.py          # TrafficViolationDetector class
├── train.py             # Full training pipeline (datasets → fine-tuning → save)
├── test.py              # Demo testing script
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── models/
    ├── yolov8s.pt
    ├── helmet_model.pt
    ├── plate_model.pt
    └── easyocr/
        └── (EasyOCR weight files)
```

---

## Setup & Installation

```bash
pip install -r requirements.txt
```

> **Note:** During evaluation, internet access is disabled. All model weights must already be present in `./models/`. Run `train.py` once (with internet access) to download and generate all weights.

---

## Training Pipeline (`train.py`)

1. **Download base model** — `yolov8s.pt` from Ultralytics.
2. **Helmet model** — Fine-tune on *Motorcycle Helmet and License Plate Detection* dataset for 50 epochs; save best weights as `models/helmet_model.pt`.
3. **Plate model** — Fine-tune on *Vehicle Registration Plates* dataset for 50 epochs; save best weights as `models/plate_model.pt`.
4. **EasyOCR** — Download English weights into `models/easyocr/`.

Training uses GPU if available (`device=0`), otherwise CPU.

---

## Inference (`solution.py`)

```python
from solution import TrafficViolationDetector

model = TrafficViolationDetector('./models')
output = model.predict('test1.jpeg')
print(output)
# {'violations': [{'num_riders': 2, 'helmet_violations': 1, 'license_plate': 'MH12AB1234'}]}
```

### Key Design Decisions

- **Padded crop:** Each detected two-wheeler is cropped with generous padding (80% top, 40% bottom, 25% sides) to capture riders and plate fully.
- **Ensemble plate detection:** Both the dedicated plate model and the helmet model's plate class are queried; NMS merges overlapping boxes.
- **3-stage OCR preprocessing:** Light (resize + grayscale) → Medium (Otsu threshold) → Heavy (CLAHE + deskew + denoising + sharpening). The pipeline returns early if a high-confidence result (> 0.70) is found.
- **Stateless predict():** No state is shared between calls; safe for sequential or parallel evaluation.

---

## Output Format

```json
{
  "violations": [
    {
      "num_riders": 2,
      "helmet_violations": 1,
      "license_plate": "MH12AB1234"
    }
  ]
}
```

- `num_riders` — total riders detected on the two-wheeler.
- `helmet_violations` — number of riders without a helmet.
- `license_plate` — alphanumeric string extracted by OCR (empty string `""` if unreadable).

A violation entry is only produced when `num_riders > 2` **or** `helmet_violations > 0`.

---

## Failure Cases & Limitations

- **Severe occlusion:** Riders whose upper body is fully hidden may not be detected by the helmet model, leading to an undercount.
- **Night / very dark images:** OCR accuracy degrades on low-contrast plates even with CLAHE.
- **Non-standard plates:** Plates with regional scripts or unusual fonts may produce partial reads.
- **Side-on angles:** The plate model performs best on front/rear-facing views; heavily angled plates may be missed.

---

## Constraints Compliance

| Constraint | Status |
|---|---|
| Total model size ≤ 250 MB | Verified in `train.py` |
| No VLMs > 1B parameters | YOLOv8s (~22M params) + EasyOCR |
| Offline execution | `download_enabled=False` in EasyOCR; YOLO loads from local path |