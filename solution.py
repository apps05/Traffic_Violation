import os
import cv2
import re
import numpy as np
from ultralytics import YOLO
import easyocr

class TrafficViolationDetector:
     # COCO class ID for motorcycle
    MOTORCYCLE_CLASS_IDS = {3}
    BIKE_CONF            = 0.30
    HELMET_CONF          = 0.30
    PLATE_CONF           = 0.30

    def __init__(self, model_dir: str = "./models"):
        # Explicit local paths prevent YOLO from trying to download from GitHub
        self.detector       = YOLO(os.path.join(model_dir, "yolov8s.pt"))
        self.helmet_model   = YOLO(os.path.join(model_dir, "helmet_model.pt"))
        self.plate_model    = YOLO(os.path.join(model_dir, "plate_model.pt"))
        
        self.helmet_classes = self.helmet_model.names
        self._hm_name2id = {v.strip().lower(): k for k, v in self.helmet_classes.items()}
         # EasyOCR model directory
        easyocr_dir = os.path.join(model_dir, "easyocr")
        os.makedirs(easyocr_dir, exist_ok=True)
        
        # download_enabled=False acts as a hard kill-switch against internet access
         # OCR initialized in fully offline mode
        self.ocr = easyocr.Reader(
            ["en"], 
            gpu=False, 
            model_storage_directory=easyocr_dir,
            download_enabled=False, 
            verbose=False,
        )

    def predict(self, image_path: str) -> dict:
        image = cv2.imread(image_path)
        if image is None: return {"violations": []}
 

        # Detect motorcycles / scooters
        two_wheelers = self._detect_two_wheelers(image)
        if not two_wheelers: return {"violations": []}

        violations = []
        for bbox, crop in two_wheelers:
            num_riders, helmet_violations = self._analyze_riders(crop)
            # Only create entry if violation exists
            if(num_riders>2 or helmet_violations>0):
                # Detect motorcycles / scooters
                license_plate = self._get_license_plate(crop)
                violations.append({
                "num_riders":        num_riders,
                "helmet_violations": helmet_violations,
                "license_plate":     license_plate
                })

            
        return {"violations": violations} 

    def _detect_two_wheelers(self, image: np.ndarray) -> list:
         # Run YOLOv8 inference on the full input image
        results = self.detector(image, verbose=False, augment=True)[0]
        h, w = image.shape[:2]
        detections = []
        # Iterate through all detected objects
        for box in results.boxes:
            if int(box.cls[0]) not in self.MOTORCYCLE_CLASS_IDS: continue
            if float(box.conf[0]) < self.BIKE_CONF: continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_w, box_h = x2 - x1, y2 - y1

            # Add padding around motorcycle region
          # This helps capture rider heads and rear license plate
            pad_x = int(box_w * 0.25)
            pad_top = int(box_h * 0.80)
            pad_bottom = int(box_h * 0.40)

            crop = image[
                max(0, y1 - pad_top) : min(h, y2 + pad_bottom),
                max(0, x1 - pad_x)   : min(w, x2 + pad_x),
            ]
            if crop.size == 0:
                continue
             # Store original bounding box and cropped vehicle image
            detections.append(((x1,y1,x2,y2), crop))
        return detections

    def _analyze_riders(self, crop: np.ndarray) -> tuple:
         # Run the custom helmet/rider detection model on the cropped two-wheeler image
        results  = self.helmet_model(crop, verbose=False, augment=True)[0]
        hid = self._hm_name2id.get('helmet')
        rid = self._hm_name2id.get('rider')
        if rid is None:
            print("Warning: 'rider' class not found in helmet model. Rider count may be 0.")
        hc = rc = 0
        for box in results.boxes:
            if float(box.conf[0]) < self.HELMET_CONF: continue
            cid = int(box.cls[0])
            if cid == hid:   hc += 1
            elif cid == rid: rc += 1
        # Use max(rc, hc) to avoid undercounting riders when body is hidden but helmet is visible
        num_riders = max(rc, hc)
        helmet_violations = max(0, num_riders - hc)
        return num_riders, helmet_violations

    def _preprocess_light(self, plate: np.ndarray) -> np.ndarray:
        # Get license plate crop height and width
        h, w = plate.shape[:2]
        p = cv2.resize(plate, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
          # Convert to grayscale if the plate image is colored
        return cv2.cvtColor(p, cv2.COLOR_BGR2GRAY) if p.ndim == 3 else p.copy()

    def _preprocess_medium(self, plate: np.ndarray) -> np.ndarray:
         # Get license plate crop height and width
        h, w = plate.shape[:2]
        p = cv2.resize(plate, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY) if p.ndim == 3 else p.copy()
        # Apply Gaussian blur to reduce small noise before thresholding
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def _preprocess_heavy(self, plate: np.ndarray) -> np.ndarray:
          # First try to straighten the plate if it is tilted
        plate = self._deskew(plate)
        h,w   = plate.shape[:2]
        plate = cv2.resize(plate, (w*3,h*3), interpolation=cv2.INTER_CUBIC)
        gray  = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY) if plate.ndim==3 else plate.copy()
        gray  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
        gray  = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2,1)))
        gray  = cv2.fastNlMeansDenoising(gray, h=10)
         # Create a blurred version for sharpening
        blur  = cv2.GaussianBlur(gray, (0,0), 3)
        # Sharpen the image so OCR can detect characters better

        return cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

    def _deskew(self, img: np.ndarray) -> np.ndarray:
         # Convert image to grayscale for edge detection
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        minL  = max(img.shape[1]//4, 10)
        lines = cv2.HoughLinesP(edges,1,np.pi/180,25,minLineLength=minL,maxLineGap=10)
        if lines is None: return img
        # Calculate angles of detected lines
        # Only keep reasonable angles below 45 degrees
        angs = [np.degrees(np.arctan2(l[0][3]-l[0][1], l[0][2]-l[0][0]))
                for l in lines if abs(np.degrees(np.arctan2(l[0][3]-l[0][1],l[0][2]-l[0][0]))) < 45]
        if not angs: return img
        a = float(np.median(angs))
        if abs(a) < 1.0 or abs(a) > 15.0: return img
        h,w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2,h//2), a, 1.0)
        # Rotate image to correct skew and preserve border pixels
        return cv2.warpAffine(img, M, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def _read_text_with_score(self, img: np.ndarray) -> tuple:
        try:
            results = self.ocr.readtext(img, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", detail=1, mag_ratio=1.5, paragraph=False)
        except Exception: 
            return ('', 0.0)

        if not results: 
            return ('', 0.0)

        # 1. Sort primarily by the Y-coordinate (top to bottom)
        # EasyOCR bbox format: [[top-left, top-right, bottom-right, bottom-left], text, conf]
        # r[0][0][1] gets the Y-coordinate of the top-left corner
        results.sort(key=lambda r: r[0][0][1])

        lines = []
        current_line = [results[0]]
        
        # 2. Group into distinct lines based on vertical overlap
        for current_box in results[1:]:
            prev_box = current_line[-1]
            
            # Calculate Y-centers and heights
            prev_y_center = (prev_box[0][0][1] + prev_box[0][2][1]) / 2
            curr_y_center = (current_box[0][0][1] + current_box[0][2][1]) / 2
            prev_height = prev_box[0][2][1] - prev_box[0][0][1]
            
            # If the vertical distance is less than half a character's height, they are on the same line
            if abs(curr_y_center - prev_y_center) < (prev_height * 0.5):
                current_line.append(current_box)
            else:
                lines.append(current_line)
                current_line = [current_box]
                
        lines.append(current_line) # Append the final line

        full_text = ""
        total_conf = 0.0
        num_boxes = 0

        # 3. Sort each line left-to-right and construct the final string
        for line in lines:
            line.sort(key=lambda r: r[0][0][0]) # Sort by X-coordinate
            for bbox, text, conf in line:
                full_text += text
                total_conf += conf
                num_boxes += 1

        return (full_text, total_conf / num_boxes if num_boxes > 0 else 0.0)

    def _get_license_plate(self, crop: np.ndarray) -> str:
        plate_crop = self._ensemble_plate_crop(crop)
        if plate_crop is None: return ""

        results_pool = []

        light_img = self._preprocess_light(plate_crop)
        light_text, light_score = self._read_text_with_score(light_img)
        if light_text: results_pool.append((light_score, light_text))
        if light_score > 0.70 and len(light_text) >= 4: return self._postprocess_plate_text(light_text)

        medium_img = self._preprocess_medium(plate_crop)
        medium_text, medium_score = self._read_text_with_score(medium_img)
        if medium_text: results_pool.append((medium_score, medium_text))
        if medium_score > 0.70 and len(medium_text) >= 4: return self._postprocess_plate_text(medium_text)

        heavy_img = self._preprocess_heavy(plate_crop)
        heavy_text, heavy_score = self._read_text_with_score(heavy_img)
        if heavy_text: results_pool.append((heavy_score, heavy_text))

        if not results_pool: return ""

        results_pool.sort(reverse=True, key=lambda x: x[0])
        return self._postprocess_plate_text(results_pool[0][1])

    def _ensemble_plate_crop(self, crop: np.ndarray):
        H, W = crop.shape[:2]
        boxes, scores = [], []
        TEMP_CONF = self.PLATE_CONF

        def is_valid_plate_shape(xyxy):
            box_w, box_h = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
            if box_h == 0 or box_w < 30 or box_h < 10: return False
            if 0.5 <= (float(box_w) / float(box_h)) <= 8.0: return True
            return False

        for box in self.plate_model(crop, verbose=False)[0].boxes:
            c = float(box.conf[0])
            if c >= TEMP_CONF and is_valid_plate_shape(box.xyxy[0]):
                boxes.append(list(map(int, box.xyxy[0]))); scores.append(c)

        pid = None
        for class_name, class_id in self._hm_name2id.items():
            if any(k in class_name for k in ['plate', 'lisc', 'number']):
                pid = class_id; break

        if pid is not None:
            for box in self.helmet_model(crop, verbose=False)[0].boxes:
                if int(box.cls[0]) == pid:
                    c = float(box.conf[0])
                    if c >= TEMP_CONF and is_valid_plate_shape(box.xyxy[0]):
                        boxes.append(list(map(int, box.xyxy[0]))); scores.append(c)

        if not boxes: return None

        keep = self._nms(np.array(boxes,np.float32), np.array(scores,np.float32))
        x1, y1, x2, y2 = map(int, np.array(boxes)[keep[0]])
        box_w, box_h = x2 - x1, y2 - y1

        PAD_X = max(40, int(box_w * 0.50))
        PAD_Y = max(15, int(box_h * 0.30))

        px1 = max(0, x1 - PAD_X)
        px2 = min(W, x2 + PAD_X)
        py1 = max(0, y1 - PAD_Y)
        py2 = min(H, y2 + PAD_Y)

        c = crop[py1:py2, px1:px2]
        return c if c.size > 0 else None

    @staticmethod
    def _nms(boxes, scores, iou=0.30):
        x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
        areas = (x2-x1+1)*(y2-y1+1)
        order = scores.argsort()[::-1]; keep=[]
        while order.size:
            i=order[0]; keep.append(i)
            ix1=np.maximum(x1[i],x1[order[1:]]); iy1=np.maximum(y1[i],y1[order[1:]])
            ix2=np.minimum(x2[i],x2[order[1:]]); iy2=np.minimum(y2[i],y2[order[1:]])
            inter=np.maximum(0,ix2-ix1+1)*np.maximum(0,iy2-iy1+1)
            iou_v=inter/(areas[i]+areas[order[1:]]-inter)
            order=order[np.where(iou_v<=iou)[0]+1]
        return keep

    def _postprocess_plate_text(self, raw: str) -> str:
        #Clean up characters: Uppercase and keep only Alpha-Numeric characters
        text = re.sub(r'[^A-Z0-9]', '', raw.upper())
        return text