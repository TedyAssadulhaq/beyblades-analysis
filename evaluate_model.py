import os
import numpy as np
from pathlib import Path
import cv2
from ultralytics import YOLO
import glob

def find_model_file():
    model_files = glob.glob(os.path.join("model", "*.pt"))
    return model_files[0]

def load_ground_truth_labels(label_path):
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5 and int(parts[0]) == 0:
                    x_center, y_center, width, height = map(float, parts[1:5])
                    labels.append({'x_center': x_center, 'y_center': y_center, 'width': width, 'height': height})
    return labels

def yolo_to_xyxy(x_center, y_center, width, height, img_width, img_height):
    x_center_abs = x_center * img_width
    y_center_abs = y_center * img_height
    width_abs = width * img_width
    height_abs = height * img_height
    
    x1 = x_center_abs - width_abs / 2
    y1 = y_center_abs - height_abs / 2
    x2 = x_center_abs + width_abs / 2
    y2 = y_center_abs + height_abs / 2
    
    return x1, y1, x2, y2

def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def calculate_ap(precisions, recalls):
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
    
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    
    return ap

def trim_bounding_box(bbox):
    x1, y1, x2, y2 = bbox
    height = y2 - y1
    y1_new = y1 + (height * 0.15)
    y2_new = y2 - (height * 0.15)
    return x1, y1_new, x2, y2_new

def calculate_map_50(gt_boxes_all, pred_boxes_all):
    all_predictions = []
    all_ground_truths = []
    
    for img_name in gt_boxes_all.keys():
        gt_boxes = gt_boxes_all[img_name]
        pred_boxes = pred_boxes_all.get(img_name, [])
        
        for gt_box in gt_boxes:
            all_ground_truths.append({'image': img_name, 'bbox': gt_box['bbox'], 'used': False})
        
        for pred_box in pred_boxes:
            all_predictions.append({'image': img_name, 'bbox': pred_box['bbox'], 'confidence': pred_box['confidence']})
    
    all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
    
    tp = 0
    fp = 0
    precisions = []
    recalls = []
    
    for pred in all_predictions:
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(all_ground_truths):
            if gt['image'] == pred['image'] and not gt['used']:
                iou = calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
        
        if best_iou >= 0.5:
            tp += 1
            all_ground_truths[best_gt_idx]['used'] = True
        else:
            fp += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / len(all_ground_truths) if len(all_ground_truths) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    if len(precisions) > 0 and len(recalls) > 0:
        return calculate_ap(np.array(precisions), np.array(recalls))
    return 0.0

def test_yolo_model(model_path, images_dir, labels_dir, confidence_threshold=0.25):
    model = YOLO(model_path)
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(images_dir).glob(f'*{ext}'))
        image_files.extend(Path(images_dir).glob(f'*{ext.upper()}'))
    
    ground_truth_boxes = {}
    predicted_boxes = {}
    
    for img_path in image_files:
        img_name = img_path.stem
        label_path = os.path.join(labels_dir, f"{img_name}.txt")
        
        img = cv2.imread(str(img_path))
        img_height, img_width = img.shape[:2]
        
        gt_labels = load_ground_truth_labels(label_path)
        gt_boxes = []
        for label in gt_labels:
            x1, y1, x2, y2 = yolo_to_xyxy(label['x_center'], label['y_center'], label['width'], label['height'], img_width, img_height)
            gt_boxes.append({'bbox': (x1, y1, x2, y2), 'class': 0})
        ground_truth_boxes[img_name] = gt_boxes
        
        results = model(str(img_path), verbose=False)
        pred_boxes = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls_id == 0 and conf >= confidence_threshold:
                        x1, y1, x2, y2 = map(float, box.xyxy[0])
                        x1, y1, x2, y2 = trim_bounding_box((x1, y1, x2, y2))
                        pred_boxes.append({'bbox': (x1, y1, x2, y2), 'confidence': conf, 'class': cls_id})
        predicted_boxes[img_name] = pred_boxes
    
    return calculate_map_50(ground_truth_boxes, predicted_boxes)

def main():
    model_path = find_model_file()
    images_dir = "testdataset/test/images"
    labels_dir = "testdataset/test/labels"
    
    map_50 = test_yolo_model(model_path, images_dir, labels_dir, confidence_threshold=0.25)
    
    print(f"mAP@0.5: {map_50:.4f}")
    print(f"mAP@0.5 (%): {map_50*100:.2f}%")

if __name__ == "__main__":
    main()