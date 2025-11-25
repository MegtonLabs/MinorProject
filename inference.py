import argparse
import csv
import json
import os

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from check_detection_model import CheckLayoutDetector, OCRProcessor, get_field_names

class CheckInference:
    """Inference class for check layout detection and OCR"""
    
    def __init__(
        self,
        model_path,
        device='cuda',
        confidence_threshold=0.7,
        model_variant=None,
        num_classes=None,
        use_custom_anchors=None,
    ):
        """
        Args:
            model_path: Path to trained model checkpoint
            device: 'cuda' or 'cpu'
            confidence_threshold: Minimum score for detections
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        self.csv_headers = ['image', 'field', 'confidence', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'text']
        checkpoint = None
        checkpoint_config = {}
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            checkpoint_config = checkpoint.get('config', {})
        else:
            print(f"Warning: No checkpoint found at {model_path}. Using random weights.")
        
        resolved_variant = (model_variant or checkpoint_config.get('model_variant') or 'resnet50_fpn')
        resolved_num_classes = num_classes or checkpoint_config.get('num_classes') or (len(get_field_names()) + 1)
        if use_custom_anchors is None:
            resolved_custom_anchors = checkpoint_config.get('use_custom_anchors', True)
        else:
            resolved_custom_anchors = use_custom_anchors
        
        # Load model
        self.model = CheckLayoutDetector(
            num_classes=resolved_num_classes,
            model_variant=resolved_variant,
            use_custom_anchors=resolved_custom_anchors,
        ).to(self.device)
        self.model.eval()
        
        # Load checkpoint weights if available
        if checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {model_path}")
            if checkpoint_config:
                print(f"Checkpoint config: variant={checkpoint_config.get('model_variant')}, classes={checkpoint_config.get('num_classes')}")
        
        # OCR processor
        self.ocr = OCRProcessor()
        self.field_names = get_field_names()
        self.model_variant = resolved_variant
    
    def predict(self, image_path):
        """
        Predict fields in a check image
        
        Args:
            image_path: Path to check image
            
        Returns:
            dict: Detection results and OCR text
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            predictions = self.model([img_tensor.squeeze(0)])[0]
        
        # Process predictions
        results = []
        boxes = predictions['boxes'].cpu()
        scores = predictions['scores'].cpu()
        labels = predictions['labels'].cpu()
        
        # Filter by confidence
        keep = scores >= self.confidence_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        
        # Convert image to numpy for OCR
        img_np = np.array(image)
        
        # Extract text from each detection
        for i in range(len(boxes)):
            box = boxes[i]
            label = labels[i].item()
            score = scores[i].item()
            
            field_name = self.field_names.get(label, f"Class_{label}")
            
            # Extract text via OCR
            text = self.ocr.extract_text(img_np, box)
            
            results.append({
                'field': field_name,
                'bbox': box.tolist(),
                'confidence': score,
                'text': text
            })
        
        return results
    
    def visualize(self, image_path, results, output_path=None):
        """
        Visualize detections on image
        
        Args:
            image_path: Input image path
            results: Detection results from predict()
            output_path: Path to save visualized image
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Colors for different fields
        colors = {
            'Date': (255, 0, 0),
            'Courtesy_Amount': (0, 255, 0),
            'Legal_Amount': (0, 0, 255),
            'Account_Number': (255, 255, 0),
            'Signature': (255, 0, 255),
            'Payee': (0, 255, 255)
        }
        
        for idx, res in enumerate(results):
            bbox = res['bbox']
            field = res['field']
            confidence = res['confidence']
            text = res['text']
            
            # Draw bounding box
            color = colors.get(field, (128, 128, 128))
            cv2.rectangle(image, 
                         (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), 
                         color, 2)
            
            # Draw label
            label_text = f"{field}: {confidence:.2f}"
            if text:
                label_text += f" | Text: {text[:20]}"
            
            cv2.putText(image, label_text, 
                       (int(bbox[0]), int(bbox[1]) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Convert back to BGR for saving
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if output_path:
            cv2.imwrite(output_path, image_bgr)
            print(f"Visualization saved to {output_path}")
        
        return image_bgr
    
    def detections_to_rows(self, image_name, detections):
        """Convert detections to CSV row format."""
        rows = []
        for det in detections:
            bbox = det['bbox']
            rows.append({
                'image': image_name,
                'field': det['field'],
                'confidence': round(det['confidence'], 4),
                'bbox_x1': round(bbox[0], 2),
                'bbox_y1': round(bbox[1], 2),
                'bbox_x2': round(bbox[2], 2),
                'bbox_y2': round(bbox[3], 2),
                'text': det['text'],
            })
        return rows
    
    def save_results_to_csv(self, csv_path, rows):
        """Append detection rows to a CSV file with headers."""
        if not csv_path or not rows:
            return
        os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
        file_exists = os.path.exists(csv_path)
        with open(csv_path, 'a', newline='', encoding='utf-8') as fp:
            writer = csv.DictWriter(fp, fieldnames=self.csv_headers)
            if not file_exists or os.path.getsize(csv_path) == 0:
                writer.writeheader()
            writer.writerows(rows)
        print(f"📝 Wrote {len(rows)} rows to {csv_path}")
    
    def process_batch(self, image_dir, output_dir, csv_path=None):
        """Process all images in a directory"""
        os.makedirs(output_dir, exist_ok=True)
        results = {}
        all_rows = []
        
        image_files = [f for f in os.listdir(image_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
        
        for img_file in tqdm(image_files, desc="Processing images"):
            img_path = os.path.join(image_dir, img_file)
            
            # Predict
            detections = self.predict(img_path)
            results[img_file] = detections
            all_rows.extend(self.detections_to_rows(img_file, detections))
            
            # Visualize
            viz_path = os.path.join(output_dir, f"viz_{img_file}")
            self.visualize(img_path, detections, viz_path)
        
        # Save results JSON
        json_path = os.path.join(output_dir, "detection_results.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Batch results saved to {json_path}")
        
        if csv_path:
            self.save_results_to_csv(csv_path, all_rows)
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Run inference on check images')
    parser.add_argument('--model', default='checkpoints/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--image', required=True,
                       help='Path to single image or directory')
    parser.add_argument('--output', default='outputs',
                       help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.7,
                       help='Confidence threshold')
    parser.add_argument('--batch', action='store_true',
                       help='Process directory as batch')
    parser.add_argument('--device', default='cuda',
                        help='Computation device (cuda or cpu)')
    parser.add_argument('--csv', default=None,
                        help='Optional CSV path (defaults to <output>/check_results.csv)')
    parser.add_argument('--model-variant', default=None,
                        help='Override model variant (auto-detected from checkpoint if omitted)')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Override number of classes (including background)')
    parser.add_argument('--no-custom-anchors', action='store_true',
                        help='Disable custom anchor configuration at inference')
    
    args = parser.parse_args()
    
    csv_path = args.csv or os.path.join(args.output, 'check_results.csv')
    
    # Initialize inference
    inference = CheckInference(
        args.model,
        device=args.device,
        confidence_threshold=args.conf,
        model_variant=args.model_variant,
        num_classes=args.num_classes,
        use_custom_anchors=False if args.no_custom_anchors else None,
    )
    
    if args.batch:
        # Process directory
        results = inference.process_batch(args.image, args.output, csv_path=csv_path)
    else:
        # Process single image
        results = inference.predict(args.image)
        print(json.dumps(results, indent=2))
        
        # Visualize
        viz_path = os.path.join(args.output, "visualized.jpg")
        os.makedirs(args.output, exist_ok=True)
        inference.visualize(args.image, results, viz_path)
        
        # Save CSV row
        image_name = os.path.basename(args.image)
        rows = inference.detections_to_rows(image_name, results)
        inference.save_results_to_csv(csv_path, rows)
        
        # Print extracted data
        print("\n=== Extracted Check Data ===")
        for res in results:
            print(f"{res['field']}: {res['text']}")


if __name__ == '__main__':
    main()