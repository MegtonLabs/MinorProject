import ssl
import urllib.request
import warnings

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    FasterRCNN_ResNet50_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.rpn import AnchorGenerator

MODEL_VARIANTS = {
    "resnet50_fpn": {
        "builder": fasterrcnn_resnet50_fpn,
        "weights_enum": FasterRCNN_ResNet50_FPN_Weights,
        "default_trainable_layers": 3,
    },
    "resnet50_fpn_v2": {
        "builder": fasterrcnn_resnet50_fpn_v2,
        "weights_enum": FasterRCNN_ResNet50_FPN_V2_Weights,
        "default_trainable_layers": 3,
    },
    "mobilenet_v3_large_320_fpn": {
        "builder": fasterrcnn_mobilenet_v3_large_320_fpn,
        "weights_enum": FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
        "default_trainable_layers": 3,
    },
}

# Fix SSL certificate verification issue on macOS
ssl._create_default_https_context = ssl._create_unverified_context

class CheckLayoutDetector(nn.Module):
    """
    Faster R-CNN for Bank Check Field Detection
    Detects 6 field types: Date, Courtesy Amount, Legal Amount, 
    Account Number, Signature, Payee
    """
    
    def __init__(
        self,
        num_classes=6,
        model_variant: str = "resnet50_fpn",
        pretrained: bool = True,
        trainable_backbone_layers: int | None = None,
        use_custom_anchors: bool = True,
    ):
        """
        Args:
            num_classes: Number of field types + background
            backbone_name: 'resnet50' or 'resnet101'
            pretrained: Use ImageNet pretrained weights
        """
        super(CheckLayoutDetector, self).__init__()

        variant_key = model_variant.lower()
        if variant_key not in MODEL_VARIANTS:
            raise ValueError(
                f"Unsupported model variant '{model_variant}'. "
                f"Choose from: {', '.join(MODEL_VARIANTS.keys())}"
            )

        builder = MODEL_VARIANTS[variant_key]["builder"]
        weights_enum = MODEL_VARIANTS[variant_key]["weights_enum"]
        weights = weights_enum.DEFAULT if pretrained else None
        weights_backbone = None

        if weights is not None:
            default_categories = weights.meta.get("categories")
            if default_categories and len(default_categories) != num_classes:
                warnings.warn(
                    "Pretrained detection head expects "
                    f"{len(default_categories)} classes but num_classes={num_classes}. "
                    "Falling back to pretrained backbone only."
                )
                weights_backbone = getattr(weights, "backbone", None)
                weights = None

        if trainable_backbone_layers is None:
            trainable_backbone_layers = MODEL_VARIANTS[variant_key]["default_trainable_layers"]

        self.model_variant = variant_key
        self.model = builder(
            weights=weights,
            weights_backbone=weights_backbone,
            trainable_backbone_layers=trainable_backbone_layers,
            num_classes=num_classes,
        )

        if use_custom_anchors:
            self._configure_detection_heads()
        else:
            # Ensure detection head uses correct thresholds even without anchor override
            self._configure_head_thresholds()

        self.num_classes = num_classes
    
    def forward(self, images, targets=None):
        """
        Forward pass
        
        Args:
            images: List of tensors [C, H, W]
            targets: List of dicts with 'boxes' and 'labels'
            
        Returns:
            In training: dict of losses
            In inference: list of dicts with 'boxes', 'labels', 'scores'
        """
        return self.model(images, targets)
    
    def get_model(self):
        """Return the underlying FasterRCNN model"""
        return self.model

    def _configure_detection_heads(self):
        """Apply check-specific anchor generator and ROI align settings."""
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

        self.model.rpn.anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=aspect_ratios,
        )

        self.model.roi_heads.box_roi_pool = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=7,
            sampling_ratio=2,
        )

        # Match previous tuned parameters for dense check layouts
        self.model.rpn._pre_nms_top_n = dict(training=2000, testing=1000)
        self.model.rpn._post_nms_top_n = dict(training=2000, testing=1000)
        self.model.rpn.nms_thresh = 0.7
        self.model.rpn.score_thresh = 0.0
        self.model.roi_heads.score_thresh = 0.5
        self.model.roi_heads.nms_thresh = 0.5
        self.model.roi_heads.detections_per_img = 100

    def _configure_head_thresholds(self):
        """Ensure inference thresholds are set even when default anchors are used."""
        self.model.roi_heads.score_thresh = 0.5
        self.model.roi_heads.nms_thresh = 0.5
        self.model.roi_heads.detections_per_img = 100


class OCRProcessor:
    """
    OCR processing for detected check fields using Tesseract
    """
    
    def __init__(self, lang='eng', psm=7):
        """
        Args:
            lang: Language code
            psm: Page segmentation mode (7=single line)
        """
        self.lang = lang
        self.psm = psm
    
    def extract_text(self, image, box, padding=5):
        """
        Extract text from a detected region
        
        Args:
            image: Original check image (PIL or numpy)
            box: Tensor [x1, y1, x2, y2]
            padding: Pixel padding around box
            
        Returns:
            Extracted text string
        """
        try:
            import pytesseract
            import numpy as np
            from PIL import Image
            
            # Convert tensor to numpy
            if isinstance(box, torch.Tensor):
                box = box.cpu().numpy()
            
            # Add padding
            x1, y1, x2, y2 = box
            x1 = max(0, int(x1) - padding)
            y1 = max(0, int(y1) - padding)
            x2 = min(image.shape[1], int(x2) + padding)
            y2 = min(image.shape[0], int(y2) + padding)
            
            # Extract region
            region = image[y1:y2, x1:x2]
            
            # Preprocess for OCR
            # Convert to grayscale
            if len(region.shape) == 3:
                region_gray = np.dot(region[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                region_gray = region
            
            # Increase contrast
            region_gray = ((region_gray - region_gray.min()) / 
                          (region_gray.max() - region_gray.min()) * 255).astype(np.uint8)
            
            # Convert to PIL Image
            region_pil = Image.fromarray(region_gray)
            
            # Run OCR
            config = f'--psm {self.psm} --oem 3 -c tessedit_char_whitelist=0123456789/.-'
            text = pytesseract.image_to_string(region_pil, lang=self.lang, config=config)
            
            return text.strip()
            
        except ImportError:
            print("pytesseract not installed. Run: pip install pytesseract")
            return ""


def get_field_names():
    """Return dictionary mapping class IDs to field names"""
    return {
        1: "Date",
        2: "Courtesy_Amount",  # Numeric amount
        3: "Legal_Amount",     # Written amount
        4: "Account_Number",
        5: "Signature",
        6: "Payee"
    }


if __name__ == "__main__":
    # Test model instantiation
    model = CheckLayoutDetector(num_classes=7)  # 6 classes + background
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Print backbone info
    print(f"Backbone output channels: {model.backbone.out_channels}")