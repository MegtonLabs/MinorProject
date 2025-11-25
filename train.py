import os
import tempfile
import time
import warnings
from contextlib import nullcontext

import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.optim as optim
from PIL import Image
from dotenv import load_dotenv
from pycocotools.coco import COCO
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from tqdm import tqdm

from check_detection_model import CheckLayoutDetector, get_field_names

# Load environment variables from .env file
load_dotenv()

# Filter warnings
warnings.filterwarnings("ignore")

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("medium")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True


def parse_bool(value: str | None, default: bool = False) -> bool:
    """Robust bool parsing for env values."""
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def should_enable_amp(flag: str, device: torch.device) -> bool:
    """
    Determine AMP usage.
    'auto' enables AMP when CUDA is available.
    """
    if flag.lower() == "auto":
        return device.type == "cuda"
    return parse_bool(flag, default=False)

class CheckDataset(torch.utils.data.Dataset):
    """
    Custom Dataset for Bank Check Images with COCO annotations
    """
    
    def __init__(self, root, annFile, transforms=None, category_mapping=None):
        """
        Args:
            root: Image directory path
            annFile: COCO format annotation file
            transforms: Optional transforms
            category_mapping: Optional dict mapping COCO category IDs to labels (for consistency)
        """
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
        
        # Get class names from COCO categories
        self.id_to_name = {cat['id']: cat['name'] 
                          for cat in self.coco.loadCats(self.coco.getCatIds())}
        self.name_to_id = {v: k for k, v in self.id_to_name.items()}
        
        # Create mapping from COCO category IDs to 1-indexed labels (0 is background)
        # Faster R-CNN expects labels starting from 1
        # If category_mapping is provided, use it (for validation set to match training)
        if category_mapping is not None:
            self.coco_id_to_label = category_mapping
        else:
            sorted_cat_ids = sorted(self.coco.getCatIds())
            self.coco_id_to_label = {coco_id: idx + 1 for idx, coco_id in enumerate(sorted_cat_ids)}
        
        print(f"✅ Loaded {len(self.ids)} images from {root}")
        print(f"✅ Loaded {len(self.coco.getCatIds())} classes: {list(self.name_to_id.keys())}")
        print(f"✅ Category ID mapping: {self.coco_id_to_label}")
    
    def __getitem__(self, index):
        """
        Args:
            index: int
            
        Returns:
            image: Tensor [C, H, W]
            target: dict with 'boxes', 'labels', 'image_id'
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # Load image
        img_info = coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        img_path = os.path.join(self.root, path)
        
        # Verify image exists
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        img = Image.open(img_path).convert('RGB')
        
        # Convert to tensor
        img_tensor = F.to_tensor(img)
        
        # Create target dictionary
        boxes = []
        labels = []
        
        for ann in anns:
            # Map COCO category ID to 1-indexed label (0 is background)
            coco_cat_id = ann['category_id']
            if coco_cat_id not in self.coco_id_to_label:
                # Skip annotations with unknown category IDs (e.g., from val set with extra classes)
                continue
                
            # COCO bbox format: [x, y, width, height]
            x, y, w, h = ann['bbox']
            # Convert to [x1, y1, x2, y2]
            boxes.append([x, y, x + w, y + h])
            labels.append(self.coco_id_to_label[coco_cat_id])
        
        # Handle empty annotations (all filtered out)
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if len(boxes) > 0 else torch.zeros((0,), dtype=torch.float32),
            'iscrowd': torch.zeros((len(labels),), dtype=torch.int64)
        }
        
        if self.transforms:
            img_tensor, target = self.transforms(img_tensor, target)
            
        return img_tensor, target
    
    def __len__(self):
        return len(self.ids)


class CheckTransforms:
    """Training and validation transforms"""
    
    def __init__(self, is_train=True):
        self.is_train = is_train
    
    def __call__(self, image, target):
        if self.is_train:
            # Random horizontal flip (checks are symmetric)
            if torch.rand(1) < 0.5:
                image = torch.flip(image, [2])  # Flip width dimension
                boxes = target['boxes']
                boxes[:, 0], boxes[:, 2] = image.shape[2] - boxes[:, 2], image.shape[2] - boxes[:, 0]
                target['boxes'] = boxes
            
            # Random brightness/contrast adjustment (simulates scanning variations)
            image = F.adjust_brightness(image, 0.8 + torch.rand(1) * 0.4)
            image = F.adjust_contrast(image, 0.8 + torch.rand(1) * 0.4)
        
        # Normalize (ImageNet stats - standard for transfer learning)
        image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        return image, target


def collate_fn(batch):
    """Custom batching for variable number of objects"""
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler=None, grad_clip=0.0, print_freq=10):
    """Train for one epoch with progress tracking"""
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    header = f'Epoch: [{epoch}]'
    
    # Learning rate warmup for first epoch
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )
    
    # Progress bar
    pbar = tqdm(data_loader, desc=f"Training Epoch {epoch+1}")
    
    amp_enabled = scaler is not None and scaler.is_enabled() if scaler is not None else False
    
    for batch_idx, (images, targets) in enumerate(pbar):
        # Move data to device
        images = [image.to(device, non_blocking=True) for image in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
        
        # Forward pass with optional mixed precision
        if amp_enabled and device.type == "cuda":
            autocast_ctx = autocast(dtype=torch.float16)
        else:
            autocast_ctx = nullcontext()
        with autocast_ctx:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        
        if not torch.isfinite(losses):
            print(f"⚠️  WARNING: non-finite loss, skipping batch: {loss_dict}")
            continue
        
        optimizer.zero_grad()
        
        # Backward pass with optional mixed precision
        if amp_enabled and scaler is not None:
            scaler.scale(losses).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        # Update metrics
        metric_logger.update(loss=losses.item(), **{k: v.item() for k, v in loss_dict.items()})
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.item():.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    
    print(f"\n📊 Epoch {epoch} Summary:")
    for k, meter in metric_logger.meters.items():
        print(f"  {k}: {meter.avg:.4f}")
    
    return metric_logger


@torch.inference_mode()
def evaluate(model, data_loader, device, use_amp=False):
    """Evaluate model and compute COCO mAP"""
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Validation:'
    
    coco = data_loader.dataset.coco
    coco_evaluator = CocoEvaluator(coco)
    
    amp_enabled = use_amp and device.type == "cuda"
    
    for images, targets in tqdm(data_loader, desc="Validating"):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        model_time = time.time()
        if amp_enabled and device.type == "cuda":
            autocast_ctx = autocast(dtype=torch.float16)
        else:
            autocast_ctx = nullcontext()
        with autocast_ctx:
            outputs = model(images)
        model_time = time.time() - model_time
        
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
    
    # Gather stats
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    
    return coco_evaluator


class MetricLogger:
    """Helper for logging metrics"""
    def __init__(self, delimiter="\t"):
        self.meters = {}
        self.delimiter = delimiter
    
    def synchronize_between_processes(self):
        """Placeholder for multi-GPU support; no-op in single-process runs."""
        return
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if isinstance(v, (float, int)):
                if k not in self.meters:
                    self.meters[k] = AverageMeter()
                self.meters[k].update(v)
    
    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        return object.__getattribute__(self, attr)
    
    def __str__(self):
        return self.delimiter.join([f"{k}: {str(v)}" for k, v in self.meters.items()])


class AverageMeter:
    """Computes and stores the average"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f'{self.avg:.4f}'


class CocoEvaluator:
    """COCO AP evaluation"""
    def __init__(self, coco_gt):
        self.coco_gt = coco_gt
        self.coco_dt = None
        self.results = []
        self.stats = None
    
    def update(self, res):
        self.results.append(res)
    
    def synchronize_between_processes(self):
        # For single GPU training
        pass
    
    def accumulate(self):
        if not self.results:
            print("⚠️  No detections to evaluate!")
            return
        
        # Combine results
        all_results = {}
        for res in self.results:
            all_results.update(res)
        
        # Convert to COCO format
        dt = []
        for img_id, output in all_results.items():
            boxes = output['boxes'].cpu().numpy()
            labels = output['labels'].cpu().numpy()
            scores = output['scores'].cpu().numpy()
            
            for i in range(len(boxes)):
                dt.append({
                    'image_id': img_id,
                    'category_id': int(labels[i]),
                    'bbox': [
                        float(boxes[i][0]),
                        float(boxes[i][1]),
                        float(boxes[i][2] - boxes[i][0]),
                        float(boxes[i][3] - boxes[i][1])
                    ],
                    'score': float(scores[i])
                })
        
        # Save temporary file (cross-platform)
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, 'coco_dt.json')
        with open(temp_path, 'w') as f:
            json.dump(dt, f)
        
        self.coco_dt = self.coco_gt.loadRes(temp_path)
    
    def summarize(self):
        if self.coco_dt is None:
            print("⚠️  No detections to summarize!")
            return
        
        from pycocotools.cocoeval import COCOeval
        
        coco_eval = COCOeval(self.coco_gt, self.coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        self.stats = coco_eval.stats
        
        print(f"\n" + "="*50)
        print(f"📈 VALIDATION RESULTS")
        print(f"="*50)
        print(f"AP (IoU=0.50:0.95): {self.stats[0]:.3f} (Primary metric)")
        print(f"AP (IoU=0.50):       {self.stats[1]:.3f}")
        print(f"AP (IoU=0.75):       {self.stats[2]:.3f}")
        print(f"AP (Small):          {self.stats[3]:.3f}")
        print(f"AP (Medium):         {self.stats[4]:.3f}")
        print(f"AP (Large):          {self.stats[5]:.3f}")
        print(f"AR (Max=1):          {self.stats[6]:.3f}")
        print(f"AR (Max=10):         {self.stats[7]:.3f}")
        print(f"AR (Max=100):        {self.stats[8]:.3f}")
        print(f"AR (Small):          {self.stats[9]:.3f}")
        print(f"AR (Medium):         {self.stats[10]:.3f}")
        print(f"AR (Large):          {self.stats[11]:.3f}")
        print(f"="*50 + "\n")


def setup_data_loaders(
    data_root,
    batch_size=2,
    num_workers=2,
    prefetch_factor=2,
    pin_memory=False,
    persistent_workers=True,
):
    """
    Setup train and validation data loaders with automatic path detection
    """
    train_transform = CheckTransforms(is_train=True)
    val_transform = CheckTransforms(is_train=False)
    
    # CORRECTED PATHS - directly using data_root structure
    train_root = os.path.join(data_root, 'train')
    train_ann = os.path.join(data_root, 'annotations', 'instances_train.json')
    val_root = os.path.join(data_root, 'val')
    val_ann = os.path.join(data_root, 'annotations', 'instances_val.json')
    
    # Verify paths exist
    if not os.path.exists(train_ann):
        raise FileNotFoundError(f"""
        ❌ Train annotations not found: {train_ann}
        Please ensure your structure is:
        ssbi_dataset/
        ├── annotations/
        │   ├── instances_train.json
        │   └── instances_val.json
        ├── train/
        └── val/
        """)
    
    if not os.path.exists(val_ann):
        raise FileNotFoundError(f"❌ Val annotations not found: {val_ann}")
    
    if not os.path.exists(train_root):
        raise FileNotFoundError(f"❌ Train images folder not found: {train_root}")
    
    if not os.path.exists(val_root):
        raise FileNotFoundError(f"❌ Val images folder not found: {val_root}")
    
    # Create datasets - train first to get category mapping
    print(f"📂 Loading training data from: {train_root}")
    dataset_train = CheckDataset(train_root, train_ann, train_transform)
    
    # Use train dataset's category mapping for validation to ensure consistency
    print(f"📂 Loading validation data from: {val_root}")
    dataset_val = CheckDataset(val_root, val_ann, val_transform, 
                               category_mapping=dataset_train.coco_id_to_label)
    
    # DataLoaders - reduced num_workers for Mac compatibility
    def _loader_kwargs(worker_count: int):
        opts = {}
        if worker_count > 0:
            opts["prefetch_factor"] = prefetch_factor
            opts["persistent_workers"] = persistent_workers
        return opts
    
    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        **_loader_kwargs(num_workers),
    )
    
    val_workers = min(num_workers, 2)
    val_loader = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=val_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        **_loader_kwargs(val_workers),
    )
    
    print(f"✅ Train samples: {len(dataset_train)}")
    print(f"✅ Val samples: {len(dataset_val)}")
    
    return train_loader, val_loader


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path, metadata=None):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    payload = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    if metadata:
        payload['config'] = metadata
    torch.save(payload, checkpoint_path)
    print(f"💾 Checkpoint saved: {checkpoint_path}")


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load model checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"⚠️  No checkpoint found at {checkpoint_path}")
        return 0, float('inf'), {}
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    config = checkpoint.get('config', {})
    print(f"📂 Checkpoint loaded: {checkpoint_path}")
    print(f"   - Epoch: {epoch}")
    print(f"   - Loss: {loss:.4f}")
    return epoch, loss, config


def main():
    """Main training function with configuration"""
    # Configuration from environment variables with defaults
    DATA_ROOT = os.getenv('DATA_ROOT', os.path.join(os.path.dirname(__file__), 'ssbi_dataset'))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', '2'))  # Adjust based on GPU memory
    NUM_EPOCHS = int(os.getenv('NUM_EPOCHS', '50'))
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', '0.001'))
    MOMENTUM = float(os.getenv('MOMENTUM', '0.9'))
    WEIGHT_DECAY = float(os.getenv('WEIGHT_DECAY', '0.0005'))
    CHECKPOINT_DIR = os.getenv('CHECKPOINT_DIR', 'checkpoints')
    NUM_WORKERS = int(os.getenv('NUM_WORKERS', '4'))  # DataLoader workers
    DEVICE_ID = int(os.getenv('CUDA_DEVICE', '0'))  # GPU device ID
    PREFETCH_FACTOR = int(os.getenv('PREFETCH_FACTOR', '2'))
    PERSISTENT_WORKERS = parse_bool(os.getenv('PERSISTENT_WORKERS', 'true'), True)
    MODEL_VARIANT = os.getenv('MODEL_VARIANT', 'resnet50_fpn')
    TRAINABLE_BACKBONE_LAYERS = int(os.getenv('TRAINABLE_BACKBONE_LAYERS', '-1'))
    USE_CUSTOM_ANCHORS = parse_bool(os.getenv('USE_CUSTOM_ANCHORS', 'true'), True)
    AMP_FLAG = os.getenv('USE_AMP', 'auto')
    VAL_INTERVAL = int(os.getenv('VAL_INTERVAL', '5'))
    GRAD_CLIP = float(os.getenv('GRAD_CLIP', '0'))
    USE_TORCH_COMPILE = parse_bool(os.getenv('USE_TORCH_COMPILE', 'false'), False)
    
    # Enhanced CUDA detection and setup
    print(f"\n" + "="*60)
    print(f"🔍 CUDA DETECTION")
    print(f"="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        
        # Use specified device ID or default to 0
        if DEVICE_ID >= torch.cuda.device_count():
            print(f"⚠️  Device ID {DEVICE_ID} not available, using GPU 0")
            DEVICE_ID = 0
        
        DEVICE = torch.device(f'cuda:{DEVICE_ID}')
        print(f"✅ Using GPU {DEVICE_ID}: {torch.cuda.get_device_name(DEVICE_ID)}")
        
        # Set default tensor type and enable optimizations
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
    else:
        print(f"⚠️  CUDA not available, falling back to CPU")
        print(f"   This may be due to:")
        print(f"   - PyTorch not compiled with CUDA support")
        print(f"   - CUDA drivers not properly installed")
        print(f"   - GPU not detected")
        print(f"   💡 To install CUDA-enabled PyTorch, visit: https://pytorch.org/get-started/locally/")
        DEVICE = torch.device('cpu')
    
    pin_memory_env = os.getenv('PIN_MEMORY', 'auto').lower()
    if pin_memory_env == 'auto':
        PIN_MEMORY = DEVICE.type == 'cuda'
    else:
        PIN_MEMORY = parse_bool(pin_memory_env, default=DEVICE.type == 'cuda')
    
    print(f"="*60)
    print(f"🏁 BANK CHECK DETECTION - TRAINING START")
    print(f"="*60)
    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATA_ROOT}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Model Variant: {MODEL_VARIANT}")
    print(f"Use Custom Anchors: {USE_CUSTOM_ANCHORS}")
    print("="*60 + "\n")
    
    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # DataLoaders
    try:
        train_loader, val_loader = setup_data_loaders(
            DATA_ROOT,
            BATCH_SIZE,
            NUM_WORKERS,
            prefetch_factor=PREFETCH_FACTOR,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS,
        )
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        return
    
    # Model - use number of classes from training set
    print(f"\n🤖 Initializing model...")
    num_classes = len(train_loader.dataset.coco_id_to_label) + 1  # +1 for background
    trainable_layers = None if TRAINABLE_BACKBONE_LAYERS < 0 else TRAINABLE_BACKBONE_LAYERS
    model = CheckLayoutDetector(
        num_classes=num_classes,
        model_variant=MODEL_VARIANT,
        trainable_backbone_layers=trainable_layers,
        use_custom_anchors=USE_CUSTOM_ANCHORS,
    )
    
    # Move model to device
    model = model.to(DEVICE)
    
    if USE_TORCH_COMPILE and hasattr(torch, "compile"):
        print("✅ torch.compile enabled")
        model = torch.compile(model)
    
    # Enable mixed precision training if configured
    use_amp = should_enable_amp(AMP_FLAG, DEVICE)
    scaler = GradScaler(enabled=use_amp) if use_amp else None
    if use_amp:
        print(f"✅ Mixed precision training enabled")
    
    print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"✅ Model configured for {num_classes} classes ({num_classes-1} object classes + background)")
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    checkpoint_metadata = {
        'num_classes': num_classes,
        'model_variant': MODEL_VARIANT,
        'trainable_backbone_layers': trainable_layers,
        'use_custom_anchors': USE_CUSTOM_ANCHORS,
    }
    
    # Resume from checkpoint if exists
    start_epoch = 0
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'latest_checkpoint.pth')
    if os.path.exists(checkpoint_path):
        ckpt_epoch, _, ckpt_config = load_checkpoint(model, optimizer, checkpoint_path, DEVICE)
        start_epoch = ckpt_epoch + 1
        if ckpt_config:
            checkpoint_metadata.update({k: v for k, v in ckpt_config.items() if k not in checkpoint_metadata})
            ckpt_variant = ckpt_config.get('model_variant')
            if ckpt_variant and ckpt_variant != MODEL_VARIANT:
                print(f"⚠️  Warning: checkpoint variant '{ckpt_variant}' differs from configured '{MODEL_VARIANT}'.")
    
    # Training loop
    print(f"\n📚 Starting training from epoch {start_epoch}...")
    best_map = 0.0
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"🔄 EPOCH {epoch+1}/{NUM_EPOCHS}")
        print(f"{'='*60}")
        
        # Train
        train_metrics = train_one_epoch(
            model,
            optimizer,
            train_loader,
            DEVICE,
            epoch,
            scaler=scaler,
            grad_clip=GRAD_CLIP,
        )
        
        # Update LR
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"📊 Current LR: {current_lr:.6f}")
        
        # Validate every 5 epochs
        if VAL_INTERVAL > 0 and (epoch + 1) % VAL_INTERVAL == 0:
            print(f"\n🔍 Running validation...")
            evaluator = evaluate(model, val_loader, DEVICE, use_amp=use_amp)
            
            if hasattr(evaluator, 'stats') and evaluator.stats is not None:
                current_map = evaluator.stats[0]
                if current_map > best_map:
                    best_map = current_map
                    save_checkpoint(
                        model,
                        optimizer,
                        epoch,
                        current_map,
                        os.path.join(CHECKPOINT_DIR, 'best_model.pth'),
                        metadata=checkpoint_metadata,
                    )
                    print(f"🏆 New best model saved with mAP: {best_map:.3f}")
        
        # Save latest checkpoint
        loss_avg = train_metrics.loss.avg if hasattr(train_metrics, 'loss') else 0.0
        save_checkpoint(
            model,
            optimizer,
            epoch,
            loss_avg,
            checkpoint_path,
            metadata=checkpoint_metadata,
        )
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"\n" + "="*60)
    print(f"✅ TRAINING COMPLETED!")
    print(f"Best mAP: {best_map:.3f}")
    print(f"="*60)


if __name__ == '__main__':
    main()