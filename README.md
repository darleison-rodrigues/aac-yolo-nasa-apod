# AAC Module for NASA APOD
Folder Structure:
```
/aac-yolo-nasa-apod/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── annotations/
├── src/
│   ├── modules/
│   │   └── aac_module.py
│   ├── models/
│   │   └── yolo_aac.py
│   ├── utils/
│   │   ├── apod_dataset.py
│   │   ├── transforms.py
│   │   ├── trainers.py
│   │   └── metrics.py
│   ├── config/
│   │   ├── default.yaml
│   │   └── yolo_variants.yaml
│   ├── scripts/
│   │   ├── preprocess_apod.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── visualize_attention.py
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── module_prototyping.ipynb
│   └── attention_analysis.ipynb
├── experiments/
│   ├── run_2025-07-15_yolov8n_baseline/
│   │   ├── weights/
│   │   ├── logs/
│   │   ├── attention_maps/
│   │   └── predictions/
│   ├── run_2025-07-16_yolov8n_aac_spatial/
│   │   ├── weights/
│   │   ├── logs/
│   │   ├── attention_maps/
│   │   └── predictions/
│   └── ...
├── .gitignore
├── README.md
├── requirements.txt
├── environment.yaml (for Conda)
```

Steps:

OS Installation (if needed): Boot a Linux distro (Ubuntu 22.04 LTS recommended) on your machine.

Driver and CUDA Installation: Install NVIDIA drivers, CUDA Toolkit (12.1 or newer compatible with your RTX 4070), and cuDNN. Follow official NVIDIA guides for your Linux distribution.

Conda Environment Setup:

Bash

conda create -n yolo_aac python=3.10  # Use a recent Python version
conda activate yolo_aac
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install ultralytics  # For YOLOv8 base
pip install opencv-python pillow matplotlib scikit-learn pandas
pip install wandb  # For experiment tracking
pip install albumentations  # For advanced data augmentation
pip install h5py  # For efficient data storage
Clone/Create Project Repository: Set up the yolo-aac-apod/ structure.

README.md: Start documenting the project, setup instructions, and goals.

requirements.txt / environment.yaml: Generate these files to ensure reproducibility.

# Data Acquisition and Preprocessing (Week 1)
Goal: Curate and prepare the NASA APOD dataset for training.

data/ folder content:

raw/: Stores downloaded raw images.

processed/: Stores images resized and possibly converted to a common format (e.g., PNG, JPG).

annotations/: Stores YOLO-format annotations (.txt files for each image).

Steps:

APOD Data Collection (scripts/preprocess_apod.py):

Automated Download: Write a script to programmatically download images from the NASA APOD archive. The API allows fetching by date. Focus on a wide date range (e.g., 2000-present).

Filtering: Filter out non-image entries (videos, text-only).

Initial Resizing: Resize images to a manageable size (e.g., 1280x720 or 640x480) and save to data/processed/.

Annotation Strategy: This is the most crucial and time-consuming part.

Manual Annotation (Initial Phase): For a significant subset (e.g., 500-1000 images), manually annotate celestial objects (galaxies, nebulae, star clusters, planets, comets, etc.) using a tool like LabelImg or Roboflow (if budget allows). Define clear object categories. This will be your initial training set.

Semi-Automated Annotation (Future Refinement): Once an initial YOLO model is trained, use it to pre-annotate new images, then manually correct.

Annotation Format: Ensure annotations are in YOLO format (class_id x_center y_center width height normalized).

Data Split: Divide the annotated dataset into train, val, and test sets (e.g., 80/10/10 split). Store paths to these splits in config/default.yaml.

utils/apod_dataset.py: Create a PyTorch Dataset class to load images and their corresponding YOLO annotations. Implement lazy loading to conserve RAM.

utils/transforms.py: Define initial data augmentation strategies using albumentations (e.g., A.Compose for random rotate, brightness/contrast, noise, flips).

# AAC Module Development (Week 2-3)
Goal: Implement and unit test the Astronomical Attention Convolution (AAC) module.

`src/modules/aac_module.py:`

```Python

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_att = torch.cat([avg_out, max_out], dim=1)
        x_att = self.conv1(x_att)
        return self.sigmoid(x_att)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        x_att = avg_out + max_out
        return self.sigmoid(x_att)

class AstronomicalAttentionConv(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel_size=7):
        super(AstronomicalAttentionConv, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x):
        # Apply channel attention
        channel_att_map = self.channel_attention(x)
        x_after_channel = x * channel_att_map

        # Apply spatial attention
        spatial_att_map = self.spatial_attention(x_after_channel)
        x_after_spatial = x_after_channel * spatial_att_map

        return x_after_spatial

# Example of how you might include domain-specific priors (conceptual)
# This would require more complex training or pre-defined weights
class AstronomicalAttentionConvWithPriors(AstronomicalAttentionConv):
    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel_size=7):
        super().__init__(in_channels, reduction_ratio, spatial_kernel_size)
        # Placeholder for astronomical-specific prior integration
        # e.g., learned weights for specific wavelength channels
        # or modules that enhance brightness gradients
        # This is where 'Astronomical-Specific Priors' and 'Spectral Channel Prioritization'
        # from your initial proposal would be implemented.
        # For a start, the basic CBAM-like structure is a good foundation.
        print("Note: Advanced astronomical priors need further integration/training strategy.")

if __name__ == '__main__':
    # Unit Test
    input_tensor = torch.randn(4, 256, 32, 32) # Batch, Channels, Height, Width

    aac_module = AstronomicalAttentionConv(in_channels=256)
    output_tensor = aac_module(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")

    # Memory Check (conceptual for a single module)
    # This is more relevant during full model training
    # You'd use torch.cuda.memory_summary() in actual training loop
    print(f"AAC module parameters: {sum(p.numel() for p in aac_module.parameters()) / 1e6:.2f} M")
```
Steps:

Initial AAC Module: Implement a foundational AAC module (e.g., inspired by CBAM or similar attention mechanisms, as shown above). Start simple to get it working.

Unit Testing (notebooks/module_prototyping.ipynb):

Create dummy tensors to test the forward pass of AstronomicalAttentionConv.

Verify input/output shapes.

Roughly estimate parameter count to ensure lightweight design.

Test different reduction_ratio and spatial_kernel_size values.

4. YOLO Integration (Week 3-4)
Goal: Integrate the AAC module into the YOLOv8 backbone.

`src/models/yolo_aac.py:`

```Python

from ultralytics import YOLO
import torch.nn as nn
from src.modules.aac_module import AstronomicalAttentionConv # Import your AAC module

class YOLOAAC(YOLO):
    def __init__(self, model='yolov8n.yaml', **kwargs):
        super().__init__(model, **kwargs)
        self.insert_aac_modules()

    def insert_aac_modules(self):
        # Iterate through the YOLO model's modules to find suitable insertion points
        # This typically involves identifying C2f modules or similar bottleneck layers
        # where feature maps are processed before being passed to the neck.
        # The exact insertion points depend on the YOLOv8 architecture definition.

        # A simplified example (requires specific knowledge of YOLOv8's internal graph)
        # You'll need to inspect the YOLOv8 model's structure (e.g., using model.model)
        # and identify the appropriate places to insert your module.
        # For instance, after certain C2f or Bottleneck layers.

        # This is a conceptual example of how you might replace/insert:
        # for i, module in enumerate(self.model.model):
        #     if isinstance(module, C2f): # Assuming C2f is a target layer
        #         # Replace C2f with a sequence of C2f + AAC
        #         # Or, insert AAC after C2f
        #         # You might need to modify the model graph directly
        #         # or subclass and override specific parts of YOLOv8's model definition.

        # For a practical start with Ultralytics YOLOv8:
        # The simplest way might be to extend existing blocks or modify the .yaml configuration.
        # Look into how custom modules are added in Ultralytics documentation.
        # You might modify the yolov8n.yaml directly to include 'AstronomicalAttentionConv'
        # as a new layer type, then parse it.

        # Let's assume for simplicity we can just add it after some detection layers
        # This is NOT how you'd typically do it but illustrates the concept:
        for i, module in enumerate(self.model.model):
            # Example: insert after a specific bottleneck block (needs careful identification)
            if isinstance(module, nn.Sequential): # This is a placeholder, be specific
                for j, sub_module in enumerate(module):
                    if isinstance(sub_module, nn.Conv2d) and sub_module.out_channels == 512: # Example
                        # Insert AAC after this convolution
                        # This would involve recreating the sequential module or using hook
                        print(f"Attempting to insert AAC after layer {i}.{j}")
                        # In reality, you'd modify the YAML or build graph dynamically
                        # self.model.model[i].add_module(f'aac_{j}', AstronomicalAttentionConv(sub_module.out_channels))
        print("AAC module insertion logic needs to be refined based on YOLOv8 model architecture.")

    def forward(self, x):
        return super().forward(x) # Call the original YOLO forward pass
```

Steps:

YOLOv8 Architecture Analysis: Dive into the ultralytics library source code or yolov8n.yaml to understand the YOLOv8 backbone (CSPDarknet). Identify strategic insertion points for your AAC module (e.g., after C2f blocks, before upsampling layers).

Module Insertion (src/models/yolo_aac.py):

Option 1 (Recommended for beginners): Modify the YOLOv8 configuration YAML file (yolov8n_aac.yaml). This is the cleanest way to integrate custom modules in Ultralytics. Define your AstronomicalAttentionConv as a new type and specify its parameters.

Option 2 (More advanced): Programmatically insert the module by traversing the YOLO model's nn.ModuleList and replacing/inserting layers. This requires more careful handling of tensor shapes.

Basic Model Check: Instantiate YOLOAAC and run a dummy forward pass to ensure the model loads without errors and the AAC module is correctly integrated.

5. Training Methodology (Week 4-7)
Goal: Train baseline and AAC-enhanced YOLO models and manage experiments.

src/scripts/train.py:

```Python

import os
from ultralytics import YOLO
import torch
import yaml
import wandb # For experiment tracking
from src.utils.apod_dataset import APODDataset # Your custom dataset
from src.utils.transforms import get_transforms # Your transforms
from src.models.yolo_aac import YOLOAAC # Your custom YOLO model

def train_model(config_path, model_name, run_name):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize Weights & Biases
    wandb.init(project="yolo-aac-apod", name=run_name, config=config)

    # Load YOLO model (either standard or your AAC-enhanced)
    if 'aac' in model_name: # Simple check for AAC variant
        model = YOLOAAC(model=config['model_cfg']) # Path to your yolov8n_aac.yaml
    else:
        model = YOLO(config['model_cfg']) # Path to yolov8n.yaml or yolov8s.yaml

    # Override default YOLOv8 training parameters from config
    # You might need to adjust these based on Ultralytics YOLO API
    model.train(
        data=config['data_cfg'], # Path to your dataset.yaml (Ultralytics format)
        epochs=config['epochs'],
        imgsz=config['imgsz'],
        batch=config['batch_size'],
        workers=config['num_workers'],
        optimizer=config['optimizer'],
        lr0=config['learning_rate'],
        amp=True, # Enable mixed precision training (PyTorch automatic mixed precision)
        # Additional Ultralytics training arguments:
        # cache=True, # Cache images for faster loading
        # patience=config['patience'], # Early stopping patience
        # resume=config['resume_path'], # For resuming training
        # val=True, # Validate during training
        # device=0 # Use GPU 0
    )

    wandb.finish()

if __name__ == '__main__':
    # Configuration for baseline YOLOv8n
    config_baseline = {
        'model_cfg': 'yolov8n.yaml', # Or path to yolov8n.pt if starting from pretrained weights
        'data_cfg': 'config/apod_dataset.yaml', # Ultralytics format dataset config
        'epochs': 100,
        'imgsz': 640,
        'batch_size': 16,
        'num_workers': 8,
        'optimizer': 'AdamW',
        'learning_rate': 0.001,
        'patience': 50,
        'resume_path': None
    }
    # Configuration for YOLOv8n with AAC
    config_aac = {
        'model_cfg': 'config/yolov8n_aac.yaml', # Your custom YAML with AAC module
        'data_cfg': 'config/apod_dataset.yaml',
        'epochs': 100,
        'imgsz': 640,
        'batch_size': 16, # Adjust based on VRAM
        'num_workers': 8,
        'optimizer': 'AdamW',
        'learning_rate': 0.001,
        'patience': 50,
        'resume_path': None
    }

    # Save configs for reproducibility
    with open('config/baseline_train.yaml', 'w') as f:
        yaml.dump(config_baseline, f)
    with open('config/aac_train.yaml', 'w') as f:
        yaml.dump(config_aac, f)

    # Run baseline training
    print("Starting Baseline YOLOv8n Training...")
    train_model('config/baseline_train.yaml', 'yolov8n_baseline', 'run_2025-07-15_yolov8n_baseline')

    # Run AAC-enhanced training
    print("Starting YOLOv8n-AAC Training...")
    train_model('config/aac_train.yaml', 'yolov8n_aac', 'run_2025-07-16_yolov8n_aac')
config/apod_dataset.yaml (Ultralytics format):
```
```YAML

path: ../data/  # Dataset root directory
train: processed/train_images.txt # Path to file listing training image paths
val: processed/val_images.txt   # Path to file listing validation image paths
test: processed/test_images.txt # Path to file listing test image paths
```

# Classes
nc: 5  # number of classes
names: ['galaxy', 'nebula', 'star_cluster', 'planet', 'comet'] # Your defined classes
Steps:

Configuration Files (config/):

Create yolov8n.yaml (copy from Ultralytics if needed).

Create yolov8n_aac.yaml by modifying yolov8n.yaml to include your AAC module at appropriate locations.

Create apod_dataset.yaml with paths to your training, validation, and test image lists and class definitions.

Training Script (scripts/train.py):

Utilize ultralytics's YOLO.train() method for streamlined training.

Implement Mixed Precision Training (amp=True) to optimize VRAM usage on your RTX 4070.

Configure num_workers for your DataLoader (e.g., 6-8).

Integrate Weights & Biases (W&B) for experiment tracking:

Log training/validation metrics (mAP, loss).

Log learning rate, batch size.

Crucially: Log sample predictions with bounding boxes and attention maps (if feasible to extract from the AAC module).

Training Regimen:

Phase A: Baseline Training: Train a standard YOLOv8n model on your annotated APOD dataset. This establishes your performance benchmark.

Phase B: AAC-Enhanced Training: Train your YOLOAAC model.

Progressive Resizing: Start with imgsz=640 and potentially increase to 1280 in later epochs or for final evaluation if memory allows.

Gradient Accumulation: If batch_size=16 is too large for your VRAM, use accumulate=N in the train call to simulate a larger batch size.

Pre-training: If available, consider pre-training on a larger general astronomical dataset (e.g., from Kaggle or public observatories) before fine-tuning on APOD.

6. Evaluation and Analysis (Week 7-8)
Goal: Quantify the effectiveness of the AAC module and analyze its impact.

`src/scripts/evaluate.py:`

```Python

import os
from ultralytics import YOLO
import wandb
import yaml
from src.models.yolo_aac import YOLOAAC # Your custom YOLO model

def evaluate_model(config_path, model_weights_path, run_name):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    wandb.init(project="yolo-aac-apod", name=f"evaluation_{run_name}", config=config)

    # Load the trained model
    if 'aac' in run_name:
        model = YOLOAAC(model=config['model_cfg'])
    else:
        model = YOLO(config['model_cfg'])

    # Load specific weights for evaluation
    model = model.load(model_weights_path)

    # Evaluate on the test set
    metrics = model.val(
        data=config['data_cfg'],
        imgsz=config['imgsz'],
        batch=config['batch_size'],
        workers=config['num_workers'],
        # Additional Ultralytics validation arguments
    )

    # Log metrics to W&B
    wandb.log({
        "metrics/mAP50-95": metrics.box.map,
        "metrics/mAP50": metrics.box.map50,
        "metrics/mAP75": metrics.box.map75,
        "metrics/precision": metrics.box.p,
        "metrics/recall": metrics.box.r,
        # Log class-specific metrics if available
    })

    wandb.finish()

if __name__ == '__main__':
    # Configuration paths for evaluation
    config_baseline_path = 'config/baseline_train.yaml'
    config_aac_path = 'config/aac_train.yaml'

    # Paths to the best weights saved during training (from experiments/run_.../weights/best.pt)
    baseline_weights = 'experiments/run_2025-07-15_yolov8n_baseline/weights/best.pt'
    aac_weights = 'experiments/run_2025-07-16_yolov8n_aac_spatial/weights/best.pt'

    print("Evaluating Baseline YOLOv8n...")
    evaluate_model(config_baseline_path, baseline_weights, 'yolov8n_baseline')

    print("Evaluating YOLOv8n-AAC...")
    evaluate_model(config_aac_path, aac_weights, 'yolov8n_aac_spatial')
src/scripts/visualize_attention.py (and notebooks/attention_analysis.ipynb):
```


```Python

import torch
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from src.models.yolo_aac import YOLOAAC # Your custom YOLO model
from src.modules.aac_module import AstronomicalAttentionConv

def visualize_aac_attention(image_path, model_weights_path, output_dir='attention_visuals'):
    os.makedirs(output_dir, exist_ok=True)

    # Load your AAC-enhanced YOLO model
    model = YOLOAAC(model='config/yolov8n_aac.yaml') # Load your AAC config
    model.load(model_weights_path)
    model.eval() # Set to evaluation mode

    # Load and preprocess image
    img_orig = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = F.interpolate(img_tensor, size=(640, 640), mode='bilinear', align_corners=False) # Resize for model input

    # Create hooks to capture feature maps and attention maps
    attention_maps = {}
    def hook_fn(module, input, output, name):
        attention_maps[name] = output.detach()

    # Register hooks on your AAC modules (you'll need to find their names in the model)
    # This requires inspecting the model's named_modules()
    for name, module in model.named_modules():
        if isinstance(module, AstronomicalAttentionConv):
            # Register hook on the output of the AAC module
            module.register_forward_hook(lambda m, i, o, name=name: hook_fn(m, i, o, name))

    with torch.no_grad():
        results = model(img_tensor) # Run inference to activate hooks

    # Process and visualize attention maps
    for name, att_map in attention_maps.items():
        # Attention map shape: [1, 1, H, W] for spatial, [1, C, 1, 1] for channel
        # You'll need to decide which part of AAC to visualize (spatial or combined output)
        if att_map.dim() == 4 and att_map.shape[1] == 1: # Assuming spatial attention map
            att_map_np = att_map.squeeze().cpu().numpy()
            att_map_resized = cv2.resize(att_map_np, (img_orig.shape[1], img_orig.shape[0]))

            # Overlay attention map on original image
            cmap = plt.get_cmap('jet')
            heatmap = cmap(att_map_resized)[:, :, :3] # Get RGB from colormap
            heatmap = (heatmap * 255).astype(np.uint8)

            result_img = cv2.addWeighted(img_orig, 0.7, heatmap, 0.3, 0) # Adjust alpha as needed

            cv2.imwrite(os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_{name}_attention.jpg"), result_img)
            print(f"Saved attention map for {name} to {output_dir}")

if __name__ == '__main__':
    # Example usage:
    sample_image = 'data/processed/sample_apod_image.jpg' # Replace with an actual image from your dataset
    trained_aac_weights = 'experiments/run_2025-07-16_yolov8n_aac_spatial/weights/best.pt'
    visualize_aac_attention(sample_image, trained_aac_weights)
```

Steps:

Evaluation Script (scripts/evaluate.py):

Load the best weights from your trained baseline and AAC-enhanced models.

Use model.val() from Ultralytics to compute standard detection metrics (mAP, precision, recall) on the test set.

Log all metrics to W&B for easy comparison.

Qualitative Analysis (scripts/visualize_attention.py, notebooks/attention_analysis.ipynb):

Implement a script to visualize the attention maps generated by your AAC module. This requires adding hooks to your AstronomicalAttentionConv module to extract its internal attention weights.

Overlay these attention maps on the original APOD images to visually confirm if the module is focusing on relevant astronomical features.

Compare attention maps between the baseline and AAC-enhanced models for the same images.

## Ablation Studies:

AAC Components: Train models with only spatial attention, only channel attention, and both, to understand their individual contributions.

Placement: Experiment with placing the AAC module at different depths in the YOLO backbone (e.g., early layers vs. later layers).

Hyperparameters: Vary reduction_ratio and spatial_kernel_size in your AAC module.

## Reporting:

Summarize quantitative results (mAP improvements, false positive reduction).

Present qualitative examples showing enhanced feature focus.

Discuss computational overhead (parameter count, inference time) of your AAC module.

## Timeline Summary
- Week 1: Project Setup, Data Acquisition, Initial Data Annotation (start of manual annotation, will continue throughout).

- Week 2-3: AAC Module Development, Unit Testing, Initial YOLO Integration.

- Week 4-7: Training Baseline, Training AAC-Enhanced Models, Experiment Management with W&B. (This phase will involve iterations of training and fine-tuning).

- Week 7-8: Evaluation, Ablation Studies, Qualitative Analysis, Documentation, and Preparation for Presentation/Paper.****
