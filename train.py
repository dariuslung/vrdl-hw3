import os
import argparse
import numpy as np
import torch
import skimage.io as sio
from skimage.measure import label
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from collections import Counter
from tqdm import tqdm # Optional, for a progress bar

# --- NEW: Import v2 transforms and tv_tensors ---
from torchvision.transforms import v2 as T
from torchvision import tv_tensors
from torchvision.models.detection.anchor_utils import AnchorGenerator


def check_class_imbalance(dataset):
    print("Scanning dataset for class frequencies...")
    instance_counts = Counter()
    image_counts = Counter()
    
    # We use the dataset without transforms here so we don't accidentally count
    # augmented versions or cropped-out boxes.
    for idx in tqdm(range(len(dataset)), desc="Checking class imbalance"):
        _, target = dataset[idx]
        labels = target["labels"].tolist()
        
        if len(labels) == 0:
            continue
            
        # Count total instances
        instance_counts.update(labels)
        
        # Count unique classes present in this specific image
        unique_classes_in_image = set(labels)
        image_counts.update(unique_classes_in_image)
        
    print("\n--- Imbalance Report ---")
    print("Total instances per class:")
    for cls, count in sorted(instance_counts.items()):
        print(f"  Class {cls}: {count} instances")
        
    print("\nTotal images containing at least one instance of the class:")
    for cls, count in sorted(image_counts.items()):
        print(f"  Class {cls}: {count} images")


# --- 1. Transforms Definition ---
def get_transforms(train=True):
    transforms = []
    if train:
        # Crop to 800x800. Pad images that are smaller than 800.
        transforms.append(T.RandomCrop(size=(800, 800), pad_if_needed=True))
        # Add random flips to improve AP50
        transforms.append(T.RandomHorizontalFlip(p=0.5))
        transforms.append(T.RandomVerticalFlip(p=0.5))
        # Lighting & Stain Simulation (Crucial for medical images)
        transforms.append(T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2))
    
    # Ensure boxes that are cut out entirely by the crop are removed
    transforms.append(T.SanitizeBoundingBoxes())
    
    return T.Compose(transforms)

# --- 2. Dataset Definition ---
class MedicalCellDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        
        self.sample_dirs = [
            os.path.join(root_dir, d) for d in os.listdir(root_dir) 
            if os.path.isdir(os.path.join(root_dir, d))
        ]
        
        self.class_mapping = {
            "class1.tif": 1,
            "class2.tif": 2,
            "class3.tif": 3,
            "class4.tif": 4
        }

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        
        img_path = os.path.join(sample_dir, "image.tif")
        img_array = sio.imread(img_path)
        
        if len(img_array.shape) == 2:
            img_array = np.stack((img_array,)*3, axis=-1)
        elif img_array.shape[-1] > 3:
            img_array = img_array[:, :, :3]
            
        img_tensor = torch.as_tensor(img_array, dtype=torch.float32).permute(2, 0, 1) / 255.0

        boxes = []
        masks_list = []
        labels = []

        # Iterate through possible class files in the folder
        for class_filename, class_id in self.class_mapping.items():
            mask_path = os.path.join(sample_dir, class_filename)
            
            if not os.path.exists(mask_path):
                continue
                
            mask_array = sio.imread(mask_path)
            
            # Ensure 2D mask
            if len(mask_array.shape) > 2:
                mask_array = mask_array[:, :, 0]

            # --- THE REAL FIX: Bulletproof Instance Separation ---
            unique_vals = np.unique(mask_array)
            unique_vals = unique_vals[unique_vals > 0] # Exclude background
            
            if len(unique_vals) == 0:
                continue # Completely empty mask
                
            if len(unique_vals) == 1:
                # It's a binary mask (e.g., all cells are 255, or all cells are 1)
                # We MUST use connected components to separate individual cells
                instance_labeled_mask = label(mask_array > 0)
            else:
                # It's already an instance mask (cells are explicitly numbered 1, 2, 3...)
                instance_labeled_mask = mask_array

            # Get the new clean instance IDs
            obj_ids = np.unique(instance_labeled_mask)
            obj_ids = obj_ids[obj_ids != 0] # Remove background (0)

            for obj_id in obj_ids:
                instance_mask = (instance_labeled_mask == obj_id)
                pos = np.where(instance_mask)
                
                # Skip empty masks
                if len(pos[0]) == 0:
                    continue
                    
                xmin, xmax = np.min(pos[1]), np.max(pos[1])
                ymin, ymax = np.min(pos[0]), np.max(pos[0])
                
                # Validate bounding box AND ensure it doesn't cover the entire image
                box_area = (xmax - xmin) * (ymax - ymin)
                image_area = mask_array.shape[0] * mask_array.shape[1]
                
                if xmax > xmin and ymax > ymin and (box_area < 0.8 * image_area):
                    boxes.append([xmin, ymin, xmax, ymax])
                    masks_list.append(instance_mask)
                    labels.append(class_id)

        # Wrap inputs in tv_tensors so v2 transforms apply correctly
        img_tensor = tv_tensors.Image(img_tensor)

        if len(boxes) == 0:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            masks = torch.empty((0, img_tensor.shape[1], img_tensor.shape[2]), dtype=torch.uint8)
            labels = torch.empty((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            masks = torch.as_tensor(np.array(masks_list), dtype=torch.uint8)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        boxes = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=img_tensor.shape[-2:])
        masks = tv_tensors.Mask(masks)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx])
        }

        # Apply spatial transforms (Cropping, Flipping)
        if self.transforms is not None:
            img_tensor, target = self.transforms(img_tensor, target)

        # Recalculate area and iscrowd AFTER transforms in case bounding box sizes changed
        final_boxes = target["boxes"]
        if len(final_boxes) > 0:
            area = (final_boxes[:, 3] - final_boxes[:, 1]) * (final_boxes[:, 2] - final_boxes[:, 0])
            iscrowd = torch.zeros((len(final_boxes),), dtype=torch.int64)
        else:
            area = torch.empty((0,), dtype=torch.float32)
            iscrowd = torch.empty((0,), dtype=torch.int64)

        target["area"] = area
        target["iscrowd"] = iscrowd

        return img_tensor, target

# --- 3. Model Setup ---
def get_model_instance_segmentation(num_classes, is_training=True):
    # Load ImageNet weights for training, use None for inference
    weights = "DEFAULT" if is_training else None
    
    model = maskrcnn_resnet50_fpn_v2(
        weights=weights,
        trainable_backbone_layers=5,
        min_size=800,  
        max_size=1024,
        rpn_pre_nms_top_n_train=2000, 
        rpn_post_nms_top_n_train=1000, 
        rpn_pre_nms_top_n_test=1000,   
        rpn_post_nms_top_n_test=1000,  
        box_detections_per_img=500    
    )
    
    # --- ASSIGNMENT MODIFICATION: Custom Micro-Anchors ---
    # Default Mask R-CNN anchors are designed for large natural objects: (32, 64, 128, 256, 512)
    # We shift the scales down to detect micro-scale medical cells.
    # Note: FPN models require exactly 5 anchor sizes (one for each feature map level).
    anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    
    micro_anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios
    )
    
    # Inject the customized module into the Region Proposal Network (RPN)
    model.rpn.anchor_generator = micro_anchor_generator
    # ----------------------------------------------------
    
    # Replace the bounding box predictor head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Replace the mask predictor head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    return model

def collate_fn(batch):
    return tuple(zip(*batch))

# --- 4. Main Training Loop ---
def main():
    parser = argparse.ArgumentParser(description="Train Instance Segmentation Model")
    parser.add_argument("--run_name", type=str, required=True, help="Name of the run for output logging (e.g., run1)")
    args = parser.parse_args()

    out_dir = os.path.join("outputs", args.run_name)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device} | Output Directory: {out_dir}")

    num_classes = 5 
    
    # Initialize the train dataset with the new v2 transforms for actual training
    dataset = MedicalCellDataset(root_dir="data/train", transforms=get_transforms(train=True))
    
    # --- FIX: Initialize a dataset WITHOUT transforms to accurately count labels ---
    clean_dataset = MedicalCellDataset(root_dir="data/train", transforms=None)
    
    # check_class_imbalance(clean_dataset) # Pass the clean one here too!
    
    indices = torch.randperm(len(dataset)).tolist()
    train_size = int(0.8 * len(dataset))
    dataset_train = torch.utils.data.Subset(dataset, indices[:train_size])

    class_counts = {
        1: 14537, 
        2: 15653, 
        3: 630, 
        4: 587
    }
    total_instances = sum(class_counts.values())

    class_weights = {cls: total_instances / count for cls, count in class_counts.items()}

    image_weights = []
    for idx in range(len(dataset_train)):
        original_idx = dataset_train.indices[idx]
        
        # --- FIX: Use the clean_dataset to see every cell in the full image ---
        _, target = clean_dataset[original_idx] 
        
        labels = target["labels"].tolist()
        
        if len(labels) == 0:
            image_weights.append(0.1) 
            continue
            
        max_weight_in_image = max([class_weights.get(lbl, 1.0) for lbl in labels])
        image_weights.append(max_weight_in_image)

    image_weights_tensor = torch.DoubleTensor(image_weights)
    sampler = WeightedRandomSampler(
        weights=image_weights_tensor, 
        num_samples=len(image_weights_tensor), 
        replacement=True
    )

    data_loader = DataLoader(
        dataset_train, 
        batch_size=1, 
        sampler=sampler, 
        collate_fn=collate_fn, 
        num_workers=4
    )
    
    num_epochs = 30

    # To keep validation consistent, we can evaluate on crops, or implement a separate uncropped dataset
    # Applying the same transforms to validation for memory stability
    dataset_val = torch.utils.data.Subset(dataset, indices[train_size:])
    data_loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=4)

    model = get_model_instance_segmentation(num_classes, is_training=True)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.002, momentum=0.9, weight_decay=0.0001)

    # Smoothly decay the LR from 0.002 down to 0.00001 over the 30 epochs
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    scaler = torch.amp.GradScaler('cuda')
    writer = SummaryWriter(log_dir=out_dir)

    best_loss = float('inf')
    accumulation_steps = 4 

    for epoch in range(num_epochs): 
        model.train()
        epoch_train_loss = 0
        
        optimizer.zero_grad() 
        
        # --- Training Phase ---
        for step, (images, targets) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.autocast(device_type='cuda'):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses = losses / accumulation_steps

            scaler.scale(losses).backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(data_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            current_step_loss = losses.item() * accumulation_steps
            epoch_train_loss += current_step_loss
            
            global_step = epoch * len(data_loader) + step
            writer.add_scalar('Loss/train_step', current_step_loss, global_step)

            del loss_dict, losses, images, targets
            torch.cuda.empty_cache()

        # --- Validation Phase ---
        epoch_val_loss = 0
        with torch.no_grad():
            for images, targets in data_loader_val:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                # Fixed deprecated autocast syntax here
                with torch.autocast(device_type='cuda'):
                    loss_dict_val = model(images, targets)
                    val_losses = sum(loss for loss in loss_dict_val.values())
                
                epoch_val_loss += val_losses.item()

        lr_scheduler.step()
        
        # --- Extract current learning rate ---
        current_lr = optimizer.param_groups[0]['lr']
        
        avg_train_loss = epoch_train_loss / len(data_loader)
        avg_val_loss = epoch_val_loss / len(data_loader_val)
        
        print(f"Epoch: {epoch+1}/{num_epochs} | LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
        writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
        
        # --- Log LR to TensorBoard ---
        writer.add_scalar('Hyperparameters/Learning_Rate', current_lr, epoch)

        latest_model_path = os.path.join(out_dir, "latest_model.pth")
        best_model_path = os.path.join(out_dir, "best_model.pth")
        
        torch.save(model.state_dict(), latest_model_path)
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"--> Saved new best model with Val Loss: {best_loss:.4f}")

    writer.close()
    print("Training Complete!")

if __name__ == "__main__":
    main()