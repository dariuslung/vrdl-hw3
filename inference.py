import os
import argparse
import json
import numpy as np
import torch
import skimage.io as sio
from pycocotools import mask as mask_utils

from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import batched_nms
from torchvision.models.detection.anchor_utils import AnchorGenerator

# --- 1. Provided Assignment Helper Functions ---
def decode_maskobj(mask_obj):
    return mask_utils.decode(mask_obj)

def encode_mask(binary_mask):
    arr = np.asfortranarray(binary_mask).astype(np.uint8)
    rle = mask_utils.encode(arr)
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle

def read_maskfile(filepath):
    mask_array = sio.imread(filepath)
    return mask_array

# --- 2. Model Setup (MUST match training architecture) ---
def get_model_instance_segmentation(num_classes, is_training=False):
    # Load ImageNet weights for training, use None for inference
    weights = "DEFAULT" if is_training else None
    
    model = maskrcnn_resnet50_fpn_v2(
        weights=weights,
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

# --- 3. Main Inference Execution ---
def main():
    parser = argparse.ArgumentParser(description="Run Inference and Generate COCO Results")
    parser.add_argument("--model_path", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--test_dir", type=str, default="data/test_release", help="Directory with test images")
    parser.add_argument("--meta_json", type=str, default="data/test_image_name_to_ids.json", help="Path to image ID mapping")
    parser.add_argument("--output", type=str, default="test-results.json", help="Output JSON filename")
    parser.add_argument("--score_threshold", type=float, default=0.15, help="Minimum confidence score")
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    num_classes = 5 
    
    print(f"Loading model weights from {args.model_path}...")
    model = get_model_instance_segmentation(num_classes, is_training=False)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"Loading metadata from {args.meta_json}...")
    with open(args.meta_json, 'r') as f:
        test_metadata = json.load(f)

    coco_results = []
    total_images = len(test_metadata)

    # --- Sliding Window Parameters ---
    PATCH_SIZE = 800
    STRIDE = 600 # 200 pixel overlap to ensure cells on boundaries aren't cut in half
    NMS_IOU_THRESHOLD = 0.3 # Threshold to remove duplicate boxes in overlapping regions

    print(f"Starting Sliding Window Inference on {total_images} images...")
    
    with torch.no_grad():
        for i, img_info in enumerate(test_metadata):
            img_filename = img_info['file_name']
            img_id = img_info['id']
            img_path = os.path.join(args.test_dir, img_filename)

            img_array = sio.imread(img_path)
            
            if len(img_array.shape) == 2:
                img_array = np.stack((img_array,)*3, axis=-1)
            elif img_array.shape[-1] > 3:
                img_array = img_array[:, :, :3]
                
            img_tensor = torch.as_tensor(img_array, dtype=torch.float32).permute(2, 0, 1) / 255.0
            _, H, W = img_tensor.shape

            # 1. Generate Grid Coordinates
            y_starts = list(range(0, H, STRIDE))
            x_starts = list(range(0, W, STRIDE))
            
            # Ensure we cover the far right and bottom edges perfectly
            if not y_starts or y_starts[-1] + PATCH_SIZE < H:
                y_starts.append(max(0, H - PATCH_SIZE))
            if not x_starts or x_starts[-1] + PATCH_SIZE < W:
                x_starts.append(max(0, W - PATCH_SIZE))
                
            y_starts = sorted(list(set(y_starts)))
            x_starts = sorted(list(set(x_starts)))

            all_boxes = []
            all_scores = []
            all_labels = []
            all_masks = []
            all_offsets = []

            # 2. Process Patches
            for y in y_starts:
                for x in x_starts:
                    y_end = min(H, y + PATCH_SIZE)
                    x_end = min(W, x + PATCH_SIZE)
                    
                    # Extract patch and send to GPU
                    patch = img_tensor[:, y:y_end, x:x_end].to(device)
                    pH, pW = patch.shape[1:]
                    
                    model.transform.min_size = (min(pH, pW),)
                    model.transform.max_size = max(pH, pW)
                    
                    boxes_list = []
                    scores_list = []
                    labels_list = []
                    masks_list = []
                    
                    with torch.autocast(device_type='cuda'):
                        # --- PASS 1: Standard Prediction ---
                        pred = model([patch])[0]
                        
                        # Instantly move to CPU and binarize to save VRAM!
                        boxes = pred['boxes'].cpu()
                        scores = pred['scores'].cpu()
                        labels = pred['labels'].cpu()
                        masks = pred['masks'].cpu() > 0.5 
                        
                        boxes_list.append(boxes)
                        scores_list.append(scores)
                        labels_list.append(labels)
                        masks_list.append(masks)
                        
                        # Delete GPU tensors and clear cache BEFORE running TTA
                        del pred
                        torch.cuda.empty_cache()
                        
                        # --- PASS 2: TTA Flipped Prediction ---
                        patch_flipped = torch.flip(patch, dims=[2])
                        pred_flipped = model([patch_flipped])[0]
                        
                        boxes_f = pred_flipped['boxes'].cpu()
                        scores_f = pred_flipped['scores'].cpu()
                        labels_f = pred_flipped['labels'].cpu()
                        masks_f = pred_flipped['masks'].cpu() > 0.5
                        
                        # Clean up GPU again
                        del patch, patch_flipped, pred_flipped
                        torch.cuda.empty_cache()
                    
                    # --- 3. "Un-flip" the TTA predictions (Done safely on CPU) ---
                    if len(boxes_f) > 0:
                        # Reverse the X coordinates: new_x = width - old_x
                        xmin = pW - boxes_f[:, 2]
                        xmax = pW - boxes_f[:, 0]
                        boxes_f[:, 0] = xmin
                        boxes_f[:, 2] = xmax
                        
                        masks_f = torch.flip(masks_f, dims=[3])
                        
                        boxes_list.append(boxes_f)
                        scores_list.append(scores_f)
                        labels_list.append(labels_f)
                        masks_list.append(masks_f)

                    # Concatenate the standard and TTA predictions
                    boxes = torch.cat(boxes_list)
                    scores = torch.cat(scores_list)
                    labels = torch.cat(labels_list)
                    masks = torch.cat(masks_list)
                    
                    if len(boxes) > 0:
                        # Shift local patch bounding boxes to global image coordinates
                        boxes[:, [0, 2]] += x
                        boxes[:, [1, 3]] += y
                        
                        all_boxes.append(boxes)
                        all_scores.append(scores)
                        all_labels.append(labels)
                        
                        for m_idx in range(len(masks)):
                            all_masks.append(masks[m_idx, 0]) 
                            all_offsets.append((x, y, x_end, y_end))

            # 3. Reconstruct the Full Image
            if len(all_boxes) == 0:
                print(f"Processed {i + 1}/{total_images} | Added 0 instances for {img_filename}")
                continue

            all_boxes = torch.cat(all_boxes)
            all_scores = torch.cat(all_scores)
            all_labels = torch.cat(all_labels)

            # Apply Batched NMS to remove duplicate detections in the overlapping regions
            keep = batched_nms(all_boxes, all_scores, all_labels, iou_threshold=NMS_IOU_THRESHOLD)

            instances_added = 0
            for k in keep:
                idx = k.item()
                score = float(all_scores[idx])
                
                if score < args.score_threshold:
                    continue
                    
                class_id = int(all_labels[idx])
                mask_patch = all_masks[idx].numpy()
                x, y, x_end, y_end = all_offsets[idx]
                
                # Place the small patch mask accurately onto a blank full-size canvas
                full_mask = np.zeros((H, W), dtype=np.uint8)
                patch_h = y_end - y
                patch_w = x_end - x
                full_mask[y:y_end, x:x_end] = mask_patch[:patch_h, :patch_w]
                
                rle_mask = encode_mask(full_mask)

                coco_results.append({
                    "image_id": img_id,
                    "category_id": class_id,
                    "segmentation": rle_mask,
                    "score": score
                })
                instances_added += 1

            if (i + 1) % 10 == 0 or (i + 1) == total_images:
                print(f"Processed {i + 1}/{total_images} | Added {instances_added} instances for {img_filename}")

    # --- 4. Save to JSON ---
    print(f"Writing {len(coco_results)} total predictions to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(coco_results, f)
        
    print("Inference complete! File is ready for submission.")

if __name__ == "__main__":
    main()