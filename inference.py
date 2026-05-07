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
def get_model_instance_segmentation(num_classes):
    model = maskrcnn_resnet50_fpn_v2(
        weights=None, # No need to download ImageNet weights for inference
        min_size=800,  
        max_size=1024,
        rpn_pre_nms_top_n_train=2000, 
        rpn_post_nms_top_n_train=1000, 
        rpn_pre_nms_top_n_test=1000,   
        rpn_post_nms_top_n_test=1000,  
        box_detections_per_img=500
    )
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
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
    parser.add_argument("--score_threshold", type=float, default=0.5, help="Minimum confidence score to include in submission")
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # 4 cell classes + 1 background class
    num_classes = 5 
    
    print(f"Loading model weights from {args.model_path}...")
    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval() # CRITICAL: Put model in evaluation mode

    print(f"Loading metadata from {args.meta_json}...")
    with open(args.meta_json, 'r') as f:
        test_metadata = json.load(f)

    coco_results = []
    total_images = len(test_metadata)

    print(f"Starting inference on {total_images} images...")
    
    # We don't need gradients for inference
    with torch.no_grad():
        for i, img_info in enumerate(test_metadata):
            img_filename = img_info['file_name']
            img_id = img_info['id']
            img_path = os.path.join(args.test_dir, img_filename)

            # 1. Load and process image exactly as in training
            img_array = sio.imread(img_path)
            
            if len(img_array.shape) == 2:
                img_array = np.stack((img_array,)*3, axis=-1)
            elif img_array.shape[-1] > 3:
                img_array = img_array[:, :, :3]
                
            img_tensor = torch.as_tensor(img_array, dtype=torch.float32).permute(2, 0, 1) / 255.0
            img_tensor = img_tensor.to(device)

            # Dynamically set the transform bounds to the exact dimensions of the current test image.
            # This forces the internal scale_factor to be exactly 1.0
            _, H, W = img_tensor.shape
            model.transform.min_size = (min(H, W),)
            model.transform.max_size = max(H, W)

            # 2. Run Forward Pass
            # For inference, the model takes a list of tensors and returns a list of dictionaries
            # We also explicitly use autocast to match our mixed precision training footprint
            with torch.autocast(device_type='cuda'):
                predictions = model([img_tensor])[0] 

            # 3. Extract outputs
            masks = predictions['masks'].cpu().numpy()
            scores = predictions['scores'].cpu().numpy()
            labels = predictions['labels'].cpu().numpy()

            # 4. Filter and encode instances
            instances_added = 0
            for j in range(len(scores)):
                score = float(scores[j])
                
                if score < args.score_threshold:
                    continue
                
                class_id = int(labels[j])
                
                # Mask R-CNN outputs a shape of [1, H, W] per instance. We need [H, W].
                # Values are probabilities [0, 1]. We threshold at 0.5 for binary mask.
                binary_mask = (masks[j, 0, :, :] > 0.5)
                
                # Encode the mask to COCO RLE format using the provided function
                rle_mask = encode_mask(binary_mask)

                # Append to our submission list
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