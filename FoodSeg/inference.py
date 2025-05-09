import os
import torch
import numpy as np
import cv2
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms as T
from PIL import Image
import argparse

# -------- CONFIG --------
NUM_CLASSES = 104  # 103 food classes + background
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- ARGPARSE SETUP --------
def parse_args():
    parser = argparse.ArgumentParser(description="Run Mask R-CNN food segmentation.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with input dish folders")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output masks")
    parser.add_argument("--category_path", type=str, required=True, help="Path to class category file")
    return parser.parse_args()

# -------- MODEL SETUP --------
def get_model(model_path):
    model = maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 512
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, NUM_CLASSES)
    data = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(data['model'])
    model.to(DEVICE)
    model.eval()
    return model

# -------- LOAD CLASS NAMES --------
def load_class_names(category_file):
    class_names = []
    with open(category_file, 'r') as f:
        for line in f:
            _, name = line.strip().split(maxsplit=1)
            class_names.append(name)
    return class_names

# -------- DRAW PREDICTIONS --------
def draw_instance_predictions(img, masks, labels, class_names, score_thresh=0.5):
    for i in range(len(masks)):
        mask = masks[i]
        color = np.random.randint(0, 255, (3,), dtype=np.uint8)
        color_mask = np.stack([mask * color[j] for j in range(3)], axis=-1)
        img = np.where(mask[:, :, None], img * 0.5 + color_mask * 0.5, img)

        coords = np.column_stack(np.where(mask))
        if coords.shape[0] == 0:
            continue
        y_center, x_center = coords.mean(axis=0).astype(int)

        label = class_names[labels[i]] if labels[i] < len(class_names) else str(labels[i])
        cv2.putText(
            img, label, (x_center, y_center),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA
        )

    return img.astype(np.uint8)

def draw_food_only_image(img, masks, score_thresh=0.5):
    if len(masks) == 0:
        return np.zeros_like(img)

    # Combine all masks above the threshold
    combined_mask = np.zeros(masks[0].shape, dtype=bool)
    for mask in masks:
        combined_mask |= mask

    # Apply mask: keep only food regions, black out background
    masked_img = np.zeros_like(img)
    masked_img[combined_mask] = img[combined_mask]
    return masked_img

# -------- MAIN FUNCTION --------
def process_directory(input_dir, output_dir, model_path, category_path):
    model = get_model(model_path)
    class_names = load_class_names(category_path)
    os.makedirs(output_dir, exist_ok=True)

    for dish_folder in os.listdir(input_dir):
        dish_path = os.path.join(input_dir, dish_folder)
        rgb_path = os.path.join(dish_path, "rgb.png")

        if not os.path.isfile(rgb_path):
            print(f"[WARN] rgb.png not found in {dish_path}")
            continue

        image = Image.open(rgb_path).convert("RGB").resize((256, 192))
        orig = np.array(image)

        transform = T.ToTensor()
        img_tensor = transform(image).to(DEVICE)

        with torch.no_grad():
            output = model([img_tensor])[0]

        masks = output["masks"].squeeze(1).cpu().numpy() > 0.5
        labels = output["labels"].cpu().numpy()

        # result = draw_instance_predictions(orig.copy(), masks, labels, class_names)
        # output_path = os.path.join(output_dir, f"{dish_folder}_mask.png")
        result = draw_food_only_image(orig.copy(), masks)
        output_path = os.path.join(output_dir, f"{dish_folder}_masked.png")
        cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print(f"[INFO] Saved mask to {output_path}")
        
if __name__ == "__main__":
    args = parse_args()
    process_directory(args.input_dir, args.output_dir, args.model_path, args.category_path)
