import os
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms as T
from tqdm import tqdm

CHECKPOINT_PATH = "/home/checkpoints/mrcnn_foodseg103_14.pth"

class FoodSeg103Dataset(torch.utils.data.Dataset):
    def __init__(self, root, subset="train", transforms=None):
        self.root = root
        self.transforms = transforms
        self.img_dir = os.path.join(root, "Images", "img_dir", subset)
        self.mask_dir = os.path.join(root, "Images", "ann_dir", subset)
        self.imgs = sorted(os.listdir(self.img_dir))
        self.masks = sorted(os.listdir(self.mask_dir))

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        # Resize both image and mask to 192x256 BEFORE any further processing
        resize_size = (256, 192)  # (width, height) as required by PIL
        img = img.resize(resize_size, resample=Image.BILINEAR)
        mask = mask.resize(resize_size, resample=Image.NEAREST)  # preserve label values

        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]

        masks = mask == obj_ids[:, None, None]

        boxes = []
        valid_masks = []
        valid_labels = []

        for i, obj_id in enumerate(obj_ids):
            pos = np.where(masks[i])
            if pos[0].size == 0 or pos[1].size == 0:
                continue

            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            valid_masks.append(masks[i])
            valid_labels.append(obj_id)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            masks = torch.zeros((0, mask.shape[0], mask.shape[1]), dtype=torch.uint8)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            valid_masks = np.stack(valid_masks, axis=0)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            masks = torch.as_tensor(valid_masks, dtype=torch.uint8)
            labels = torch.as_tensor(valid_labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((labels.shape[0],), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

def get_transform():
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

def get_model_instance_segmentation(num_classes):
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 512
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

def collate_fn(batch):
    return tuple(zip(*batch))

def compute_iou(pred_mask, gt_mask):
    intersection = (pred_mask & gt_mask).float().sum()
    union = (pred_mask | gt_mask).float().sum()
    if union == 0:
        return torch.tensor(0.0)
    else:
        return intersection / union

def evaluate(model, data_loader, device, iou_thresholds=[0.5, 0.75]):
    model.eval()
    APs = {iou_thresh: [] for iou_thresh in iou_thresholds}
    IoUs = []

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="[Evaluation]"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            for output, target in zip(outputs, targets):
                gt_masks = target["masks"]
                gt_labels = target["labels"]

                pred_masks = output["masks"].squeeze(1) > 0.5
                pred_labels = output["labels"]
                pred_scores = output["scores"]

                num_gt = gt_labels.shape[0]
                num_pred = pred_labels.shape[0]

                ious = torch.zeros((num_pred, num_gt))

                for pred_idx in range(num_pred):
                    for gt_idx in range(num_gt):
                        ious[pred_idx, gt_idx] = compute_iou(pred_masks[pred_idx], gt_masks[gt_idx])

                if num_gt == 0:
                    continue

                for iou_thresh in iou_thresholds:
                    matches = []
                    used_gts = set()
                    for pred_idx in pred_scores.argsort(descending=True):  # highest confidence first
                        max_iou = 0
                        max_gt_idx = -1
                        for gt_idx in range(num_gt):
                            if gt_idx in used_gts:
                                continue
                            if ious[pred_idx, gt_idx] > max_iou:
                                max_iou = ious[pred_idx, gt_idx]
                                max_gt_idx = gt_idx

                        if max_iou >= iou_thresh:
                            matches.append(1)  # true positive
                            used_gts.add(max_gt_idx)
                        else:
                            matches.append(0)  # false positive

                    tp = torch.tensor(matches).cumsum(0)
                    fp = (1 - torch.tensor(matches)).cumsum(0)
                    recalls = tp / num_gt
                    precisions = tp / (tp + fp)

                    if len(precisions) == 0:
                        ap = 0.0
                    else:
                        ap = torch.trapz(precisions, recalls).item()

                    APs[iou_thresh].append(ap)

                # For IoU mean calculation
                for pred_idx in range(num_pred):
                    best_iou = ious[pred_idx].max()
                    IoUs.append(best_iou.item())

    mean_ious = sum(IoUs) / len(IoUs) if len(IoUs) > 0 else 0.0
    mean_aps = {iou_thresh: (sum(aps) / len(aps) if aps else 0.0) for iou_thresh, aps in APs.items()}
    return mean_aps, mean_ious

def main():
    dataset_root = "/home/dataset/FoodSeg103"
    num_classes = 104  # 103 classes + background
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"[INFO] Using device: {device}")

    # dataset = FoodSeg103Dataset(dataset_root, subset="train", transforms=get_transform())
    dataset_test = FoodSeg103Dataset(dataset_root, subset="test", transforms=get_transform())

    # data_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=collate_fn)
    data_loader_test = DataLoader(dataset_test, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn)

    print("[INFO] DataLoader_test created: ", len(data_loader_test))

    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    mean_aps, mean_iou = evaluate(model, data_loader_test, device)
    print(f"[INFO] Evaluation Results:")
    print(f" - Mean IoU: {mean_iou:.4f}")
    for iou_thresh, ap in mean_aps.items():
        print(f" - mAP@{iou_thresh:.2f}: {ap:.4f}")

if __name__ == "__main__":
    main()