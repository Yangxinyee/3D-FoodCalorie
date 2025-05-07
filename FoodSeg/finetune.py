import os
import numpy as np
import argparse
import torch
import torchvision
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as T
from tqdm import tqdm

class FoodSeg103Dataset(torch.utils.data.Dataset):
    def __init__(self, root, subset="train", transforms=None):
        self.root = root
        self.transforms = transforms
        self.img_dir = os.path.join(root, "Images", "img_dir", subset)
        self.mask_dir = os.path.join(root, "Images", "ann_dir", subset)
        self.imgs = sorted(os.listdir(self.img_dir))
        self.masks = sorted(os.listdir(self.mask_dir))
        assert len(self.imgs) == len(self.masks), f"Mismatch: {len(self.imgs)} images vs {len(self.masks)} masks"

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


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--gpus', type=str, default='0', help='Comma-separated GPU IDs to use, e.g. "0,1,2,3"')
#     parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training and evaluation')
#     parser.add_argument('--save_path', type=str, default='./checkpoints', help='Path to save checkpoints and logs')
#     parser.add_argument('--resume', action='store_true', help='Resume training from the last checkpoint')
#     parser.add_argument('--num_epochs', type=int, default=30, help='Total number of training epochs')
#     parser.add_argument('--dataset_root', type=str, default='../FoodSeg103', help='Path to the FoodSeg103 dataset root')
#     parser.add_argument('--start_epoch', type=int, default=None, help='Start epoch for training')

#     args = parser.parse_args()

#     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
#     os.makedirs(args.save_path, exist_ok=True)
#     log_file_path = os.path.join(args.save_path, "eval_log.txt")

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"[INFO] Using device: {device}")

#     dataset_root = args.dataset_root
#     num_classes = 104  # 103 classes + background

#     dataset = FoodSeg103Dataset(dataset_root, subset="train", transforms=get_transform())
#     dataset_test = FoodSeg103Dataset(dataset_root, subset="test", transforms=get_transform())
#     data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
#     data_loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
#     print("[INFO] DataLoader created: ", len(data_loader))

#     model = get_model_instance_segmentation(num_classes)
#     model.to(device)
#     if torch.cuda.device_count() > 1:
#         print(f"[INFO] Using {torch.cuda.device_count()} GPUs")
#         model = torch.nn.DataParallel(model)
#     else:
#         print("[INFO] Using a single GPU")

#     start_epoch = 0
#     num_epochs = args.num_epochs

#     last_path = os.path.join(args.save_path, "last.pth")
#     params = [p for p in model.parameters() if p.requires_grad]
#     optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
#     lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

#     if args.resume and os.path.exists(last_path):
#         print(f"[INFO] Resuming from checkpoint: {last_path}")
#         checkpoint = torch.load(last_path, map_location=device)

#         model_state = checkpoint.get('model', checkpoint)  # fallback for legacy format
#         if isinstance(model, torch.nn.DataParallel):
#             model.module.load_state_dict(model_state)
#         else:
#             model.load_state_dict(model_state)

#         if 'optimizer' in checkpoint:
#             optimizer.load_state_dict(checkpoint['optimizer'])
#         if 'lr_scheduler' in checkpoint:
#             lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

#         if 'epoch' in checkpoint:
#             start_epoch = checkpoint['epoch'] + 1
#             print(f"[INFO] Resumed from epoch {checkpoint['epoch']}")

#     if args.start_epoch is not None:
#         print(f"[INFO] Overriding start_epoch with {args.start_epoch}")
#         if args.start_epoch > num_epochs:
#             raise ValueError(f"Start epoch {args.start_epoch} is greater than the total number of epochs {num_epochs}")
#         start_epoch = args.start_epoch

#     if not args.resume:
#         with open(log_file_path, "w") as f:
#             f.write("Epoch,Mean_IoU,IoU_Threshold,AP\n")

#     for epoch in range(start_epoch, num_epochs):
#         print(f"\n[INFO] Starting epoch {epoch+1}/{num_epochs}...")
#         model.train()
#         epoch_loss = 0
#         progress_bar = tqdm(data_loader, desc=f"[Training] Epoch {epoch+1}")

#         for _, (images, targets) in enumerate(progress_bar):
#             images = [img.to(device) for img in images]
#             targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

#             loss_dict = model(images, targets)
#             losses = sum(loss for loss in loss_dict.values())
#             epoch_loss += losses.item()

#             optimizer.zero_grad()
#             losses.backward()
#             optimizer.step()

#             losses.detach().cpu()
#             progress_bar.set_postfix(loss=losses.item())

#         lr_scheduler.step()
#         print(f"[INFO] Training Loss after Epoch {epoch+1}: {epoch_loss:.4f}")

#         # Evaluation
#         mean_aps, mean_iou = evaluate(model, data_loader_test, device)
#         print(f"[INFO] Evaluation Results after Epoch {epoch+1}:")
#         print(f" - Mean IoU: {mean_iou:.4f}")
#         for iou_thresh, ap in mean_aps.items():
#             print(f" - mAP@{iou_thresh:.2f}: {ap:.4f}")

#         # Write evaluation results to log file
#         with open(log_file_path, "a") as f:
#             for iou_thresh, ap in mean_aps.items():
#                 f.write(f"{epoch},{mean_iou:.4f},{iou_thresh:.2f},{ap:.4f}\n")

#         # Save checkpoint
#         checkpoint_path = os.path.join(args.save_path, f"mrcnn_foodseg103_{epoch}.pth")
#         save_dict = {
#             'epoch': epoch,
#             'model': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
#             'optimizer': optimizer.state_dict(),
#             'lr_scheduler': lr_scheduler.state_dict()
#         }
#         torch.save(save_dict, checkpoint_path)
#         torch.save(save_dict, last_path)
#         print(f"[INFO] Checkpoint saved at {checkpoint_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--save_path', type=str, default='./checkpoints')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--dataset_root', type=str, default='../FoodSeg103')
    parser.add_argument('--start_epoch', type=int, default=None)
    args = parser.parse_args()
    print(f"[DEBUG] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, local_rank={args.local_rank}, device_id={torch.cuda.current_device()}")


    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

    os.makedirs(args.save_path, exist_ok=True)
    log_file_path = os.path.join(args.save_path, "eval_log.txt")

    dataset = FoodSeg103Dataset(args.dataset_root, subset="train", transforms=get_transform())
    dataset_test = FoodSeg103Dataset(args.dataset_root, subset="test", transforms=get_transform())

    train_sampler = DistributedSampler(dataset)
    test_sampler = DistributedSampler(dataset_test, shuffle=False)

    data_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler,
                             num_workers=4, collate_fn=collate_fn, pin_memory=True)
    data_loader_test = DataLoader(dataset_test, batch_size=args.batch_size, sampler=test_sampler,
                                  num_workers=4, collate_fn=collate_fn, pin_memory=True)

    model = get_model_instance_segmentation(num_classes=104).to(device)
    model = DDP(model, device_ids=[args.local_rank])

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    last_path = os.path.join(args.save_path, "last.pth")
    start_epoch = 0
    if args.resume and os.path.exists(last_path):
        checkpoint = torch.load(last_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint["epoch"] + 1

    if args.start_epoch is not None:
        start_epoch = args.start_epoch

    if args.local_rank == 0 and not args.resume:
        with open(log_file_path, "w") as f:
            f.write("Epoch,Mean_IoU,IoU_Threshold,AP\n")

    for epoch in range(start_epoch, args.num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0

        for images, targets in data_loader:
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()

        lr_scheduler.step()

        if args.local_rank == 0:
            print(f"[Epoch {epoch+1}] Loss: {epoch_loss:.4f}")
            mean_aps, mean_iou = evaluate(model.module, data_loader_test, device)
            with open(log_file_path, "a") as f:
                for iou_thresh, ap in mean_aps.items():
                    f.write(f"{epoch},{mean_iou:.4f},{iou_thresh:.2f},{ap:.4f}\n")

            save_dict = {
                'epoch': epoch,
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()
            }
            torch.save(save_dict, os.path.join(args.save_path, f"mrcnn_foodseg103_{epoch}.pth"))
            torch.save(save_dict, last_path)
            print(f"[INFO] Checkpoint saved after epoch {epoch+1}")

if __name__ == "__main__":
    main()
