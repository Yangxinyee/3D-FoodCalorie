import os
import yaml
import numpy as np
import argparse
import torch
import torch.distributed as dist
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from core.evaluation.evaluate_depth import eval_depth
from core.networks.model_depth_pose import Model_depth_pose

seed = 42

def collate_fn(batch):
    imgs, depths = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    depths = torch.stack(depths, dim=0)
    return imgs, depths

def get_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225])
    ])

def load_model(path, model, sigmoid=True):
    data = torch.load(path)
    state_dict = data['model_state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    state_dict = new_state_dict
    if not sigmoid:
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if not k.startswith('depth_net.decoder'):
                filtered_state_dict[k] = v
        state_dict = filtered_state_dict
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        raise RuntimeError(f"Failed to load weights: {e}")
    return model

def depth_to_disp(gt_depth, min_depth=0.03, max_depth=1.2):
    """
    Convert depth (in meters) to normalized disparity in [0,1]
    Invalid values (<=0 or >= max_sensor_range) will be set to 0
    """
    valid_mask = (gt_depth > min_depth) & (gt_depth < max_depth)

    inv_depth = 1.0 / torch.clamp(gt_depth, min=min_depth, max=max_depth)
    min_disp = 1.0 / max_depth
    max_disp = 1.0 / min_depth

    disp = (inv_depth - min_disp) / (max_disp - min_disp)
    disp[~valid_mask] = 0.0

    return disp

class DepthFinetuneDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, target_size=(192, 256), split_ratio=0.8, subset='train', few_shot=-1):
        self.root = root
        self.sample_dirs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        self.transform = transform
        self.target_size = target_size  # (H, W)

        all_samples = sorted([os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        np.random.seed(seed)
        np.random.shuffle(all_samples)

        if few_shot == -1:
            split_index = int(len(all_samples) * split_ratio)
            if subset == 'train':
                self.sample_dirs = all_samples[:split_index]
            elif subset == 'test':
                self.sample_dirs = all_samples[split_index:]
            elif subset == 'all':
                self.sample_dirs = all_samples
            else:
                raise ValueError(f"subset should be 'train' or 'test', got {subset}")
        else:
            if subset == 'train':
                self.sample_dirs = all_samples[:few_shot]
            elif subset == 'test':
                self.sample_dirs = all_samples[few_shot:]
            else:
                raise ValueError(f"subset should be 'train' or 'test', got {subset}")

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        rgb_path = os.path.join(sample_dir, 'rgb.png')
        depth_path = os.path.join(sample_dir, 'depth_raw.png')

        # Load images
        try:
            rgb = Image.open(rgb_path).convert('RGB')
            depth = Image.open(depth_path)
        except Exception as e:
            print(f"[WARNING] Skipping invalid image at {sample_dir}: {e}")
            return self.__getitem__((idx + 1) % len(self.sample_dirs))

        # Resize
        rgb = rgb.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)
        depth = depth.resize((self.target_size[1], self.target_size[0]), Image.NEAREST)

        # To tensor
        depth_np = np.array(depth).astype(np.float32) * 1e-4  # convert mm to meters if needed
        depth_tensor = torch.from_numpy(depth_np).unsqueeze(0)  # [1, H, W]

        if self.transform:
            rgb_tensor = self.transform(rgb)
        else:
            rgb_tensor = T.ToTensor()(rgb)

        return rgb_tensor, depth_tensor
    
def silog_loss(pred, target, mask=None, variance_focus=0.85):
    """
    Scale-Invariant Logarithmic Loss (SILog), suitable for depth regression.
    Reference: Eigen et al. 2014
    """
    eps = 1e-6
    if mask is not None:
        pred, target = pred[mask], target[mask]

    g = torch.log(pred + eps) - torch.log(target + eps)
    d = torch.var(g) + variance_focus * (torch.mean(g) ** 2)
    return d * 10.0

def mse_loss(pred, target, mask=None):
    if mask is not None:
        pred, target = pred[mask], target[mask]
    return torch.mean((pred - target) ** 2)

def finetune_one_epoch(model, data_loader, optimizer, device, epoch, max_depth=1.2, min_depth=0.03, sigmoid=True):
    model.train()
    epoch_loss = 0.0

    rank = dist.get_rank()
    if rank == 0:
        pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"[Train] Epoch {epoch+1}")
    else:
        pbar = enumerate(data_loader)

    for i, (rgb, gt_depth) in pbar:
        rgb = rgb.to(device, non_blocking=True)            # [B, 3, H, W]
        gt_depth_raw = gt_depth.to(device, non_blocking=True) # [B, 1, H, W]

        optimizer.zero_grad()

        pred_disp = model.module.infer_depth(rgb)
        if not sigmoid:
            pred_depth = 1.0 / (pred_disp + 1e-6)
        else:
            min_disp = 1.0 / max_depth
            max_disp = 1.0 / min_depth
            scaled_disp = min_disp + (max_disp - min_disp) * pred_disp
            pred_depth = 1.0 / scaled_disp

        mask = (gt_depth_raw > min_depth) & (gt_depth_raw < max_depth)
        pred_depth = torch.clamp(pred_depth, min=min_depth, max=max_depth)
        gt_depth = torch.clamp(gt_depth_raw, min=min_depth, max=max_depth)

        # loss = silog_loss(pred_depth, gt_depth, mask)
        loss = mse_loss(pred_depth, gt_depth, mask)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

        if rank == 0:
            pbar.set_postfix(loss=loss.item())

    return epoch_loss / len(data_loader)

def evaluate_on_dataset(model, data_loader, device, min_depth=0.03, max_depth=1.2):
    model.eval()
    pred_depths, gt_depths = [], []

    with torch.no_grad():
        for rgb, gt_depth in data_loader:
            rgb = rgb.to(device)
            gt_depth = gt_depth.squeeze(1).cpu().numpy()  # [B, H, W]

            pred_disp = model.infer_depth(rgb) 
            # min_disp = 1.0 / max_depth
            # max_disp = 1.0 / min_depth
            # scaled_disp = min_disp + (max_disp - min_disp) * pred_disp
            # pred_depth = 1.0 / scaled_disp    
            _, pred_depth = model.disp2depth(pred_disp, min_depth=min_depth, max_depth=max_depth)

            # pred_depth = 1.0 / (pred_disp + 1e-6)
            # pred_depth = torch.clamp(pred_depth, min=min_depth, max=max_depth)
            pred_depth = pred_depth.squeeze(1).cpu().numpy()

            gt_depths.extend(gt_depth)
            pred_depths.extend(pred_depth)
    return eval_depth(gt_depths, pred_depths, min_depth=min_depth, max_depth=max_depth, nyu=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', type=str, required=True, help='Path to pretrained model.')
    parser.add_argument('--config_file', type=str, required=True, help='Path to config YAML.')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--save_path', type=str, default='./checkpoints')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--encoder_decoder', type=int, default=0, help='0: encoder-decoder, 1: encoder-only, 2: decoder-only')
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--mode', type=str, default='depth', help='training mode.')
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--start_epoch', type=int, default=None)
    parser.add_argument('--few_shot', type=int, default=-1, help='number of few-shot training samples')
    args = parser.parse_args()

    if not os.path.exists(args.config_file):
        raise ValueError('config file not found.')
    with open(args.config_file, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['img_hw'] = (cfg['img_hw'][0], cfg['img_hw'][1])

    # copy attr into cfg
    for attr in dir(args):
        if attr[:2] != '__':
            cfg[attr] = getattr(args, attr)
    class pObject(object):
        def __init__(self):
            pass
    cfg_new = pObject()
    for attr in list(cfg.keys()):
        setattr(cfg_new, attr, cfg[attr])

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_id = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)

    os.makedirs(args.save_path, exist_ok=True)

    dataset = DepthFinetuneDataset(args.dataset_root, transform=get_transform(), split_ratio=0.8, subset='all', few_shot=args.few_shot)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    dataset_test = DepthFinetuneDataset(args.dataset_root, transform=get_transform(), split_ratio=0.8, subset='all')
    sampler_test = DistributedSampler(dataset_test, num_replicas=world_size, rank=rank, shuffle=False)

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    data_loader_test = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        sampler=sampler_test,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    model = Model_depth_pose(cfg_new)
    model = load_model(args.pretrained_model, model)
    model = model.to(device)
    model = DDP(model, device_ids=[device_id])
    for name, param in model.named_parameters():
        if args.encoder_decoder == 0:
            if not (name.startswith("module.depth_net.encoder") or name.startswith("module.depth_net.decoder")):
                param.requires_grad = False
        elif args.encoder_decoder == 1:
            if not name.startswith("module.depth_net.encoder"):
                param.requires_grad = False
        elif args.encoder_decoder == 2:
            if not name.startswith("module.depth_net.decoder"):
                param.requires_grad = False
        else:
            raise ValueError(f"Invalid encoder_decoder value: {args.encoder_decoder}")
            
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=1e-2
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Resume
    last_ckpt = os.path.join(args.save_path, 'last.pth')
    if args.resume and os.path.exists(last_ckpt):
        checkpoint = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if rank == 0:
            print(f"[INFO] Resumed from checkpoint at epoch {checkpoint['epoch']}")

    if args.start_epoch is None:
        start_epoch = 0
    else:
        start_epoch = args.start_epoch

    log_path = os.path.join(args.save_path, 'eval_log.txt')
    if rank == 0 and start_epoch == 0:
        with open(log_path, 'w') as f:
            f.write("Epoch,abs_rel,sq_rel,rms,log_rms,a1,a2,a3\n")

    for epoch in range(start_epoch, args.num_epochs):
        sampler.set_epoch(epoch)
        epoch_loss = finetune_one_epoch(model, data_loader, optimizer, device, epoch)
        lr_scheduler.step()

        if rank == 0:
            print(f"[Epoch {epoch+1}] Average Training Loss: {epoch_loss:.4f}")

            eval_metrics = evaluate_on_dataset(model.module, data_loader_test, device, min_depth=0.03, max_depth=1.2, nyu=True)
            abs_rel, sq_rel, rms, log_rms, a1, a2, a3 = eval_metrics
            with open(log_path, 'a') as f:
                f.write(f"{epoch+1},{abs_rel:.4f},{sq_rel:.4f},{rms:.4f},{log_rms:.4f},{a1:.4f},{a2:.4f},{a3:.4f}\n")

            save_dict = {
                'epoch': epoch,
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()
            }

            torch.save(save_dict, os.path.join(args.save_path, f'depth_epoch_{epoch}.pth'))
            torch.save(save_dict, last_ckpt)
            print(f"[INFO] Checkpoint saved after epoch {epoch+1}")

if __name__ == "__main__":
    main()


