import os
import csv
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pycocotools import mask as mask_util
from ForensicHub.registry import MODELS, build_from_registry

def mask_to_rle(mask):
    mask = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_util.encode(mask)
    rle['counts'] = rle['counts'].decode('utf-8')  # for JSON/csv serialization
    return rle


class ImageFolderDataset(Dataset):
    def __init__(self, input_dir, image_size):
        self.image_size = image_size
        self.samples = []
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        for folder in os.listdir(input_dir):
            subdir = os.path.join(input_dir, folder)
            if not os.path.isdir(subdir):
                continue
            for fname in os.listdir(subdir):
                if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                    full_path = os.path.join(subdir, fname)
                    rel_path = os.path.relpath(full_path, input_dir)
                    self.samples.append((rel_path, full_path))
        self.input_dir = input_dir

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, full_path = self.samples[idx]
        image = Image.open(full_path).convert("RGB")
        image_tensor = self.transform(image)
        return rel_path, image_tensor


def infer_and_save_csv(input_dir, output_csv, model_args,
                       batch_size=16, num_workers=4, use_amp=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = build_from_registry(MODELS, model_args).to(device)
    checkpoint = torch.load(model_args['init_path'], map_location=device)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.eval()

    image_size = model_args['init_config'].get('image_size', 512)
    dataset = ImageFolderDataset(input_dir, image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    results_dict = {} 

    for rel_paths, images in tqdm(dataloader, desc="Running inference"):
        images = images.to(device)
        dummy_label = torch.ones(images.size(0), device=device)
        dummy_mask = torch.ones(images.size(0), 1, image_size, image_size, device=device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                out_dict = model(image=images, label=dummy_label, mask=dummy_mask)
                logits = out_dict["pred_label"]  # [B]
                probs = torch.sigmoid(logits)    # [B]
                pred_masks = torch.sigmoid(out_dict["pred_mask"]).cpu().numpy()  # [B, 1, H, W]

        for i in range(images.size(0)):
            rel_path = rel_paths[i]
            prob = probs[i].item()
            if prob >= 0.5:
                binary_mask = (pred_masks[i, 0] > 0.5).astype(np.uint8)
                rle = mask_to_rle(binary_mask)
                results_dict[rel_path] = [rel_path, str(rle), 1]
            else:
                results_dict[rel_path] = [rel_path, "", 0]

    # 最终写入 CSV 的结果列表
    results = [["Path", "Region", "Label"]] + list(results_dict.values())

    # 写入 CSV 文件
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(results)

    print(f"Saved CSV to {output_csv}")


# Example usage
if __name__ == "__main__":
    input_dir = "/kaggle/input/foren-tmp-data/_output_/testA_change"
    output_csv = "inference_results_testA.csv"
    model_args = {
        "name": "BisaiBaseline",
        "init_config": {
        },
        "init_path": "/kaggle/input/forensichub/pytorch/main/1/bisai1/checkpoint-19.pth"
    }

    infer_and_save_csv(
        input_dir=input_dir,
        output_csv=output_csv,
        model_args=model_args,
        batch_size=64,
        num_workers=32,
        use_amp=True
    )
