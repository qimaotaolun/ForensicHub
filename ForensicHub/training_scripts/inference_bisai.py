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
import torch.nn as nn # 导入 nn 模块

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
    # 检查 GPU 数量，并设置设备
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"检测到 {torch.cuda.device_count()} 个 GPU，将使用 DataParallel。")
        device = torch.device("cuda:0") # 指定主设备为cuda:0
        use_data_parallel = True
    elif torch.cuda.is_available():
        print("检测到单个 GPU，将在单个 GPU 上运行。")
        device = torch.device("cuda")
        use_data_parallel = False
    else:
        print("未检测到 GPU，将在 CPU 上运行。")
        device = torch.device("cpu")
        use_data_parallel = False

    # Load model
    model = build_from_registry(MODELS, model_args).to(device)
    checkpoint = torch.load(model_args['init_path'], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.eval()

    # 如果有多个 GPU，封装模型
    if use_data_parallel:
        model = nn.DataParallel(model) # 将模型封装在 DataParallel 中

    image_size = model_args['init_config'].get('image_size', 512)
    dataset = ImageFolderDataset(input_dir, image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    results_dict = {} 

    for rel_paths, images in tqdm(dataloader, desc="Running inference"):
        images = images.to(device) # 将数据发送到主设备
        dummy_label = torch.ones(images.size(0), device=device)
        dummy_mask = torch.ones(images.size(0), 1, image_size, image_size, device=device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                out_dict = model(image=images, label=dummy_label, mask=dummy_mask)
                # 当使用 DataParallel 时，输出可能被包装成一个列表或元组，需要处理
                if use_data_parallel and isinstance(out_dict, list):
                    # DataParallel 会在每个 GPU 上计算，然后合并结果。
                    # 如果模型输出是字典，它会将字典的值转换为列表并合并。
                    # 这里假设 out_dict 是一个字典，如果你的模型在DataParallel下
                    # 返回的是一个字典，那么需要对字典的值进行合并
                    # 通常情况下，DataParallel会自动处理这些。
                    # 如果模型输出是张量，DataParallel会直接拼接张量。
                    # 对于你的情况，model(image, label, mask) 返回的是一个字典，
                    # DataParallel 会处理它并返回一个合并的字典。
                    # 所以这里的逻辑可能不需要额外的处理，保留原样。
                    pass # 不需要额外处理，因为 model 会返回一个合并的字典
                
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
        batch_size=128, # 批处理大小可以根据 GPU 数量和显存适当增大
        num_workers=4,
        use_amp=True
    )