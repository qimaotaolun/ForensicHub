import json
import os
from typing import Dict, Any, List, Optional
import torch
from torch.utils.data import Dataset
from ForensicHub.core.base_dataset import BaseDataset
from ForensicHub.registry import register_dataset
from PIL import Image
import numpy as np
import csv
from pycocotools import mask as mask_util

import ast 


def rle_to_mask(rle_input):
    try:
        if isinstance(rle_input, str):
            rle_dict = ast.literal_eval(rle_input)
        elif isinstance(rle_input, dict):
            rle_dict = rle_input
        else:
            raise TypeError("输入必须是字符串或字典格式的RLE对象")

        if isinstance(rle_dict['counts'], str):
            rle_dict['counts'] = rle_dict['counts'].encode('utf-8')

        binary_mask = mask_util.decode(rle_dict)
        return binary_mask.astype(np.uint8)  # 确保为 uint8 格式
    except Exception as e:
        print(f"将RLE转换为Mask时发生错误: {e}")
        return None

@register_dataset("BisaiDatasetWithMask")
class BisaiDatasetWithMask(BaseDataset):
    def __init__(self,
                path: str,
                root_dir: str,
                image_size: int = 512,
                **kwargs):
        self.root_dir = root_dir
        self.image_size = image_size
        super().__init__(path=path, **kwargs)

    def _init_dataset_path(self) -> None:
        """Read CSV file and parse image paths, RLE masks, and labels."""
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"CSV file not found at {self.path}")

        self.samples = []
        with open(self.path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Clean up any escaped slashes in Region string
                region = row["Region"].replace("//", "////") if row["Region"] else ""
                self.samples.append({
                    "path": row["Path"],
                    "region": region,
                    "label": int(row["Label"])
                })

        self.entry_path = self.path

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        image_path = os.path.join(self.root_dir, sample["path"])
        region = sample["region"]
        label = sample["label"]

        # Load and resize image
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size))
        image = np.array(image)
        # Process mask
        if region.strip() == "":
            fill_value = 1 if label == 1 else 0
            mask = np.full((self.image_size, self.image_size), fill_value, dtype=np.uint8)
        else:
            rle = region.replace("//", "////")
            mask = rle_to_mask(rle)
            mask = Image.fromarray(mask).resize((self.image_size, self.image_size), resample=Image.NEAREST)
            mask = np.array(mask).astype(np.uint8)

        if self.common_transform:
            image = self.common_transform(image=image)['image']
        
        if self.post_transform:
            transformed = self.post_transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        mask = mask.clone().detach().long().unsqueeze(0)
    
        output = {
            "image": image,
            "label": torch.tensor(label, dtype=torch.float),
            "mask": mask
        }

        return output

from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

def main():
    # 假设你有一个数据目录和 CSV
    root_dir = "/ossfs/workspace/shenhe"  
    dataset = BisaiDataset(path='/ossfs/workspace/shenhe/train.csv',root_dir=root_dir, image_size=512)

    save_dir = "./debug_samples"  # 保存图像和mask的目录
    os.makedirs(save_dir, exist_ok=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for i, batch in enumerate(dataloader):
        image = batch["image"]      # [1, C, H, W]
        label = batch["label"]
        mask = batch["mask"]        # [1, 1, H, W]

        if mask.sum().item() > 0:
            print(f"✅ Found non-zero mask at Sample {i}")
            print(f"  Label: {label.item()}")
            print(f"  Saving to {save_dir}")

            # 去除 batch 维度
            image_tensor = image[0]  # [C, H, W]
            mask_tensor = mask[0][0] # [H, W]

            # 转为 PIL 图像
            image_pil = TF.to_pil_image(image_tensor)
            mask_np = mask_tensor.numpy().astype(np.uint8) * 255  # 放大为 0/255
            mask_pil = Image.fromarray(mask_np)

            # 保存
            image_pil.save(os.path.join(save_dir, f"sample_{i}_image.jpg"))
            mask_pil.save(os.path.join(save_dir, f"sample_{i}_mask.png"))
            import pdb; pdb.set_trace()

if __name__ == '__main__':
    main()