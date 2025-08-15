import timm
import torch 
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from transformers import UperNetPreTrainedModel,UperNetForSemanticSegmentation,UperNetConfig
from ForensicHub.registry import register_model

@register_model("BisaiBaseline")
class BisaiBaseline(UperNetPreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        config.num_labels = 1
        # 初始时不创建 transformer，因为 from_pretrained 会直接赋值
        self.transformer = None # 或直接不初始化，因为它会在 from_pretrained 中被覆盖
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
        # 获取设备信息，用于加载预训练模型
        # 这里需要注意，from_pretrained 可能会在多个场景下调用
        # 如果是用于推理，你可能希望它直接加载到 CUDA 设备上
        # 如果是用于训练，且使用DDP，则每个进程会单独加载
        
        # 在这里我们假设你希望将其加载到默认的 CUDA 设备，
        # 如果后续会调用 .to(device) 或 DataParallel，这是一个合理的起点。
        # 对于多GPU推理，通常加载到主设备'cuda:0'
        
        device = kwargs.pop("device", None) # 尝试从 kwargs 获取设备参数
        if device is None and torch.cuda.is_available():
            device = torch.device("cuda") 
        elif device is None:
            device = torch.device("cpu")
            
        print(f"Loading UperNetForSemanticSegmentation from '{pretrained_model_name_or_path}' with pretrained weights to {device}...")
        
        # 1. 直接将预训练模型加载到指定设备
        # transformers 库的 from_pretrained 方法通常支持 map_location
        # 或者在较新版本中，直接通过 .to(device) 链式调用
        upernet_model = UperNetForSemanticSegmentation.from_pretrained(
            pretrained_model_name_or_path, 
            *model_args, 
            **kwargs
        ).to(device) # <--- 关键改动：在这里立即移动到设备
        
        print("UperNetForSemanticSegmentation loaded successfully and moved to device.")

        # 2. 获取加载模型的配置。
        config = upernet_model.config

        # 3. 创建 BisaiBaseline 的实例。
        # 注意：这里 __init__ 不再创建 transformer，因为我们将直接赋值
        model = cls(config)

        # 4. 将第一步加载的、带有预训练权重的 upernet_model 赋值给 BisaiBaseline 实例的 self.transformer。
        # 此时 upernet_model 已经位于正确的设备上
        model.transformer = upernet_model
        print("Pretrained UperNetForSemanticSegmentation assigned to self.transformer.")
        
        return model

    def forward(self, image, mask, label, *args, **kwargs):
        # import pdb; pdb.set_trace()
        label = label.float()  # [B] or [B,1]
        B = image.size(0)
        
        # Step 1: backbone forward
        outputs= self.transformer(image)  # feat: [B, C, H/32, W/32]
        # print(outputs.logits.shape)

        # # Step 2: local head — 只对 mask 有效样本计算
        pred_masks = outputs.logits  # [B, 1, H, W]
        loss_all = F.binary_cross_entropy_with_logits(
            pred_masks, mask.float(), reduction='none'  # [B, 1, H, W]
        )  # 每个像素的 loss
        
        # # Step 3: detect head — 全部参与
        # pred_logits = self.detect_head(out_detect).squeeze(dim=1)  # [B]
        # loss_label = F.binary_cross_entropy_with_logits(pred_logits, label)
        
        # Step 2: 平均成样本级别 [B]
        loss_per_sample = loss_all.view(loss_all.size(0), -1).mean(dim=1)  # [B]

        # Step 3: 筛选有效样本
        mask_valid = (mask.sum(dim=[1, 2, 3]) > 0).float()  # [B]
        num_valid = mask_valid.sum()
        loss_mask = (loss_per_sample * mask_valid).sum() / (num_valid + 1e-6)

        return {
            "backward_loss": loss_mask,
            # "backward_loss": loss_label + loss_mask,
            'pred_mask' : pred_masks,
            'pred_label': label,
            "visual_loss": {
                # "loss_label": loss_label,
                "loss_label": loss_mask,
                "loss_mask": loss_mask,
            }
        }
    
    def save_only_transformer(self, output_dir):
        self.transformer.save_pretrained(output_dir)
    
    def load_only_transformer(self, input_dir):
        self.transformer = UperNetForSemanticSegmentation.from_pretrained(input_dir).to(self.device)

def main():
    import torch
    import random
    
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # config = UperNetConfig.from_pretrained("openmmlab/upernet-convnext-large")
    # model = BisaiBaseline(config).to(device)
    model = BisaiBaseline.from_pretrained("openmmlab/upernet-convnext-large",num_labels=1,ignore_mismatched_sizes=True,device=device)
    
    # model.save_only_transformer("forensichub_checkpoint")
    model.load_only_transformer("forensichub_checkpoint")
    
    model.eval()

    B, C, H, W = 4, 3, 512, 512  # Batch size, channels, image size

    # 随机生成图像数据
    image = torch.randn(B, C, H, W)

    # label 都设为 1（或你想要的值）
    label = torch.ones(B)

    # 构造 mask：一部分是随机掩码，一部分是全 0
    mask = torch.zeros(B, 1, H, W)
    for i in range(B):
        if random.random() > 0.5:
            mask[i, 0] = torch.randint(0, 2, (H, W)).float()

    # 执行前向传播
    with torch.no_grad():
        out = model(image.to(device), mask.to(device), label.to(device))
        
    # torch.onnx.export(
    #     model=model,
    #     args=(image.to(0), mask.to(0), label.to(0)),
    #     f="forensic.onnx",
    #     input_names=["image","mask", "label"],
    #     output_names=["backward_loss", "pred_mask", "pred_label","loss_label", "loss_mask:"],
    #     dynamic_axes={
    #         "input": {0: "batch_size"},
    #         "output": {0: "batch_size"},
    #     },
    # )
    
    # 打印结果
    print("== Forward Results ==")
    print(f"backward_loss: {out['backward_loss'].item():.4f}")
    print(f"loss_label:    {out['visual_loss']['loss_label'].item():.4f}")
    print(f"loss_mask:     {out['visual_loss']['loss_mask'].item():.4f}")
    
    print(f"pred_mask:     {out['pred_mask'].shape}")
    
    print(f"param")
    num_params = count_parameters(model)
    print(f"模型的总可训练参数量: {num_params}")
    # 或者以百万为单位打印
    print(f"模型的总可训练参数量: {num_params / 1e6:.2f} M")
        
def count_parameters(model):
    """
    计算模型中可训练参数的总量。
    """
    return sum(p.numel() for p in model.parameters())
    # return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    main()