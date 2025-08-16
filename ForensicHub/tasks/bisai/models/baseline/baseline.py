import timm
import torch 
import math
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from transformers import UperNetPreTrainedModel,UperNetForSemanticSegmentation,UperNetConfig
from ForensicHub.registry import register_model

def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (torch.Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        torch.Tensor: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb

class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        out_dim: int = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = nn.SiLU()

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        sample = self.act(sample)

        sample = self.linear_2(sample)
        sample = self.act(sample)
        return sample
    
@register_model("BisaiBaseline")
class BisaiBaseline(UperNetPreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        # Timestep 
        decoder_channels = [1536,768,384,192] # [192, 384, 768, 1536]
        self.time_proj = Timesteps(decoder_channels[0], flip_sin_to_cos=True, downscale_freq_shift=0)
        timestep_input_dim = decoder_channels[0]
        self.time_embedding = TimestepEmbedding(timestep_input_dim, decoder_channels[0])
        config.num_labels = 1
        self.transformer = UperNetForSemanticSegmentation(config)
        
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
        # 1. 首先加载 UperNetForSemanticSegmentation 模型及其预训练权重。
        # 这一步会自动处理下载和权重加载，并将参数映射到 UperNetForSemanticSegmentation 的内部结构。
        print(f"Loading UperNetForSemanticSegmentation from '{pretrained_model_name_or_path}' with pretrained weights...")
        upernet_model = UperNetForSemanticSegmentation.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        print("UperNetForSemanticSegmentation loaded successfully.")

        # 2. 获取加载模型的配置。
        config = upernet_model.config

        # 3. 创建 BisaiBaseline 的实例。
        # 此时，BisaiBaseline.__init__ 会被调用，并创建一个随机初始化的 self.transformer
        model = cls(config)

        # 4. 将第一步加载的、带有预训练权重的 upernet_model 赋值给 BisaiBaseline 实例的 self.transformer。
        # 这样，所有 upernet_model 的参数就都被“嵌套”在 BisaiBaseline 的 'transformer.' 命名空间下。
        model.transformer = upernet_model

        print("Pretrained UperNetForSemanticSegmentation assigned to self.transformer.")
        
        return model

    def forward(self, image, mask, label, *args, **kwargs):
        # 1. time
        timesteps = 0.222
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=image.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(image.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        if image.shape[0]!=1:
            timesteps = timesteps * torch.ones(image.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        # t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)
        
        # import pdb; pdb.set_trace()
        label = label.float()  # [B] or [B,1]
        B = image.size(0)
        
        # Step 1: backbone forward
        outputs= self.transformer(image,emb,output_hidden_states=True)  # feat: [B, C, H/32, W/32]
        # print(outputs.logits.shape)

        # Step 2: local head — 只对 mask 有效样本计算
        pred_masks = outputs.logits  # [B, 1, H, W]
        loss_all = F.binary_cross_entropy_with_logits(
            pred_masks, mask.float(), reduction='none'  # [B, 1, H, W]
        )  # 每个像素的 loss

        
        # # Step 3: detect head — 全部参与
        # pred_logits = self.detect_head(feat).squeeze(dim=1)  # [B]
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
            'pred_label': pred_masks,
            "visual_loss": {
                # "loss_label": loss_label,
                "loss_label": loss_mask,
                "loss_mask": loss_mask,
            }
        }
    
    def _init_time_related_weights(self):
        """
        初始化与时间相关的层（bn_t 和 time_emb_proj）的参数。
        这些层在加载原始 UperNetForSemanticSegmentation 检查点时通常是未初始化的。
        """
        # 定义需要初始化的层名称模式
        bn_t_names = ["bn_t.bias", "bn_t.weight"]
        time_emb_proj_names = ["time_emb_proj.bias", "time_emb_proj.weight"]
        
        # 记录均值和方差的名称，这些通常由BatchNorm层自动管理，但在初始化时需要归零
        bn_t_running_stats_names = ["bn_t.running_mean", "bn_t.running_var", "bn_t.num_batches_tracked"]

        print("Initializing newly added time-related weights...")

        for name, module in self.named_modules():
            # 检查是否是 BatchNorm 层，并初始化其 running_mean, running_var, num_batches_tracked
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                if any(bn_name in name for bn_name in ["bn_t"]): # 匹配 bn_t 层
                    if module.running_mean is not None:
                        nn.init.zeros_(module.running_mean)
                    if module.running_var is not None:
                        nn.init.ones_(module.running_var) # 方差通常初始化为1
                    if hasattr(module, 'num_batches_tracked') and module.num_batches_tracked is not None:
                        module.num_batches_tracked.zero_()
                    
                    # 对于可学习的参数（affine=True时），通常按照标准BN初始化
                    if module.affine:
                        nn.init.ones_(module.weight) # weight (gamma) 初始化为1
                        nn.init.zeros_(module.bias)  # bias (beta) 初始化为0
                    # print(f"Initialized BatchNorm layer: {name}")

            # 检查是否是 Linear 层（time_emb_proj）
            if isinstance(module, nn.Linear):
                if any(proj_name in name for proj_name in ["time_emb_proj"]): # 匹配 time_emb_proj
                    # 权重使用正态分布初始化
                    nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
                    # 偏差初始化为零
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                    # print(f"Initialized Linear layer: {name}")

        print("Finished initializing time-related weights.")
    
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
    
    config = UperNetConfig.from_pretrained("openmmlab/upernet-convnext-large")
    model = BisaiBaseline(config).to(device)
    # model = BisaiBaseline.from_pretrained("openmmlab/upernet-convnext-large",num_labels=1,ignore_mismatched_sizes=True).to(device)
    
    # model.save_only_transformer("forensichub_checkpoint")
    model.load_only_transformer("forensichub_checkpoint")
    
    model._init_time_related_weights()

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
    # print(f"model: {model}")
        
def count_parameters(model):
    """
    计算模型中可训练参数的总量。
    """
    # return sum(p.numel() for p in model.parameters()) # 计算所有参数
    return sum(p.numel() for p in model.parameters() if p.requires_grad) # 只计算需要梯度的参数

if __name__ == '__main__':
    main()