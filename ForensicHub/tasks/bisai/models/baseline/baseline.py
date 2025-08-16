import timm
import torch 
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
# from ForensicHub.registry import register_model

class ConvNeXt(timm.models.convnext.ConvNeXt):
    def __init__(self,conv_pretrain=True):
        super(ConvNeXt, self).__init__(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536])
        if conv_pretrain:
            print("Load Convnext pretrain.And with Unet decoder.")
            model = timm.create_model('convnext_large', pretrained=False)
            self.load_state_dict(model.state_dict())

    def forward_features(self, x):
        x = self.stem(x)
        out = []
        for stage in self.stages:
            x = stage(x)
            out.append(x)
        x = self.norm_pre(x)
        return x , out

class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)
        super(Conv2dReLU, self).__init__(conv, bn, relu)

class Attention(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)

class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = Attention(
            attention_type, in_channels=in_channels + skip_channels
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

class UnetDecoder(nn.Module):
    def __init__(
        self,
        decoder_channels=[768, 384, 192, 96, 1],
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        dims=[96, 192, 384, 768, 1536]
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )
        # reverse channels to start from head of encoder
        dims = dims[::-1]

        # computing blocks input and output channels
        head_channels =dims[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(dims[1:-1]) + [0,0]
        out_channels = decoder_channels
        self.center = nn.Identity()
        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):
        # features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x

# class UpsampleLocalHead(nn.Module):
#     def __init__(self, in_channels, out_channels=1, up_factor=32):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(in_channels // 2)
#         self.relu = nn.ReLU(inplace=True)
#         self.out_conv = nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)

#         self.up_factor = up_factor

#     def forward(self, x):
#         x = self.relu(self.bn1(self.conv1(x)))
#         x = self.out_conv(x)
#         x = F.interpolate(x, scale_factor=self.up_factor, mode='bilinear', align_corners=False)
#         return x


class DetectHead(nn.Module):
    def __init__(self, in_channels, hidden_dim=256, out_dim=1):
        super().__init__()
        self.pool_avg = nn.AdaptiveAvgPool2d(1)
        self.pool_max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        print(x.shape)
        avg = self.pool_avg(x).flatten(1)
        max_ = self.pool_max(x).flatten(1)
        feat = torch.cat([avg, max_], dim=1)
        return self.mlp(feat)

# @register_model("BisaiBaseline")
class BisaiBaseline(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.backbone = ConvNeXt()
        # self.local_head = UpsampleLocalHead(in_channels=1536, out_channels=1, up_factor=32)
        self.local_head = UnetDecoder()
        self.detect_head = DetectHead(in_channels=1536, hidden_dim=256, out_dim=1)


    def forward(self, image, mask, label, *args, **kwargs):
        # import pdb; pdb.set_trace()
        label = label.float()  # [B] or [B,1]
        B = image.size(0)
        
        # Step 1: backbone forward
        feat, features = self.backbone.forward_features(image)  # feat: [B, C, H/32, W/32]

        # Step 2: detect head — 全部参与
        pred_logits = self.detect_head(feat).squeeze(dim=1)  # [B]
        loss_label = F.binary_cross_entropy_with_logits(pred_logits, label)

        # # Step 3: local head — 只对 mask 有效样本计算
        pred_masks = self.local_head(*features)
        loss_all = F.binary_cross_entropy_with_logits(
            pred_masks, mask.float(), reduction='none'  # [B, 1, H, W]
        )  # 每个像素的 loss

        # Step 2: 平均成样本级别 [B]
        loss_per_sample = loss_all.view(loss_all.size(0), -1).mean(dim=1)  # [B]

        # Step 3: 筛选有效样本
        mask_valid = (mask.sum(dim=[1, 2, 3]) > 0).float()  # [B]
        num_valid = mask_valid.sum()
        loss_mask = (loss_per_sample * mask_valid).sum() / (num_valid + 1e-6)

        return {
            "backward_loss": loss_label + loss_mask,
            'pred_mask' : pred_masks,
            'pred_label': pred_logits,
            "visual_loss": {
                "loss_label": loss_label,
                "loss_mask": loss_mask,
            }
        }

def main():
    import torch
    import random
    random.seed(42)
    model = BisaiBaseline().to(0)
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
        out = model(image.to(0), mask.to(0), label.to(0))
        
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

if __name__ == '__main__':
    main()