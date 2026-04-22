import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm
from PIL import Image
import numpy as np

from dataset import MedicalDataset
from unet import UNet_DTC_PraNet

# --- 超参数与路径设置 ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
EPOCHS = 200
IMAGE_DIR = r"E:\jishe\unet\data\images"
MASK_DIR = r"E:\jishe\unet\data\masks"
CHECKPOINT_DIR = r"E:\jishe\unet\checkpoints"
# 结果输出路径
RESULT_SAVE_DIR = r"E:\jishe\unet\best_results_red"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def calculate_dice(predictions, targets, smooth=1e-8):
    preds = torch.sigmoid(predictions)
    preds = (preds > 0.5).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()


def save_best_20_results(model, dataset, device, output_dir):
    """
    保存前20张分割结果：红底黑图，等比例还原，不压缩变形
    """
    model.eval()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"--- 正在保存前20张最优分割图 (红色提取) ---")

    # 获取前20张图片的列表
    indices = range(min(20, len(dataset.images)))

    with torch.no_grad():
        for i in indices:
            img_name = dataset.images[i]
            img_path = os.path.join(dataset.image_dir, img_name)

            # 1. 加载原图并记录尺寸，解决压缩问题的核心：Letterbox
            img_org = Image.open(img_path).convert("RGB")
            w_org, h_org = img_org.size

            # 计算缩放比例，保持长宽比
            ratio = 256 / max(w_org, h_org)
            new_w, new_h = int(w_org * ratio), int(h_org * ratio)
            img_resized = img_org.resize((new_w, new_h), Image.BILINEAR)

            # 填充到 256x256 黑色背景中心
            input_img = Image.new("RGB", (256, 256), (0, 0, 0))
            pad_left = (256 - new_w) // 2
            pad_top = (256 - new_h) // 2
            input_img.paste(img_resized, (pad_left, pad_top))

            # 2. 推理
            img_tensor = TF.to_tensor(input_img).unsqueeze(0).to(device)
            # 根据模型结构，返回 coarse, mid, fine，取最精细的 fine
            pred_fine = model(img_tensor)

            # 3. 后处理生成掩码
            mask_probs = torch.sigmoid(pred_fine).squeeze().cpu().numpy()
            mask_binary = (mask_probs > 0.5).astype(np.uint8)

            # 4. 还原尺寸：先裁剪掉黑边，再缩放回原图大小
            mask_256 = Image.fromarray(mask_binary * 255)  # 临时转为255方便裁剪
            mask_cropped = mask_256.crop((pad_left, pad_top, pad_left + new_w, pad_top + new_h))
            mask_final_binary = mask_cropped.resize((w_org, h_org), Image.NEAREST)

            # 5. 生成红底黑图 (RGB模式)
            # 创建纯黑背景
            red_result = Image.new("RGB", (w_org, h_org), (0, 0, 0))
            # 创建纯红颜色层
            red_layer = Image.new("RGB", (w_org, h_org), (255, 0, 0))
            # 使用二值化掩码作为 Alpha 通道将红色复合到黑色背景上
            red_result = Image.composite(red_layer, red_result, mask_final_binary)

            # 6. 保存，文件名保持一致
            red_result.save(os.path.join(output_dir, img_name))


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    total_loss = 0
    total_dice = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)

        with torch.cuda.amp.autocast():
            # 模型输出三个层级的预测
            p1, p2, p3 = model(data)
            loss = loss_fn(p1, targets) + loss_fn(p2, targets) + loss_fn(p3, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        dice = calculate_dice(p3, targets)
        total_loss += loss.item()
        total_dice += dice

        loop.set_postfix(loss=loss.item(), dice=dice)

    return total_loss / len(loader), total_dice / len(loader)


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 训练时的 transform (依然保持 256x256，但注意这里可能会有轻微变形，
    # 建议在 dataset.py 中也同步改为等比例填充，此处保持你原始逻辑)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
    ])

    dataset = MedicalDataset(image_dir=IMAGE_DIR, mask_dir=MASK_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    model = UNet_DTC_PraNet(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    best_dice = 0.0

    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
        model.train()

        avg_loss, avg_dice = train_fn(loader, model, optimizer, loss_fn, scaler)
        print(f"Avg Loss: {avg_loss:.4f}, Avg Dice: {avg_dice:.4f}")

        # 核心逻辑：只有当准确度（Dice）创下新高时，才更新保存那 20 张图
        if avg_dice > best_dice:
            best_dice = avg_dice
            print(f"*** 检测到性能提升，正在更新最优模型与 Top20 分割图 ***")
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))

            # 调用自定义保存函数
            save_best_20_results(model, dataset, DEVICE, RESULT_SAVE_DIR)


if __name__ == "__main__":
    main()