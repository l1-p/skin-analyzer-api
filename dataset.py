import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class MedicalDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # --- 修改点 1: 过滤杂质文件，只保留图片 ---
        # 只读取后缀为 .jpg, .jpeg, .png 的文件，排除 .txt, .md 等
        valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.PNG')
        self.images = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(valid_extensions)
        ]

        # 如果过滤后一张图片都没有，给出提示
        if len(self.images) == 0:
            raise RuntimeError(f"在 {image_dir} 中没找到任何图片文件！请检查路径。")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # 拿到核心编号 (例如 'ISIC_0000000')
        base_name = os.path.splitext(img_name)[0]

        # --- 修改点 2: 适配你真实的掩码文件名 ---
        # 根据你报错信息的提示，你的文件带了 _segmentation
        possible_mask_names = [
            base_name + '_segmentation.png',  # 匹配 ISIC_0000000_segmentation.png
            base_name + '.png',  # 备选
            base_name + '_mask.png'  # 备选
        ]

        mask_path = None
        for name in possible_mask_names:
            temp_path = os.path.join(self.mask_dir, name)
            if os.path.exists(temp_path):
                mask_path = temp_path
                break

        if mask_path is None:
            raise FileNotFoundError(
                f"无法为图片 {img_name} 找到对应的掩码文件。\n"
                f"请检查 {self.mask_dir} 中是否存在类似 {base_name}_segmentation.png 的文件。"
            )

        # 读取图像和掩码
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)

        # 掩码预处理
        mask_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
        mask = mask_transform(mask)
        mask = (mask > 0.5).float()

        return image, mask