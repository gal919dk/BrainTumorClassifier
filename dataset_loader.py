import os
import cv2
from torch.utils.data import Dataset

class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # בונים מילון של קטגוריות
        classes = sorted([
            cls_name for cls_name in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, cls_name))
        ])

        print(f"Detected classes: {classes}")  # נשאיר הדפסה לבדיקה
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        for cls in classes:
            cls_path = os.path.join(root_dir, cls)
            if os.path.isdir(cls_path):
                for img_name in os.listdir(cls_path):
                    img_path = os.path.join(cls_path, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ✨ נוסיף Resize פה ✨
        image = cv2.resize(image, (128, 128))

        if self.transform:
            image = self.transform(image)

        return image, label
