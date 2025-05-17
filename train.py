from utils.dataset_loader import BrainTumorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import torch

# הנתיב לדאטה (איפה ה-Training יושב)
data_path = "/Users/galshemesh/Desktop/gal shemesh payton/PythonProject1/archive (1)/Training"
import os

# נבדוק מה יש בתיקייה הראשית
data_path = "/Users/galshemesh/Desktop/gal shemesh payton/PythonProject1/archive (1)/Training"

print("תיקיות שנמצאות בתוך Training:")
print(os.listdir(data_path))

# טוענים את הדאטה
dataset = BrainTumorDataset(root_dir=data_path)

# בודקים כמה תמונות יש
print(f"Number of images: {len(dataset)}")

# מחלקים את הדאטה ל-80% Train ו-20% Validation
indices = list(range(len(dataset)))
train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

# יוצרים תתי-דאטה
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

# יוצרים DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# הדפסה - כמה יש בכל חלק
print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# מציגים דוגמא אחת אם יש דאטה

import torch.nn as nn
import torch.nn.functional as F

# נגדיר רשת CNN פשוטה
class BrainTumorCNN(nn.Module):
    def __init__(self):
        super(BrainTumorCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        # התאמה לפי גודל תמונה
        self.fc2 = nn.Linear(128, 4)  # יש 4 קטגוריות: glioma, meningioma, pituitary, no tumor

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # 🛠️ שימוש בגודל אוטומטי
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# הגדרת המודל
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BrainTumorCNN().to(device)

# פונקציית איבוד
criterion = nn.CrossEntropyLoss()

# אופטימייזר
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# לולאת אימון
num_epochs = 5  # נתחיל בקטן - 5 אפוקים
for epoch in range(num_epochs):
    model.train()  # מצב אימון
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        # מעבירים ל-Device (GPU אם יש, אחרת CPU)
        images = images.permute(0, 3, 1, 2)  # להפוך מ-HWC ל-CHW
        images = images.float() / 255.0  # נרמול ל-0-1
        images = images.to(device)
        labels = labels.to(device)

        # אפס גרדיאנטים
        optimizer.zero_grad()

        # תחזית
        outputs = model(images)

        # חישוב הפסד
        loss = criterion(outputs, labels)

        # חישוב גרדיאנטים ועדכון משקלים
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # חישוב דיוק
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

print("Finished Training ✅")
# ======================================
# הערכת המודל על ה-Validation Set
# ======================================
model.eval()  # מצב הערכה, לא אימון
val_loss = 0.0
correct = 0
total = 0

with torch.no_grad():  # אין צורך בגרדיאנטים בבדיקה
    for images, labels in val_loader:
        images = images.permute(0, 3, 1, 2)
        images = images.float() / 255.0
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        val_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

avg_val_loss = val_loss / len(val_loader)
val_accuracy = 100 * correct / total

print(f"\nValidation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")


    # ======================================
    # שמירת המודל המאומן
    # ======================================
model_save_path = "brain_tumor_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"\n✅ Model saved to {model_save_path}")
