import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import string


# Tạo bộ dữ liệu OCR (giả lập)
class OCRDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.alphabet = string.ascii_letters + string.digits  # A-Z, a-z, 0-9
        self.char_to_idx = {char: idx for idx, char in enumerate(self.alphabet)}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        # Chuyển nhãn (chuỗi ký tự) thành chỉ số
        label_indices = [self.char_to_idx[char] for char in label]
        return image, torch.tensor(label_indices, dtype=torch.long)


# Mô hình OCR cơ bản
class OCRModel(nn.Module):
    def __init__(self, num_classes, input_channels=1):
        super(OCRModel, self).__init__()
        # Backbone CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 1/2 kích thước
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 1/4 kích thước
        )

        # Sequence Modeling (RNN - LSTM)
        self.rnn = nn.LSTM(64 * 7, 128, bidirectional=True, batch_first=True)

        # Classifier
        self.fc = nn.Linear(128 * 2, num_classes)  # Bidirectional nên x2

    def forward(self, x):
        # Trích xuất đặc trưng từ CNN
        features = self.cnn(x)  # [B, C, H, W]
        b, c, h, w = features.size()
        features = features.permute(0, 3, 1, 2)  # [B, W, C, H]
        features = features.reshape(b, w, c * h)  # [B, W, C*H]

        # Sequence Modeling
        rnn_out, _ = self.rnn(features)  # [B, W, 2*hidden_size]

        # Classifier
        output = self.fc(rnn_out)  # [B, W, num_classes]
        return output


# Hàm huấn luyện
def train(model, dataloader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Dự đoán
            outputs = model(images)  # [B, W, num_classes]

            # Tính CTC Loss
            input_lengths = torch.full(
                size=(images.size(0),), fill_value=outputs.size(1), dtype=torch.long
            )
            target_lengths = torch.tensor(
                [len(label) for label in labels], dtype=torch.long
            )
            outputs = outputs.log_softmax(2)  # CTC yêu cầu log_softmax
            loss = criterion(outputs, labels, input_lengths, target_lengths)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")


# Tạo dữ liệu giả lập
def generate_dummy_data(num_samples=1000):
    images = []
    labels = []
    alphabet = string.ascii_letters + string.digits
    for _ in range(num_samples):
        # Tạo ảnh trắng (28x28) và nhãn ngẫu nhiên
        img = Image.fromarray(np.random.randint(0, 255, (28, 28), dtype=np.uint8))
        label = "".join(np.random.choice(list(alphabet), size=np.random.randint(1, 5)))
        images.append(img)
        labels.append(label)
    return images, labels


# Main
if __name__ == "__main__":
    # Tạo dữ liệu
    images, labels = generate_dummy_data()
    transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = OCRDataset(images, labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Khởi tạo mô hình
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OCRModel(num_classes=len(string.ascii_letters + string.digits)).to(device)

    # Cấu hình huấn luyện
    criterion = nn.CTCLoss(blank=0)  # Ký tự trống (blank) để CTC Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Huấn luyện
    train(model, dataloader, optimizer, criterion, epochs=5)
