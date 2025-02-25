import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, Grayscale, RandomRotation, RandomAffine

from datasets import load_dataset
from torch.utils.data import DataLoader

import json
import ast


def load_data():
    dataset = load_dataset("MohamedExperio/ICDAR2019")
    # Tạo tập ký tự
    chars = set()
    for sample in dataset["train"]["ground_truth"]:
        gt = json.loads(sample["gt_parses"][0])
        for field in ["company", "date", "address", "total"]:
            chars.update(gt[field].upper())

    special_chars = {"<BLANK>", "<UNK>"}  # CTC blank + unknown
    total_chars = chars.union(special_chars)
    
    char2idx = {char: idx for idx, char in enumerate(sorted(total_chars))}
    idx2char = {idx: char for char, idx in char2idx.items()}

    num_classes = len(total_chars)
    
    # Áp dụng data augmentation
    transform = Compose(
        [
            Resize((32, 128)),
            Grayscale(num_output_channels=1),
            RandomRotation(degrees=2),  # xoay nhẹ để tăng tính đa dạng
            RandomAffine(degrees=0, translate=(0.02, 0.02)),  # dịch chuyển nhẹ
            ToTensor(),
            Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    def apply_transforms(batch):
        batch["image"] = [transform(img) for img in batch["image"]]
        return batch

    dataset = dataset.with_transform(apply_transforms)

    trainloader = DataLoader(
        dataset["train"], batch_size=32, shuffle=True
    )

    valloader = DataLoader(dataset["validation"], batch_size=32)
    testloader = DataLoader(dataset["test"], batch_size=32)
    return trainloader, valloader, testloader, num_classes, char2idx, idx2char


class CRNN(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.2):
        super(CRNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 1))
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.batch_norm5 = nn.BatchNorm2d(512)
        
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batch_norm6 = nn.BatchNorm2d(512)
        self.pool6 = nn.MaxPool2d(kernel_size=(2, 1))
        
        self.conv7 = nn.Conv2d(512, 512, kernel_size=2)
        self.dropout_conv = nn.Dropout2d(dropout_rate)
        
        # Bidirectional LSTM layers
        # LSTM có dropout giữa các lớp (áp dụng nếu num_layers > 1)
        self.lstm1 = nn.LSTM(512, 128, bidirectional=True, batch_first=True, dropout=dropout_rate)
        self.lstm2 = nn.LSTM(256, 128, bidirectional=True, batch_first=True, dropout=dropout_rate)
        self.dropout_lstm = nn.Dropout(dropout_rate)
        
        # Fully connected layer
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        
        x = F.relu(self.conv5(x))
        x = self.batch_norm5(x)
        
        x = F.relu(self.conv6(x))
        x = self.batch_norm6(x)
        x = self.pool6(x)
        
        x = F.relu(self.conv7(x))
        x = self.dropout_conv(x)
        
        # Squeeze chiều height và đổi thứ tự các chiều: (batch, width, channels)
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        
        # Bidirectional LSTM layers
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.dropout_lstm(x)
        
        x = self.fc(x)
        
        return F.log_softmax(x, dim=2)


def process_ground_truth(batch_gt, char2idx):
    """Chuyển đổi ground truth sang các chỉ số số học"""
    all_targets = []
    all_lengths = []

    gt_parses_list = batch_gt["gt_parses"]

    for item in gt_parses_list:
        if isinstance(item, tuple):
            parse_strings = item
        elif isinstance(item, str):
            parse_strings = [item]
        else:
            continue

        for parse_str in parse_strings:
            try:
                parse = json.loads(parse_str)
                text = parse.get("company", "")
                indices = [char2idx.get(c, char2idx["<UNK>"]) for c in text.upper()]
                all_targets.extend(indices)
                all_lengths.append(len(indices))
            except (json.JSONDecodeError, ValueError, SyntaxError):
                all_targets.extend([char2idx["<BLANK>"]])
                all_lengths.append(1)

    return torch.IntTensor(all_targets), torch.IntTensor(all_lengths)


def train(net, trainloader, epochs, device, lr=0.001, clip=5.0):
    net.to(device)
    # Sử dụng CTCLoss với blank token
    blank_idx = char2idx["<BLANK>"]
    criterion = nn.CTCLoss(blank=blank_idx, reduction='mean', zero_infinity=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    net.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for batch in trainloader:
            images = batch["image"].to(device)
            targets, target_lengths = process_ground_truth(batch["ground_truth"], char2idx)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()
            outputs = net(images)  # outputs có shape (batch, sequence_length, num_classes)

            # Sử dụng outputs.size(1) là chiều sequence
            input_lengths = torch.full(
                size=(images.size(0),), 
                fill_value=outputs.size(1), 
                dtype=torch.long
            ).to(device)

            loss = criterion(
                outputs.log_softmax(2).permute(1, 0, 2),  # (T, N, C)
                targets,
                input_lengths,
                target_lengths
            )

            loss.backward()
            # Áp dụng gradient clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        avg_trainloss = running_loss / len(trainloader)
        print(f"Epoch {epoch+1} Loss: {avg_trainloss:.4f}")


# Usage example
if __name__ == "__main__":
    trainloader, valloader, testloader, num_classes, char2idx, idx2char = load_data()
    net = CRNN(num_classes, dropout_rate=0.2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    train(net, trainloader, epochs=100, device=device, lr=0.001)
