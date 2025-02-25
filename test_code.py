import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor,
    Resize,
    Grayscale,
    RandomRotation,
    RandomAffine,
    RandomPerspective
)
from datasets import load_dataset
from torch.utils.data import DataLoader
import json
import ast


def load_data():
    dataset = load_dataset("MohamedExperio/ICDAR2019")
    # Tạo tập các ký tự từ ground truth
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
    
    # Data augmentation: thêm RandomRotation, RandomAffine và RandomPerspective
    transform = Compose([
        Resize((32, 128)),
        Grayscale(num_output_channels=1),
        RandomRotation(degrees=2),  # xoay nhẹ
        RandomAffine(degrees=0, translate=(0.02, 0.02)),  # dịch chuyển nhẹ
        RandomPerspective(distortion_scale=0.05, p=0.5),  # hiệu ứng phối cảnh nhẹ
        ToTensor(),
        Normalize(mean=[0.5], std=[0.5])
    ])

    def apply_transforms(batch):
        batch["image"] = [transform(img) for img in batch["image"]]
        return batch

    dataset = dataset.with_transform(apply_transforms)
    trainloader = DataLoader(dataset["train"], batch_size=32, shuffle=True)
    valloader = DataLoader(dataset["validation"], batch_size=32)
    testloader = DataLoader(dataset["test"], batch_size=32)
    return trainloader, valloader, testloader, num_classes, char2idx, idx2char


class CRNN(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.3):
        super(CRNN, self).__init__()
        # Sử dụng các block convolution theo dạng Sequential để tăng khả năng trích xuất đặc trưng
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1))
        )
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv_block6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1))
        )
        self.conv_block7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )
        # Sau các conv layers, loại bỏ chiều height và hoán đổi thứ tự các chiều cho LSTM
        # Sử dụng 2 lớp LSTM song hướng để xử lý chuỗi
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate
        )
        # Fully-connected mapping từ 512 (256*2) đến số lớp ký tự
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.conv_block6(x)
        x = self.conv_block7(x)
        # Giả sử chiều height sau conv layers bằng 1, loại bỏ chiều này
        x = x.squeeze(2)  # (batch, channels, width)
        x = x.permute(0, 2, 1)  # (batch, width, channels)
        x, _ = self.lstm(x)
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


def train(net, trainloader, epochs, device, lr=0.001, clip=5.0, weight_decay=1e-5):
    net.to(device)
    # CTCLoss với blank token
    blank_idx = char2idx["<BLANK>"]
    criterion = nn.CTCLoss(blank=blank_idx, reduction='mean', zero_infinity=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    # Sử dụng CosineAnnealingLR để điều chỉnh learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
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

            # Chiều sequence được lấy từ outputs.size(1)
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
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        avg_trainloss = running_loss / len(trainloader)
        print(f"Epoch {epoch+1} Loss: {avg_trainloss:.4f}")


# Usage example
if __name__ == "__main__":
    trainloader, valloader, testloader, num_classes, char2idx, idx2char = load_data()
    net = CRNN(num_classes, dropout_rate=0.3)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    train(net, trainloader, epochs=100, device=device, lr=0.001, clip=5.0, weight_decay=1e-5)
