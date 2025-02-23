import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, Grayscale

from datasets import load_dataset
from torch.utils.data import DataLoader

import json

from warpctc_pytorch import CTCLoss


def load_data():
    dataset = load_dataset("MohamedExperio/ICDAR2019")
    # print(dataset)

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
    print(f"Num Class {num_classes}")

    transform = Compose(
        [
            Resize((32, 128)),
            Grayscale(num_output_channels=1),  # Chuyển sang grayscale
            ToTensor(),
            Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    def apply_transforms(batch):
        batch["image"] = [transform(img) for img in batch["image"]]
        return batch

    dataset = dataset.with_transform(apply_transforms)

    # for x in dataset["train"]["image"]:
    #     print(x.size())

    trainloader = DataLoader(
        dataset["train"], batch_size=32, num_workers=8, shuffle=True
    )

    # for batch in trainloader:
    #     print(batch["image"].size())

    valloader = DataLoader(dataset["validation"], batch_size=32, num_workers=8)
    testloader = DataLoader(dataset["test"], batch_size=32, num_workers=8)
    return trainloader, valloader, testloader, num_classes, char2idx, idx2char


# class CRNN(nn.Module):
#     def __init__(self, num_classes):
#         super(CRNN, self).__init__()
        
#         # Convolutional layers
#         self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1)
#         self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
#         self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
#         self.conv4 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
#         self.pool4 = nn.MaxPool2d(kernel_size=(2, 1))
        
#         self.conv5 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1)
#         self.batch_norm5 = nn.BatchNorm2d(512)
        
#         self.conv6 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
#         self.batch_norm6 = nn.BatchNorm2d(512)
#         self.pool6 = nn.MaxPool2d(kernel_size=(2, 1))
        
#         self.conv7 = nn.Conv2d(512, 512, kernel_size=(2, 2))
        
#         # Bidirectional LSTM layers
#         self.lstm1 = nn.LSTM(512, 128, bidirectional=True, batch_first=True, dropout=0.2)
#         self.lstm2 = nn.LSTM(256, 128, bidirectional=True, batch_first=True, dropout=0.2)
        
#         # Fully connected layer
#         self.fc = nn.Linear(256, num_classes)
    
#     def forward(self, x):
#         # Convolutional layers
#         x = F.relu(self.conv1(x))
#         x = self.pool1(x)
        
#         x = F.relu(self.conv2(x))
#         x = self.pool2(x)
        
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.conv4(x))
#         x = self.pool4(x)
        
#         x = F.relu(self.conv5(x))
#         x = self.batch_norm5(x)
        
#         x = F.relu(self.conv6(x))
#         x = self.batch_norm6(x)
#         x = self.pool6(x)
        
#         x = F.relu(self.conv7(x))
        
#         # Squeeze the height dimension
#         x = x.squeeze(2)  # Squeeze the height dimension (axis=2 in PyTorch)
#         x = x.permute(0, 2, 1)  # Change shape to (batch, width, channels)
        
#         # Bidirectional LSTM layers
#         x, _ = self.lstm1(x)
#         x, _ = self.lstm2(x)
        
#         # Fully connected layer
#         x = self.fc(x)
        
#         return F.log_softmax(x, dim=2)


def process_ground_truth(batch_gt, char2idx):
    """Convert text labels to numerical indices"""
    all_targets = []
    all_lengths = []

    for gt in batch_gt:
        text = json.loads(gt["gt_parses"][0])["company"]  # Example field
        indices = [char2idx.get(c, char2idx["<UNK>"]) for c in text.upper()]
        all_targets.extend(indices)
        all_lengths.append(len(indices))

    return (torch.IntTensor(all_targets), torch.IntTensor(all_lengths))


def train(net, trainloader, epochs, device, lr=0.01):
    net.to(device)
    # FIX 1: Use CTCLoss instead of CrossEntropyLoss and instantiate properly
    criterion = nn.CTCLoss(blank=net.fc.out_features - 1)  # Last index is blank
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for batch in trainloader:
            images = batch["image"].to(device)

            # FIX 2: Process ground truth texts to CTC format
            targets, target_lengths = process_ground_truth(
                batch["ground_truth"], char2idx  # Need to implement character mapping
            )
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()
            outputs = net(images)  # (W, B, C)

            # FIX 3: Prepare CTC inputs
            input_lengths = torch.full(
                size=(images.size(0),), fill_value=outputs.size(0), dtype=torch.long
            ).to(device)

            loss = criterion(
                outputs.log_softmax(2).permute(1, 0, 2),  # (T, N, C) -> (N, T, C)
                targets,
                input_lengths,
                target_lengths,
            )

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_trainloss = running_loss / len(trainloader)
        print(f"Epoch {epoch+1} Loss: {avg_trainloss:.4f}")


# Usage example
if __name__ == "__main__":
    trainloader, valloader, testloader, num_classes, char2idx, idx2char = load_data()
    print(trainloader, valloader, testloader)
    net = CRNN(num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # train(net, trainloader, 100, device)
