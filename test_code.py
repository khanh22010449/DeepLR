'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, Grayscale

from datasets import load_dataset
from torch.utils.data import DataLoader

import json


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

    num_classes = len(total_chars)
    print(f"Num Class {num_classes}")

    transform = Compose(
        [
            Resize((64, 256)),
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
    return trainloader, valloader, testloader, num_classes


class OCRModel(nn.Module):
    def __init__(self, num_classes):
        super(OCRModel, self).__init__()

        # CNN Backbone (Bottom to Top as per table)
        self.cnn = nn.Sequential(
            # Input: (1, 32, W) - Gray-scale image with height 32
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2),
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(512),
        )

        # Map-to-Sequence: Convert CNN output to sequence format
        self.map_to_seq = nn.Linear(512, 256)

        # RNN: Bidirectional LSTMs
        self.rnn = nn.Sequential(
            nn.LSTM(256, 256, bidirectional=True, batch_first=True),
            nn.LSTM(512, 256, bidirectional=True, batch_first=True),
        )

        # Final Output Layer
        self.fc = nn.Linear(512, num_classes + 1)  # +1 for CTC blank

    def forward(self, x):
        # CNN Feature Extraction
        x = self.cnn(x)  # (B, C, H, W)

        # Convert to sequence format: (W, B, C)
        x = x.permute(3, 0, 1, 2)  # (W, B, C, H)
        x = x.flatten(2)  # (W, B, C*H)
        x = self.map_to_seq(x)  # (W, B, 256)

        # RNN Processing
        x, _ = self.rnn[0](x)
        x, _ = self.rnn[1](x)

        # Output Layer
        x = self.fc(x)
        return x


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
    trainloader, valloader, testloader, num_classes = load_data()
    print(trainloader, valloader, testloader)
    net = OCRModel(num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train(net, trainloader, 100, device)
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, Grayscale, ToTensor, Normalize
from datasets import load_dataset
import json
from collections import defaultdict
import numpy as np
import torch.nn.functional as F


# --------------------- Configuration ---------------------
class Config:
    batch_size = 16
    lr = 3e-5
    epochs = 30
    grad_clip = 1.0
    weight_decay = 1e-5
    warmup_epochs = 5
    image_size = (64, 256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------- Data Processing ---------------------
class CharMapper:
    def __init__(self, dataset):
        self.char_counts = defaultdict(int)
        self.special_tokens = ["<BLANK>", "<UNK>", "<SOS>", "<EOS>"]
        self._build_vocab(dataset)

    def _build_vocab(self, dataset):
        for split in ["train", "validation"]:
            for sample in dataset[split]["ground_truth"]:
                gt = json.loads(sample["gt_parses"][0])
                for field in ["company", "date", "address", "total"]:
                    text = gt[field].upper()
                    for char in text:
                        self.char_counts[char] += 1

        # Filter rare characters and build mapping
        self.chars = self.special_tokens + [
            char for char, count in self.char_counts.items() if count >= 3
        ]

        self.char2idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        self.blank_idx = self.char2idx["<BLANK>"]
        self.unk_idx = self.char2idx["<UNK>"]

    def text_to_indices(self, text):
        return [self.char2idx.get(char.upper(), self.unk_idx) for char in text]


class OCRCollator:
    def __init__(self, char_mapper, transform):
        self.char_mapper = char_mapper
        self.transform = transform

    def __call__(self, batch):
        processed = {"pixels": [], "targets": [], "target_lengths": []}

        for item in batch:
            # Process image
            img = item["image"].convert("L")  # Ensure grayscale
            img = self.transform(img)
            processed["pixels"].append(img)

            # Process text
            gt = json.loads(item["ground_truth"]["gt_parses"][0])
            text = f"{gt['company']} {gt['date']} {gt['address']} {gt['total']}".upper()
            indices = self.char_mapper.text_to_indices(text)

            processed["targets"].extend(indices)
            processed["target_lengths"].append(len(indices))

        return {
            "images": torch.stack(processed["pixels"]),
            "targets": torch.IntTensor(processed["targets"]),
            "target_lengths": torch.IntTensor(processed["target_lengths"]),
        }


# --------------------- Model Architecture ---------------------
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 1))
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1)
        self.batch_norm5 = nn.BatchNorm2d(512)
        
        self.conv6 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.batch_norm6 = nn.BatchNorm2d(512)
        self.pool6 = nn.MaxPool2d(kernel_size=(2, 1))
        
        self.conv7 = nn.Conv2d(512, 512, kernel_size=(2, 2))
        
        # Bidirectional LSTM layers
        self.lstm1 = nn.LSTM(512, 128, bidirectional=True, batch_first=True, dropout=0.2)
        self.lstm2 = nn.LSTM(256, 128, bidirectional=True, batch_first=True, dropout=0.2)
        
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
        
        # Squeeze the height dimension
        x = x.squeeze(2)  # Squeeze the height dimension (axis=2 in PyTorch)
        x = x.permute(0, 2, 1)  # Change shape to (batch, width, channels)
        
        # Bidirectional LSTM layers
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        
        # Fully connected layer
        x = self.fc(x)
        
        return F.log_softmax(x, dim=2)

# --------------------- Training Utilities ---------------------
class CTCLossWrapper(nn.CTCLoss):
    def forward(self, log_probs, targets, target_lengths):
        input_lengths = torch.full(
            size=(log_probs.size(1),),
            fill_value=log_probs.size(0),
            dtype=torch.long,
            device=log_probs.device,
        )

        # Add numerical stability
        log_probs = torch.clamp(log_probs, min=-1e4, max=1e4)

        # Verify lengths
        assert (target_lengths <= targets.size(0)).all(), "Invalid target lengths"

        return super().forward(
            log_probs, targets, input_lengths, target_lengths  # (T, N, C)  # (N, S)
        )


def validate_batch(model, batch, criterion, device):
    with torch.no_grad():
        images = batch["images"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        outputs = model(images)
        loss = criterion(outputs, targets, target_lengths)

        # Check for invalid loss
        if torch.isnan(loss) or torch.isinf(loss):
            return float("inf")
        return loss.item()


# --------------------- Main Training Loop ---------------------
def main():
    cfg = Config()

    # Load dataset and build character mapping
    dataset = load_dataset("MohamedExperio/ICDAR2019")
    char_mapper = CharMapper(dataset)

    # Create transforms
    transform = Compose(
        [
            Resize(cfg.image_size),
            Grayscale(num_output_channels=1),
            ToTensor(),
            Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    # Create data loaders
    collator = OCRCollator(char_mapper, transform)
    train_loader = DataLoader(
        dataset["train"],
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
    )

    # Initialize model
    model = CRNN(len(char_mapper.chars)).to(cfg.device)
    criterion = CTCLossWrapper(blank=char_mapper.blank_idx)
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    # Learning rate warmup
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: min(1.0, (epoch + 1) / cfg.warmup_epochs)
    )

    # Training loop
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            # Data to device
            images = batch["images"].to(cfg.device, non_blocking=True)
            targets = batch["targets"].to(cfg.device, non_blocking=True)
            target_lengths = batch["target_lengths"].to(cfg.device, non_blocking=True)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets, target_lengths)

            # Backpropagation
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            optimizer.step()

            # Loss tracking
            total_loss += loss.item()

            # Nan checks
            if torch.isnan(loss):
                print(f"NaN detected at batch {batch_idx}")
                break

            # Intermediate validation
            if batch_idx % 50 == 0:
                val_loss = validate_batch(model, batch, criterion, cfg.device)
                print(
                    f"Epoch {epoch+1} | Batch {batch_idx} | "
                    f"Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}"
                )

        # Update learning rate
        scheduler.step()

        # Epoch statistics
        avg_loss = total_loss / len(train_loader)
        print(
            f"Epoch {epoch+1}/{cfg.epochs} | Avg Loss: {avg_loss:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )


if __name__ == "__main__":
    main()
