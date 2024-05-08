import sys
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

class CustomDataset(Dataset):
    def __init__(self, image_path, mask_path, transform=None):
        self.images = np.load(image_path)  # Load the image data (55000 x 12288)
        self.masks = np.load(mask_path)    # Load the mask data (55000 x 4096)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index].reshape(64, 64, 3).astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min())
        
        # stack from (64,64,11) to (64,64)
        mask = np.zeros((64, 64, 11), dtype=np.float32)

        data = self.masks[index].reshape(64, 64)

        # Fill the one-hot encoded array based on class labels
        for i in range(11):
            # Now each mask will be 64 x 64 x 11
            mask[:, :, i] = (data == i).astype(np.float32)



        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            # self, in_channels=3, out_channels=2, features=[64, 128, 256, 512],
            # self, in_channels=3, out_channels=10, features=[64, 128, 256, 512],
            self, in_channels=3, out_channels=11, features=[64, 128, 256, 512],

    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        # This will map 3 to 64 channels, then 64 to 128, then 128 to 256, then 256 to 512
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    # we double it becasue we need to make space for the residual
                    # we start from 1028 to 512, 512 to 256, 256 to 128, 128 to 64
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            # After we go up, then we go through two convolutional neural network
            self.ups.append(DoubleConv(feature*2, feature))

        # This is the lowest point of the UNET
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # This is the last stage
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # After each layer of conv, we need to save the result. We will use this
        # saved result to add it to the other end.
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "./mnistdd_rgb_train_valid/train_X.npy"
TRAIN_MASK_DIR = "./mnistdd_rgb_train_valid/train_seg.npy"
VAL_IMG_DIR = "./mnistdd_rgb_train_valid/valid_X.npy"
VAL_MASK_DIR = "./mnistdd_rgb_train_valid/valid_seg.npy"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            targets = targets.reshape(targets.shape[0], targets.shape[2], targets.shape[3], 11)
            targets = targets.argmax(dim=3)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

def save_checkpoint(state, filename="my_checkpoint_vscode.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = CustomDataset(
        image_path=train_dir,
        mask_path=train_maskdir,
        transform = train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = CustomDataset(
        image_path=val_dir,
        mask_path=val_maskdir,
        transform = train_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            # Forward pass
            logits = model(x)
            preds = logits.permute(0, 2, 3, 1)
            argmax_indices = torch.argmax(preds, dim= -1)
            one_hot = torch.zeros_like(preds)
            one_hot.scatter_(-1, argmax_indices.unsqueeze(-1), 1)
            preds = one_hot

            num_correct += (preds == y).sum().item()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()



def save_predictions_as_imgs(loader, model, folder="./saved_images", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            logits = model(x) #logits.shape is  torch.Size([16, 11, 64, 64])
            preds = torch.argmax(logits, dim=1)  # Get predicted class labels #preds1 shape is torch.Size([16, 64, 64])

        print(torch.max(preds))
        print(torch.min(preds))

        for i in range(preds.size(0)):

            a = preds[i].float()
            a = a/10

            folder = "./saved_images"
            torchvision.utils.save_image(a, f"{folder}/pred_{idx}_{i}.png")

            # saving the numpy array that is Nx4096


            # z = y[i].float()


            # reconstructed_data_tensor = (z * torch.arange(11, dtype=torch.float32)).sum(2)
            # reconstructed_data_tensor = reconstructed_data_tensor/10



def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    # This is important
    model = UNET(in_channels=3, out_channels=11).to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth_vscode.tar"), model)


    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images", device=DEVICE
        )

if __name__ == "__main__":
    main()