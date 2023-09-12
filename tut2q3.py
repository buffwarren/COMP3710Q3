#!/usr/bin/env python3
#/Users/khainguyentri/venv/bin/


import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Ensure any idle GPU memory is cleared
torch.cuda.empty_cache()

# Paths
seg_testing_path = "keras_png_slices_data/keras_png_slices_seg_test"
seg_training_path = "keras_png_slices_data/keras_png_slices_seg_train"
seg_validation_path = "keras_png_slices_data/keras_png_slices_seg_validate"

testing_path = "keras_png_slices_data/keras_png_slices_test"
training_path = "keras_png_slices_data/keras_png_slices_train"
validation_path = "keras_png_slices_data/keras_png_slices_validate"

# Define your dataset
class MRIDataset(Dataset):
    def __init__(self, img_dir, seg_dir, transform=None):
        self.img_dir = img_dir
        self.seg_dir = seg_dir
        self.transform = transform
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.images[idx])

        # Replace 'case' with 'seg'
        seg_image_name = self.images[idx].replace('case', 'seg')
        seg_name = os.path.join(self.seg_dir, seg_image_name)

        print("Trying to open image:", img_name)
        print("Trying to open mask:", seg_name)

        try:
            image = Image.open(img_name)
            mask = Image.open(seg_name)
        except Exception as e:
            print(f"Error occurred when reading files: {e}")
            return

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        # Contracting path
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Max-pooling layers for downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottom layer
        self.bottleneck = self.conv_block(512, 1024)

        # Expansive path
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)

        # Output layer
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Contracting path
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottom layer
        bottleneck = self.bottleneck(self.pool(enc4))

        # Expansive path
        dec4 = torch.cat([self.upconv4(bottleneck), enc4], dim=1)
        dec4 = self.dec4(dec4)

        dec3 = torch.cat([self.upconv3(dec4), enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = torch.cat([self.upconv2(dec3), enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = torch.cat([self.upconv1(dec2), enc1], dim=1)
        dec1 = self.dec1(dec1)

        return torch.sigmoid(self.out(dec1))  # Using sigmoid activation for binary segmentation

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

# Dice Similarity Coefficient (DSC) Calculation
def dice_coefficient(predicted, target):
    smooth = 1.0  # To avoid division by zero
    product = predicted * target
    intersection = 2. * product.sum()
    coefficient = (intersection + smooth) / (predicted.sum() + target.sum() + smooth)
    return coefficient.item()

# Visualization of Segmentation Results
def visualize_results(original, mask, prediction, index):
    plt.figure(figsize=(10,5))
    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original MRI')

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Ground Truth Mask')

    plt.subplot(1, 3, 3)
    plt.imshow(prediction, cmap='gray')
    plt.title('Predicted Mask')

    plt.savefig(f"result_{index}.png")
    plt.close()

# Running Inference on Test Dataset
def test_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    total_dsc = 0.0

    with torch.no_grad():
        for images, masks in test_loader:
            outputs = model(images)
            predicted_masks = torch.sigmoid(outputs) > 0.5  # Assuming binary segmentation
            dsc = dice_coefficient(predicted_masks, masks)
            total_dsc += dsc

    average_dsc = total_dsc / len(test_loader)
    return average_dsc

# Define the number of epochs
epochs = 5

# Set up DataLoader
transform = transforms.Compose([
    transforms.ToTensor(),
])
train_dataset = MRIDataset(training_path, seg_training_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# For testing, create test_loader
test_dataset = MRIDataset(testing_path, seg_testing_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model, optimizer, and loss
model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# Best Dice Score
best_dsc = 0.0

# Training Loop
for epoch in range(epochs):
    model.train()
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        predicted_masks = torch.sigmoid(outputs) > 0.5
        dsc = dice_coefficient(predicted_masks, masks)

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Dice Score: {dsc:.4f}")

    if epoch % 5 == 0:
        visualize_results(images[0, 0].cpu().numpy(), masks[0, 0].cpu().numpy(), predicted_masks[0, 0].cpu().numpy(), epoch)

    if dsc > best_dsc:
        best_dsc = dsc
        torch.save(model.state_dict(), 'best_model.pth')

# Clear GPU memory cache
torch.cuda.empty_cache()

# Load best model for inference on test set
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Assuming test_loader has been created similar to train_loader
average_dsc = test_model(model, test_loader)
print(f"Average Dice Score on Test Set: {average_dsc:.4f}")
