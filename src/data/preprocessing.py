import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import random

class FlowerDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        # Collect all images and labels
        for class_dir in self.data_dir.iterdir():
            if class_dir.is_dir():
                class_idx = self.class_to_idx[class_dir.name]
                for img_path in class_dir.glob("*.jpg"):
                    self.images.append(img_path)
                    self.labels.append(class_idx)
                    
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return image, label

class DataPreprocessor:
    def __init__(self, data_dir, batch_size=32, num_workers=4):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
    def get_dataloaders(self):
        """Create train, validation, and test dataloaders"""
        train_dataset = FlowerDataset(
            self.data_dir, 
            split='train',
            transform=self.train_transform
        )
        
        val_dataset = FlowerDataset(
            self.data_dir,
            split='val',
            transform=self.val_transform
        )
        
        test_dataset = FlowerDataset(
            self.data_dir,
            split='test',
            transform=self.val_transform
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
        
    def get_num_classes(self):
        """Get the number of classes in the dataset"""
        train_dir = self.data_dir / 'train'
        return len([d for d in train_dir.iterdir() if d.is_dir()])

if __name__ == "__main__":
    preprocessor = DataPreprocessor("data/processed")
    train_loader, val_loader, test_loader = preprocessor.get_dataloaders()
    print(f"Number of classes: {preprocessor.get_num_classes()}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")