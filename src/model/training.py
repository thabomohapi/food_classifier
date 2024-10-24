import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import autocast, GradScaler
from pathlib import Path
import json, time
from tqdm import tqdm
import torch.backends.cudnn as cudnn

from ..data.preprocessing import DataPreprocessor
from ..model.enhanced_resnet import EnhancedResNet

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record start time
        result = func(*args, **kwargs)  # Call the actual function
        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time  # Calculate time taken
        print(f"Function '{func.__name__}' took {elapsed_time:.6f} seconds to execute")
        return result  # Return the function's result
    return wrapper

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device='cuda', learning_rate=0.001, num_epochs=50):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_epochs = num_epochs
        
        # Enable cuDNN auto-tuner
        cudnn.benchmark = True
        
        # Initialize mixed precision training
        self.scaler = GradScaler('cuda')
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # One Cycle learning rate scheduler
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            epochs=num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.2,
            div_factor=25.0,
            final_div_factor=1000.0
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'best_val_acc': 0.0,
            'learning_rates': []
        }
        
        # Gradient accumulation steps
        self.accumulation_steps = 2
        
    @timing_decorator
    def train_epoch(self):
        """Train for one epoch with mixed precision"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        self.optimizer.zero_grad()
        
        for i, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Mixed precision forward pass
            with autocast('cuda'):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss = loss / self.accumulation_steps
            
            # Mixed precision backward pass
            self.scaler.scale(loss).backward()
            
            # Update weights if accumulation steps reached
            if (i + 1) % self.accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
            
            # Statistics
            total_loss += loss.item() * self.accumulation_steps
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss/(pbar.n+1),
                'acc': 100.*correct/total,
                'lr': self.scheduler.get_last_lr()[0]
            })
            
            # Clear cache periodically
            if i % 50 == 0:
                torch.cuda.empty_cache()
            
        return total_loss/len(self.train_loader), 100.*correct/total
        
    @timing_decorator
    def evaluate(self, loader, desc='Evaluating'):
        """Evaluate model on given loader"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            with autocast('cuda'):
                pbar = tqdm(loader, desc=desc)
                for inputs, targets in pbar:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    
                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    
                    pbar.set_postfix({
                        'loss': total_loss/(pbar.n+1),
                        'acc': 100.*correct/total
                    })
                    
        return total_loss/len(loader), 100.*correct/total
        
    @timing_decorator
    def train(self, save_dir='checkpoints'):
        """Complete training loop with checkpointing"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            # Train and evaluate
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.evaluate(self.val_loader, desc='Validating')
            
            # Save learning rate
            self.history['learning_rates'].append(self.scheduler.get_last_lr()[0])
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Save best model
            if val_acc > self.history['best_val_acc']:
                self.history['best_val_acc'] = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'scaler_state_dict': self.scaler.state_dict(),
                    'history': self.history
                }, save_dir / 'best_model.pth')
                
            # Save training history
            with open(save_dir / 'history.json', 'w') as f:
                json.dump(self.history, f)
                
            # Clear cache between epochs
            torch.cuda.empty_cache()
                
        # Final evaluation on test set
        test_loss, test_acc = self.evaluate(self.test_loader, desc='Testing')
        print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
        
        return self.history

def main():
    # Set memory optimizations
    torch.cuda.set_per_process_memory_fraction(1.0) # reserve some memory for system
    # torch.backends.cudnn.benchmark = True

    #Environment variables for memory optimization
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Reduce batch size and workers for memory efficiency
    preprocessor = DataPreprocessor(
        "data/processed",
        batch_size=32,  # Reduced from default due to Vram/Memory limitations
    )
    train_loader, val_loader, test_loader = preprocessor.get_dataloaders()
    num_classes = preprocessor.get_num_classes()
    
    # Setup model
    model = EnhancedResNet(num_classes=num_classes)
    model = model.to(device)
    
    # Train model
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=0.001,
        num_epochs=60
    )
    
    history = trainer.train()
    print("Best validation accuracy:", history['best_val_acc'])

if __name__ == "__main__":
    main()