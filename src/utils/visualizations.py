import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
from pathlib import Path
import torch
from torchviz import make_dot
import pandas as pd
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class TrainingVisualizer:
    def __init__(self, history_path, save_dir="src/utils/visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Load training history
        with open(history_path) as f:
            self.history = json.load(f)
            
        # Set style
        # plt.style.use('seaborn')
        sns.set_palette("husl")
        
    def plot_training_curves(self):
        """Create an interactive plot of training and validation metrics"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Model Accuracy', 'Model Loss'),
            vertical_spacing=0.15
        )

        # Accuracy subplot
        fig.add_trace(
            go.Scatter(y=self.history['train_acc'], name="Train Accuracy",
                      line=dict(color='#2ecc71', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(y=self.history['val_acc'], name="Validation Accuracy",
                      line=dict(color='#e74c3c', width=2)),
            row=1, col=1
        )

        # Loss subplot
        fig.add_trace(
            go.Scatter(y=self.history['train_loss'], name="Train Loss",
                      line=dict(color='#2ecc71', width=2)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(y=self.history['val_loss'], name="Validation Loss",
                      line=dict(color='#e74c3c', width=2)),
            row=2, col=1
        )

        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Training and Validation Metrics",
            title_x=0.5,
            template="plotly_white"
        )

        # Update y-axes labels
        fig.update_yaxes(title_text="Accuracy (%)", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=2, col=1)
        
        # Update x-axes labels
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        
        # Save interactive plot
        fig.write_html(self.save_dir / "training_curves.html")
        
    def plot_learning_rate(self):
        """Plot learning rate schedule if available"""
        if 'learning_rates' in self.history:
            plt.figure(figsize=(10, 5))
            plt.plot(self.history['learning_rates'], 
                    color='#3498db', linewidth=2)
            plt.title('Learning Rate Schedule')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            plt.savefig(self.save_dir / 'learning_rate.png', 
                       bbox_inches='tight', dpi=300)
            plt.close()
            
    def create_training_report(self):
        """Generate an HTML report with all training visualizations"""
        report = f"""
        <html>
        <head>
            <title>Training Report - ResNet Flower Classification</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f6fa;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                h1, h2 {{
                    color: #2c3e50;
                }}
                .metric-card {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 10px 0;
                }}
                .highlight {{
                    color: #2980b9;
                    font-weight: bold;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>ResNet Flower Classification - Training Report</h1>
                
                <h2>Training Summary</h2>
                <div class="metric-card">
                    <p>Best Validation Accuracy: <span class="highlight">{self.history['best_val_acc']:.2f}%</span></p>
                    <p>Final Training Accuracy: <span class="highlight">{self.history['train_acc'][-1]:.2f}%</span></p>
                    <p>Final Validation Accuracy: <span class="highlight">{self.history['val_acc'][-1]:.2f}%</span></p>
                    <p>Total Epochs: <span class="highlight">{len(self.history['train_acc'])}</span></p>
                </div>
                
                <h2>Training Curves</h2>
                <iframe src="training_curves.html" width="100%" height="800px" frameborder="0"></iframe>
            </div>
        </body>
        </html>
        """
        
        with open(self.save_dir / 'training_report.html', 'w') as f:
            f.write(report)

def visualize_predictions(model, dataloader, class_names, device, num_images=16):
    """Create a grid of predictions vs actual labels"""
    model.eval()
    images, labels, preds = [], [], []
    
    with torch.no_grad():
        for batch_images, batch_labels in dataloader:
            batch_images = batch_images.to(device)
            outputs = model(batch_images)
            _, predicted = outputs.max(1)
            
            images.extend(batch_images.cpu())
            labels.extend(batch_labels)
            preds.extend(predicted.cpu())
            
            if len(images) >= num_images:
                break
    
    # Create grid plot
    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    axes = axes.ravel()
    
    for idx in range(num_images):
        image = images[idx].permute(1, 2, 0).numpy()
        image = (image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1)
        
        axes[idx].imshow(image)
        label = class_names[labels[idx]]
        pred = class_names[preds[idx]]
        color = 'green' if label == pred else 'red'
        
        axes[idx].set_title(f'True: {label}\nPred: {pred}', color=color)
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig