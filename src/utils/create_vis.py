import torch
from pathlib import Path
from src.utils.visualizations import TrainingVisualizer, visualize_predictions
from src.model.enhanced_resnet import EnhancedResNet
from src.data.preprocessing import DataPreprocessor

def main():
    # Paths
    checkpoints_dir = Path("checkpoints")
    history_path = checkpoints_dir / "history.json"
    model_path = checkpoints_dir / "best_model.pth"
    
    # Create visualizations of training history
    visualizer = TrainingVisualizer(history_path)
    visualizer.plot_training_curves()
    visualizer.plot_learning_rate()
    visualizer.create_training_report()
    
    # Load model and create prediction visualizations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    preprocessor = DataPreprocessor("data/processed")
    _, val_loader, _ = preprocessor.get_dataloaders()
    num_classes = preprocessor.get_num_classes()
    
    # Get class names
    class_names = sorted(Path("data/processed/train").glob("*/"))
    class_names = [p.name for p in class_names]
    
    # Load model
    model = EnhancedResNet(num_classes=num_classes)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Create prediction visualization
    fig = visualize_predictions(model, val_loader, class_names, device)
    fig.savefig("src/utils/visualizations/predictions_grid.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()