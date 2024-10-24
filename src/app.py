from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image
import io
import torchvision.transforms as transforms
from src.model.enhanced_resnet import EnhancedResNet
from pathlib import Path

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = "data/processed"

dir = Path(data_dir)

def get_class_names():
    return open("data/class_names.txt", "r").read()

def load_model():
    """Load and configure the model"""
    try:
        # Model configuration
        class_names = get_class_names().split(", ")
        model = EnhancedResNet(num_classes=len(class_names))
        
        # Load model weights
        checkpoint_path = Path("checkpoints/best_model.pth")
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {checkpoint_path}")
            if 'history' in checkpoint and 'best_val_acc' in checkpoint['history']:
                print(f"Best validation accuracy: {checkpoint['history']['best_val_acc']:.2f}%")
        else:
            print("Warning: No checkpoint found. Using untrained model.")
        
        model = model.to(device)
        model.eval()
        return model, class_names
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None
    
model, class_names = load_model()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_bytes):
    """Preprocess image bytes for model inference"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transformations
        image_tensor = transform(image).unsqueeze(0)
        
        return image_tensor, None
    except Exception as e:
        return None, f"Error preprocessing image: {str(e)}"

def get_prediction(image_tensor):
    """Get model prediction with confidence scores"""
    try:
        with torch.no_grad():
            if device.type == 'cuda':
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(image_tensor.to(device))
            else:
                outputs = model(image_tensor.to(device))
            
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            # Get top 3 predictions
            top_probs, top_indices = torch.topk(probabilities, min(3, len(class_names)))
            
            predictions = [
                {
                    'class': class_names[idx.item()],
                    'probability': float(prob) * 100
                }
                for prob, idx in zip(top_probs, top_indices)
            ]
            
            return predictions, None
    except Exception as e:
        return None, f"Error during prediction: {str(e)}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if model is loaded
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    # Check if file is empty
    if file.filename == '':
        return jsonify({'error': 'Empty file submitted'}), 400
    
    try:
        # Read and preprocess image
        image_bytes = file.read()
        image_tensor, preprocess_error = preprocess_image(image_bytes)
        
        if preprocess_error:
            return jsonify({'error': preprocess_error}), 400
        
        # Get prediction
        predictions, prediction_error = get_prediction(image_tensor)
        
        if prediction_error:
            return jsonify({'error': prediction_error}), 500
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'model_info': {
                'name': 'EnhancedResNet',
                'input_size': '224x224',
                'preprocessing': 'RGB normalization with ImageNet stats'
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500
    
@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device),
        'num_classes': len(class_names) if class_names else 0,
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    })

def create_app(test_config=None):
    """Application factory function"""
    if test_config is None:
        # Load production config
        app.config.update(
            MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max file size
            UPLOAD_EXTENSIONS=['.jpg', '.jpeg', '.png', '.webp'],
            DEBUG=False
        )
    else:
        # Load test config
        app.config.update(test_config)
    
    return app

if __name__ == '__main__':
    # Memory optimization
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Run app
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=False)