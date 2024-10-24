# Food-Image Classifier with ResNet

A high-performance image classification model optimized for mobile and consumer GPUs. This implementation features an enhanced ResNet architecture with modern attention mechanisms, mobile-optimized blocks, and efficient training strategies.

## Features

- **Optimized ResNet Architecture**: Designed for faster convergence and good accuracy
- **Efficient MBConv Blocks**: Memory-bandwidth optimized blocks with depthwise separable convolutions
- **Advanced Stem**: Efficient early feature extraction with grouped convolutions
- **Channel Shuffle Operations**: Efficient information flow between groups
- **Feature Pyramid Design**: Multi-scale feature extraction with shared weights
- **Memory Efficient**: Designed to run on consumer-grade GPUs
- **Easy-to-use Web Interface**: Drag-and-drop interface for image classification
- **Docker Support**: Ready to deploy with Docker and CUDA support
- **Good Accuracy**: Achieves 82% validation accuracy on the Food dataset

## Quick Start

### Using Docker (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/thabomohapi/food-classifier.git
cd food-classifier
```

2. Build and run with Docker:
```bash
docker build -t food_classifier .
docker run --name food_classifier -p 5000:5000 food_classifier
```

3. Open your browser and navigate to:
```
http://localhost:5000
```

### Local Installation

1. Create the conda environment:
```bash
conda env create -f environment.yml
conda activate food_classifier
```

2. Run the application:
```bash
python -m src.app
```

## Model Architecture 🧠

The FastConvergenceResNet architecture includes several optimizations:

- Enhanced feature extraction with pyramid blocks
- Efficient attention mechanisms
- Channel shuffle operations
- Stochastic depth regularization
- Advanced stem for better initial feature extraction

## Performance 📊

| Metric | Value |
|--------|--------|
| Best Validation Accuracy | 82.22% |
| Final Training Accuracy | 79.84% |
| Final Validation Accuracy | 82.06% |
| Training Time | ~9 hours on NVIDIA GeForce RTX 3050 GPU |

## Dataset

The model is trained on a Food dataset containing 99 classes which include:
- cheesecake
- cup_cakes
- donuts
- dumplings
- carrot cake
- etc...

Click [here](https://www.kaggle.com/api/v1/datasets/download/kmader/food41) to download the dataset.

## Training Your Own Model 🏋️‍♂️

1. Download and prepare the dataset:
```bash
python -m src.data.loader
```

2. Start training:
```bash
python -m src.model.training
```

Training configurations can be modified in `src/model/training.py`.

## API Usage

The model can be accessed through a REST API:

```python
import requests

# Make a prediction
url = 'http://localhost:5000/predict'
files = {'file': open('image.jpg', 'rb')}
response = requests.post(url, files=files)
predictions = response.json()
```

## Requirements

- Python 3.10
- PyTorch 2.1.0
- CUDA-capable GPU (optional but recommended)
- See environment.yml for complete list

## Troubleshooting

### Common Issues

1. CUDA Out of Memory
```bash
# Reduce batch size in src/model/training.py
batch_size = 16  # Default: 32
```

2. Image Loading Errors
```bash
# Ensure image format is supported (jpg, jpeg, png)
# Check image is not corrupted
```

## Technical Specifications

- **Input Resolution**: 224x224 RGB images
- **Base Channels**: 64 (progressively increasing to 512)
- **Model Depth**: 3 stages with dual MBConv blocks each
- **Attention Mechanism**: Squeeze-and-Excitation with reduced channels
- **Activation Function**: SiLU (Swish) for mobile efficiency

## Training Capabilities

- One-Cycle learning rate scheduling
- Label smoothing for better generalization
- AdamW optimizer with weight decay
- Gradient accumulation steps for stability
- Mixed precision training with automatic scaling

## Mobile GPU Optimizations

- Mixed precision (FP16) support for reduced memory usage
- Memory-efficient convolution operations
- Optimized memory management with gradient accumulation
- Regular cache clearing to prevent fragmentation
- Configurable batch sizes for different GPU capabilities

### GPU Support

To enable GPU support, ensure:
- NVIDIA GPU with CUDA support
- NVIDIA drivers installed
- CUDA toolkit installed

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
