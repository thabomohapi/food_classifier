# Food-Image Classifier with ResNet 🌸

A high-performance food classification system using an optimized ResNet architecture with enhanced attention mechanisms and efficient training strategies. This project provides both a trained model and a web interface for real-time food classification.

## Features ✨

- **Optimized ResNet Architecture**: Custom-designed for faster convergence and better accuracy
- **Advanced Attention Mechanisms**: Incorporates CBAM (Convolutional Block Attention Module)
- **Memory Efficient**: Designed to run on consumer-grade GPUs
- **Easy-to-use Web Interface**: Drag-and-drop interface for image classification
- **Docker Support**: Ready to deploy with Docker and CUDA support
- **High Accuracy**: Achieves 86.41% validation accuracy on the Food dataset

## Quick Start 🚀

### Using Docker (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/thabomohapi/flower-classifier.git
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
| Best Validation Accuracy | 86.41% |
| Final Training Accuracy | 85.16% |
| Final Validation Accuracy | 84.78% |
| Training Time | ~2 hours on NVIDIA GPU |

## Dataset 📸

The model is trained on a Food dataset containing 500+ classes which include:
- cheesecake
- cup_cakes
- donuts
- dumplings
- carrot cake
- etc...

Download the dataset [here](https://www.kaggle.com/api/v1/datasets/download/kmader/food41).

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

## API Usage 🔌

The model can be accessed through a REST API:

```python
import requests

# Make a prediction
url = 'http://localhost:5000/predict'
files = {'file': open('image.jpg', 'rb')}
response = requests.post(url, files=files)
predictions = response.json()
```

## Requirements 📋

- Python 3.10
- PyTorch 2.1.0
- CUDA-capable GPU (optional but recommended)
- See environment.yml for complete list

## Development 👩‍💻

1. Fork the repository
2. Create your feature branch
3. Install development dependencies:
```bash
conda env create -f environment.yml
```
4. Make your changes
5. Run tests:
```bash
python -m pytest tests/
```
6. Submit a pull request

## Troubleshooting 🔧

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

### GPU Support

To enable GPU support, ensure:
- NVIDIA GPU with CUDA support
- NVIDIA drivers installed
- CUDA toolkit installed

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation 📚

If you use this project in your research, please cite:

```bibtex
@software{food_classifier,
  author = {Thabo Mohapi},
  title = {Food-Image Classifier with ResNet},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/thabomohapi/food-classifier}
}
```

## Acknowledgments 🙏

- [PyTorch](https://pytorch.org/) team for the amazing framework
- Original ResNet paper authors
- Kaggle for the food dataset
