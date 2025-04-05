# Image Classification using Convolutional Neural Networks (CNNs)

This project implements and compares several CNN-based neural network architectures from scratch using PyTorch. The models are trained on the CIFAR-10 dataset to evaluate how different architectural components (e.g., MLPs, convolutions, residual connections, inception modules) affect classification performance.

---

## Project Structure

```text
cnn_image_classification/
├── Image Classification using Convolutional Neural Networks (CNNs).ipynb  # Main notebook
├── conv_best.pt                   # Best checkpoint for ConvNet
├── resPlain_best.pt               # Best checkpoint for plain ResNet
├── resBottleneck_best.pt          # Best checkpoint for bottleneck ResNet
├── inception_best.pt              # Best checkpoint for Inception-like model
├── requirements.txt               # Python dependencies for setup
├── README.md                      # Project documentation

---

## Architectures

The notebook contains modular implementations of five neural network architectures:

- **MLP**: Fully connected layers on flattened images
- **ConvBlock**: Basic convolutional layers with BatchNorm and ReLU
- **ResBlockPlain**: Basic residual blocks (no bottleneck)
- **ResBlockBottleneck**: Residual blocks with a bottleneck design (1×1 → 3×3 → 1×1)
- **InceptionBlock**: Inspired by the Inception module, with multiple branches (1x1, 3x3, 5x5, pooling)

---

## Dataset & Training

- **Dataset**: CIFAR-10 (32×32 color images, 10 classes)
- **Data augmentation**: Random crop and horizontal flip (training set)
- **Optimization**:
  - Optimizer: SGD + Momentum
  - Learning Rate: 0.1
  - LR Scheduler: MultiStepLR
  - Epochs: 100
  - Batch size: 128
- **Logging**: TensorBoard
- **Checkpoints**: Saved at regular intervals and best model per architecture

---

## Results

| Method        | Accuracy (%) | # Params   | Expected Accuracy (%) | Expected # Params |
|---------------|--------------|------------|------------------------|-------------------|
| `mlp`         | 62.62        | 1,649,354  | 62.6                   | 1,649,354         |
| `conv`        | 81.93        | 510,426    | 81.9                   | 510,426           |
| `resPlain`    | 88.85        | 510,426    | 88.6                   | 510,426           |
| `resBottleneck` | 87.03      | 113,946    | 86.5                   | 113,946           |
| `inception`   | 84.02        | 124,026    | 83.7                   | 124,026           |

---

## How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. **If you have Jupyter installed, run:**:
  ```bash
  jupyter notebook
  ```

3. **Then open the following file in your browser**:
  ```bash
  Image Classification using Convolutional Neural Networks (CNNs).ipynb
  ```

4. **In the notebook, set the desired block type by modifying:.**
  ```bash
  args.block_type = 'mlp'  # or 'conv', 'resPlain', 'resBottleneck', 'inception'
  ```
  
5. **Run all cells**
~Train from scratch (will take time), or
~Load a saved .pt checkpoint to evaluate test accuracy.
