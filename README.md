# QoR-Predictor-and-RL-agent

## Project Overview

The QoR (Quality of Results) Predictor for AIGs (And-Inverter Graphs) is a machine learning system that automatically optimizes digital circuit synthesis processes. It uses reinforcement learning to find optimal synthesis recipes and a Graph Neural Network (GNN) with LSTM model to predict circuit area outcomes.

## Key Features

- **Reinforcement Learning Agent**: Automatically discovers optimal synthesis recipes
*: Predicts circuit area based on design structure and synthesis recipe
- **ABC Tool Integration**: Interfaces with industry-standard ABC synthesis tool
- **Multi-Design Support**: Works across different circuit designs
- **Significant Area Reduction**: Achieves 20-27% area reduction on benchmark circuits

## Prerequisites

### System Requirements
- Python 3.7 or higher
- GCC compiler (for building ABC tool)
- At least 8GB RAM for training
- macOS, Linux, or Windows with WSL

### Python Dependencies
```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install numpy pandas matplotlib scikit-learn jupyter
```

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd QoR-Predictor-for-AIGs
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Build ABC Synthesis Tool
```bash
# Download ABC source
wget https://github.com/berkeley-abc/abc/archive/master.tar.gz -O abc.tar.gz
tar -xzf abc.tar.gz
cd abc-master
make
cd ..
cp abc-master/abc .
```

### 4. Verify Installation
```bash
# Test ABC tool
./abc -c "echo Hello World"

# Test Python environment
python3 -c "import torch; import torch_geometric; print('Environment OK')"
```

## Project Structure

```
QoR-Predictor-for-AIGs/
├── abc*                  # Compiled ABC synthesis tool
├── rl.py                 # Main reinforcement learning implementation
├── byee.ipynb            # GNN-LSTM prediction model development
├── designs/             # Circuit design files (.bench format)
├── models/               # Trained model checkpoints
├── results/              # Training results and logs
├── RL Results.txt        # Detailed training logs
├── PROJECT.md            # Complete technical documentation
└── QWEN.md              # Project context documentation
```

## Usage

### Training Mode
Train the RL agent to discover optimal synthesis recipes:
```bash
python3 rl.py --mode train --episodes 5000
```

### Inference Mode
Apply trained model to generate optimal recipes for new designs:
```bash
python3 rl.py --mode infer --model models/final_model.pth
```

### Custom Training
Train with specific parameters:
```bash
python3 rl.py --mode train --episodes 1000 --learning_rate 0.001
```

## Core Components

### 1. Reinforcement Learning Agent (`rl.py`)
- **CircuitSynthesisRL Class**: Main RL agent implementation
- **Policy Network**: 3-layer neural network for action selection
- **ABC Integration**: Interfaces with ABC synthesis tool
- **Reward Function**: Based on area reduction percentage

### 2. QoR Prediction Model (`byee.ipynb`)
- **GNN-LSTM Hybrid**: Combines Graph Neural Networks with LSTM
- **Graph Representation**: Uses torch_geometric for circuit structures
- **Recipe Processing**: LSTM for synthesis command sequences
- **Area Prediction**: Predicts final circuit area

### 3. Data Processing Pipeline
- **Bench Files**: Circuit designs in standard .bench format
- **Vocabulary Building**: Gate types and synthesis command vocabulary
- **Graph Conversion**: Transforms circuits to graph representations
- **Dataset Preprocessing**: Prepares data for training

## Circuit Synthesis Commands

The RL agent can select from these synthesis commands:
- `strash`: Structural hashing
- `balance`: Logic balancing
- `rewrite`: Algebraic rewriting
- `refactor`: Boolean factoring
- `resub`: Resubstitution
- `fraig`: Functionally reduced AND-INV graphs
- `resyn`: Basic resynthesis
- `resyn2`: Advanced resynthesis
- Commands with `-z` flag for zero-cost transformations

## Training Results

The system achieves significant area reduction:
- **AES Circuit**: Up to 27.56% area reduction
- **I2C Circuit**: Up to 25.06% area reduction
- **TV80 Circuit**: Up to 27.56% area reduction

## Model Architecture

### Policy Network
- Input layer: State representation (recipe length, area ratio, design encoding)
- Hidden layers: 2×128-node ReLU layers
- Output layer: Action probabilities for 16 synthesis commands

### GNN-LSTM Model
- **Graph Processing**: 2-layer GCN for circuit structure analysis
- **Recipe Processing**: LSTM for synthesis command sequences
- **Feature Fusion**: Concatenated graph and recipe embeddings
- **Prediction**: Final area prediction layers

## File Formats

### .bench Files
Standard circuit design format with:
- INPUT declarations
- Logic gate definitions (AND, OR, NOT, etc.)
- OUTPUT declarations

### Model Checkpoints
Saved in PyTorch format (.pth) containing:
- Neural network weights
- Training state information
- Hyperparameters

## Performance Metrics

### RL Agent Performance
- **Training Episodes**: Up to 5000 episodes
- **Area Reduction**: 20-27% across benchmark circuits
- **Learning Rate**: Adaptive based on performance

### Prediction Model Performance
- **RMSE**: 26-67 (normalized scale)
- **Cross-Design Generalization**: Good transfer between circuit types
- **Prediction Accuracy**: High correlation with actual results

## Applications

### Electronic Design Automation (EDA)
- Automated circuit synthesis optimization
- Quality prediction for synthesis flows
- Design space exploration

### Chip Design
- Area optimization for integrated circuits
- Power and performance improvement
- Manufacturing cost reduction

### Research and Development
- Novel synthesis recipe discovery
- ML-driven EDA tool development
- Circuit optimization benchmarking

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- ABC synthesis tool team at UC Berkeley
- PyTorch Geometric development team
- Circuit benchmark providers
>>>>>>> ee15a6f (Initial commit)
