# Character-Level LSTM from Scratch

A complete implementation of Long Short-Term Memory (LSTM) networks built from scratch using only NumPy. This project demonstrates character-level language modeling, training via backpropagation through time (BPTT), and text generation.

## 🎯 Features

- **Pure NumPy implementation** - No deep learning frameworks
- **Complete LSTM architecture** - Forget, input, candidate, and output gates
- **Character-level modeling** - Learn patterns at the character level
- **Backpropagation Through Time (BPTT)** - Full gradient computation
- **Text generation** - Sample text with temperature control
- **Gradient clipping** - Prevent exploding gradients
- **Modular design** - Clean separation of concerns

## 📚 Project Structure

```
LSTMS/
├── src/lstms/
│   ├── Activation.py      # Sigmoid, tanh, softmax functions
│   ├── Cell.py            # Core LSTM cell (forward & backward)
│   ├── Lstm.py            # LSTM layer (sequence processing)
│   ├── Tokenizer.py       # Character encoding/decoding
│   ├── Trainer.py         # Training loop and weight updates
│   ├── Generator.py       # Text generation with temperature
│   └── __main__.py        # Main training script
├── input.txt              # Training data (your text file)
├── pyproject.toml         # Project configuration
└── README.md              # This file
```

## 🏗️ Architecture

### LSTM Cell
The core computational unit implementing the standard LSTM equations:

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)    # Forget gate
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)    # Input gate
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C) # Candidate cell state
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)    # Output gate

C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t       # Update cell state
h_t = o_t ⊙ tanh(C_t)                   # Update hidden state
```

**Components:**
- 4 weight matrices (forget, input, candidate, output gates)
- Hidden state (`h`) for short-term memory
- Cell state (`C`) for long-term memory
- Forward and backward propagation methods

### Data Flow

```
Text Input
    ↓
[Tokenizer] → Character indices
    ↓
[LSTM Layer] → Process sequences timestep-by-timestep
    ↓ [Cell] → Gates → State updates
    ↓
[Output Layer] → Softmax probabilities
    ↓
[Loss] → Cross-entropy
    ↓
[BPTT] → Compute gradients backward through time
    ↓
[Trainer] → Update weights with gradient descent
    ↓
Trained Model → [Generator] → Generate new text
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
git clone https://github.com/Baraa-Rj/LSTMS-From-Scratch
cd LSTMS-From-Scratch

python -m venv .venv
source .venv/bin/activate

# Install the package and its dependency (NumPy); add [test] for pytest
pip install -e '.[test]'
```

### 2. Prepare Training Data

Create or add your `input.txt` file with training text:

```bash
echo "Your training text goes here. The more data, the better the results." > input.txt
```

### 3. Train the Model

Run from the repository root (the script reads `input.txt` from the current directory):

```bash
python -m lstms
```

### Run the Tests

```bash
python -m pytest
```

**Training parameters** (edit in `__main__.py`):
- `seq_length`: Length of input sequences (default: 10)
- `hidden_size`: LSTM hidden layer size (default: 64)
- `learning_rate`: Gradient descent step size (default: 0.01)
- `epochs`: Number of training iterations (default: 100)

### 4. Expected Output

```
==================================================
Character-Level LSTM Training
==================================================

1. Preparing training data...
   Vocabulary size: 65
   Training sequences: 1,115,384

2. Initializing LSTM...
   Input size: 65
   Hidden size: 64

3. Creating trainer...
   Learning rate: 0.01

4. Training...
Epoch 10/100, Loss: 1.8234
Epoch 20/100, Loss: 1.5123
...
Epoch 100/100, Loss: 0.9876

5. Training complete!

6. Generating text samples...
   Temperature 0.5: "hello world hello world h..."
   Temperature 1.0: "hello worle hello wopld..."
   Temperature 1.5: "hxllo qorld wemlo vorld..."
```

## 📖 Usage Examples

### Train with Custom Parameters

```python
from lstms.Lstm import Lstm
from lstms.Trainer import Trainer
from lstms.Tokenizer import Tokenizer

# Load your data
with open('input.txt', 'r') as f:
    text = f.read()

# Prepare data
tokenizer = Tokenizer(text)
# ... prepare inputs_list, targets_list ...

# Create model with larger hidden size
lstm = Lstm(input_size=tokenizer.vocab_size, hidden_size=128)

# Train with custom learning rate
trainer = Trainer(lstm, learning_rate=0.005, clip_value=5.0)
losses = trainer.train(inputs_list, targets_list, epochs=200)
```

### Generate Text

```python
from lstms.Generator import Generator

generator = Generator(lstm, tokenizer)

# Conservative generation (predictable)
text1 = generator.generate("The", length=100, temperature=0.5)

# Normal generation
text2 = generator.generate("The", length=100, temperature=1.0)

# Creative generation (more random)
text3 = generator.generate("The", length=100, temperature=1.5)

print(text1)
```

## 🧮 Model Parameters

For a model with:
- Vocabulary size: 65 characters
- Hidden size: 64 units

**Total parameters:** ~37,500

```
Cell weights:
  - Forget gate (W_f, b_f):     64 × 129 + 64 = 8,320
  - Input gate (W_i, b_i):      64 × 129 + 64 = 8,320
  - Candidate (W_C, b_C):       64 × 129 + 64 = 8,320
  - Output gate (W_o, b_o):     64 × 129 + 64 = 8,320
  
Output layer:
  - Weights (W_y):              65 × 64 = 4,160
  - Bias (b_y):                 65
```

## 🔬 How It Works

### 1. Character Tokenization
Converts text to integer sequences:
```python
"hello" → [46, 43, 50, 50, 53]
```

### 2. Sliding Window Training
Creates input-target pairs:
```python
Input:  "hello worl" → [46, 43, 50, 50, 53, 1, 61, 53, 56, 50]
Target: "ello world" → [43, 50, 50, 53, 1, 61, 53, 56, 50, 42]
```

### 3. Forward Pass
- One-hot encode each character
- Process through LSTM cell timestep by timestep
- Maintain hidden state (`h`) and cell state (`C`)
- Project to vocabulary size with output layer

### 4. Loss Computation
- Apply softmax to get probability distribution
- Compute cross-entropy loss: `-log(p(correct_char))`
- Calculate gradients of loss w.r.t. outputs

### 5. Backward Pass (BPTT)
- Backpropagate through time (right to left)
- Compute gradients for all gates
- Accumulate gradients across timesteps
- Clip gradients to prevent explosion

### 6. Weight Update
- Apply gradient descent: `W = W - lr × ∇W`
- Update all LSTM cell weights and output layer

### 7. Generation
- Start with seed text to warm up state
- Sample next character from probability distribution
- Use temperature to control randomness
- Feed sampled character back as input
- Repeat for desired length

## 🎛️ Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seq_length` | 10 | Length of input sequences |
| `hidden_size` | 64 | Number of LSTM hidden units |
| `learning_rate` | 0.01 | Gradient descent step size |
| `clip_value` | 5.0 | Gradient clipping threshold |
| `epochs` | 100 | Number of training iterations |
| `temperature` | 1.0 | Sampling randomness (0.5-2.0) |

**Tuning tips:**
- Increase `hidden_size` for more model capacity
- Decrease `learning_rate` if loss is unstable
- Increase `seq_length` to learn longer dependencies
- Lower `temperature` for more coherent generation
- Increase `epochs` until loss plateaus

## 📊 Example Results

After training on Shakespeare's works:

**Temperature 0.5 (Conservative):**
```
"To be or not to be, that is the question of the matter of the state"
```

**Temperature 1.0 (Balanced):**
```
"To be or not to be, that is the quection that mights to seart"
```

**Temperature 1.5 (Creative):**
```
"To be xr nat po ke, thet as fhe vuestmen whan strond bo"
```

## 🐛 Troubleshooting

### Loss is NaN
- Reduce learning rate (try 0.001)
- Check gradient clipping is enabled
- Verify input data is properly encoded

### Loss not decreasing
- Increase model size (`hidden_size`)
- Train for more epochs
- Check you have enough training data
- Verify targets align with inputs

### Out of memory
- Reduce `hidden_size`
- Use shorter sequences (`seq_length`)
- Process fewer sequences per epoch

### Generated text is gibberish
- Train for more epochs
- Increase training data size
- Check vocabulary contains expected characters

## 🔧 Advanced Usage

### Save/Load Model

```python
import pickle

# Save trained model
with open('lstm_model.pkl', 'wb') as f:
    pickle.dump({'lstm': lstm, 'tokenizer': tokenizer}, f)

# Load model
with open('lstm_model.pkl', 'rb') as f:
    data = pickle.load(f)
    lstm = data['lstm']
    tokenizer = data['tokenizer']
```

### Batch Processing

For large datasets, process in batches:

```python
batch_size = 100
for i in range(0, len(inputs_list), batch_size):
    batch_inputs = inputs_list[i:i+batch_size]
    batch_targets = targets_list[i:i+batch_size]
    # Train on batch
```

## 📝 Implementation Details

### Numerical Stability
- Softmax: Subtract max before exponential
- Log probability: Add small epsilon to avoid log(0)
- Gradient clipping: Prevent exploding gradients

### Memory Efficiency
- Store only necessary cache values
- Process sequences one at a time
- Clear intermediate gradients after update

### Gradient Computation
- Uses chain rule through all gates
- Accumulates gradients across timesteps
- Handles vanishing gradients via cell state

## 🎓 Learning Resources

This implementation follows the standard LSTM architecture from:
- **Original paper:** Hochreiter & Schmidhuber (1997)
- **Understanding LSTMs:** colah.github.io/posts/2015-08-Understanding-LSTMs
- **Backpropagation:** Karpathy's "The Unreasonable Effectiveness of RNNs"

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

Built from scratch to understand LSTM internals, backpropagation through time, and character-level language modeling.

---

**Note:** This is a pedagogical implementation. For production use, consider frameworks like PyTorch or TensorFlow that offer optimized implementations with GPU acceleration.
