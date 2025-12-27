
import numpy as np
from Tokenizer import Tokenizer
from Cell import Cell
def main() -> None:
  vocab_size = 26
  hidden_size = 128
  cell = Cell(vocab_size, hidden_size)
  x = np.zeros((vocab_size, 1))
  x[0] = 1
  h = np.zeros((hidden_size, 1))
  C = np.zeros((hidden_size, 1))
  h_new, C_new, cache = cell.forward(x, h, C)
  assert h_new.shape == (hidden_size, 1)
  assert C_new.shape == (hidden_size, 1)

  assert np.all(h_new >= -1) and np.all(h_new <= 1)
  print("✓ Forward pass works")
  
   
def create_training_data(text, seq_length):
    inputs = []
    targets = []
    tokenizer = Tokenizer(text)
    encoded = tokenizer.encode(text)
    for i in range(0, len(encoded) - seq_length):
        inputs.append(encoded[i:i+seq_length])
        targets.append(encoded[i+seq_length])    
    return np.array(inputs), np.array(targets), tokenizer
    
if __name__ == "__main__":
    main()
