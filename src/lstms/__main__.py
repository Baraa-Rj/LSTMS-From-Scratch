import numpy as np
from Tokenizer import Tokenizer
def main() -> None:
   
   text = "hello world"
   seq_length = 5
   inputs, targets, tokenizer = create_training_data(text, seq_length)
   print("Inputs:", inputs)
   print("Targets:", targets)
   print(f"First input: {tokenizer.decode(inputs[0])}")
   print(f"First target: {tokenizer.decode([targets[0]])}")
   print(f"Total sequences: {len(inputs)}")
   # Verify: the last 4 chars of input + target should equal the next sequence's first 5 chars
   print(f"\nVerification: '{tokenizer.decode(inputs[0])}' -> '{tokenizer.decode([targets[0]])}'")
   print("Data preparation successful!")
   
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
