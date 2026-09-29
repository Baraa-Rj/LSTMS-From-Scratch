import numpy as np
from .Tokenizer import Tokenizer
from .Lstm import Lstm
from .Trainer import Trainer
from .Generator import Generator

def prepare_training_data(text, seq_length):
    tokenizer = Tokenizer(text)
    encoded = tokenizer.encode(text)
    
    inputs_list = []
    targets_list = []
    
    for i in range(len(encoded) - seq_length):
        inputs = encoded[i:i+seq_length]
        targets = encoded[i+1:i+seq_length+1]
        
        inputs_list.append(inputs)
        targets_list.append(targets)
    
    return inputs_list, targets_list, tokenizer

def main() -> None:
    print("=" * 50)
    print("Character-Level LSTM Training")
    print("=" * 50)
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()
    seq_length = 10  
    
    print(f"\n1. Preparing training data...")
    inputs_list, targets_list, tokenizer = prepare_training_data(text, seq_length)
    print(f"   Vocabulary size: {tokenizer.vocab_size}")
    print(f"   Characters: {tokenizer.chars}")
    print(f"   Total sequences available: {len(inputs_list)}")
    
    import random
    sample_size = min(10000, len(inputs_list)) 
    indices = random.sample(range(len(inputs_list)), sample_size)
    inputs_list = [inputs_list[i] for i in indices]
    targets_list = [targets_list[i] for i in indices]
    print(f"   Using {len(inputs_list)} sequences for training")
    
    print(f"\n2. Initializing LSTM...")
    hidden_size = 64  
    lstm = Lstm(input_size=tokenizer.vocab_size, hidden_size=hidden_size)
    print(f"   Input size: {tokenizer.vocab_size}")
    print(f"   Hidden size: {hidden_size}")
    
    print(f"\n3. Creating trainer...")
    learning_rate = 0.01
    trainer = Trainer(lstm, learning_rate=learning_rate, clip_value=5.0)
    print(f"   Learning rate: {learning_rate}")
    print(f"   Gradient clipping: 5.0")
    
    print(f"\n4. Training...")
    epochs = 10
    losses = trainer.train(inputs_list, targets_list, epochs=epochs, print_every=10)
    
    print(f"\n5. Training complete!")
    print(f"   Final loss: {losses[-1]:.4f}")
    print(f"   Initial loss: {losses[0]:.4f}")
    print(f"   Loss reduction: {losses[0] - losses[-1]:.4f}")
    
    print(f"\n6. Generating text samples...")
    generator = Generator(lstm, tokenizer)
    
    print("\n   Temperature 0.5 (Conservative):")
    print(f"   '{generator.generate('hello mom', 30, temperature=0.5)}'")
    
    print("\n   Temperature 1.0 (Normal):")
    print(f"   '{generator.generate('fuc', 30, temperature=1.0)}'")
    
    print("\n   Temperature 1.5 (Creative):")
    print(f"   '{generator.generate('wor', 30, temperature=1.5)}'")
    
    print("\n" + "=" * 50)
    print("Testing complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()
