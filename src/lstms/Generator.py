import numpy as np
from Activation import Activation

class Generator:
    def __init__(self, lstm, tokenizer):
        self.lstm = lstm
        self.tokenizer = tokenizer

    def generate(self, seed_text, length, temperature=1.0):
        """Generate text starting from seed_text
        
        Args:
            seed_text: starting text to condition generation
            length: number of new characters to generate
            temperature: sampling temperature (0.5=conservative, 1.0=normal, 2.0=creative)
        
        Returns:
            generated text (seed + new characters)
        """
        encoded = self.tokenizer.encode(seed_text)
        h = np.zeros((self.lstm.hidden_size, 1))
        C = np.zeros((self.lstm.hidden_size, 1))
        generated_indices = []

        # Warm up state with seed text
        for idx in encoded:
            x = self.lstm.one_hot(idx)
            h, C, _ = self.lstm.cell.forward(x, h, C)

        current_idx = encoded[-1]

        # Generate new characters
        for _ in range(length):
            x = self.lstm.one_hot(current_idx)
            h, C, _ = self.lstm.cell.forward(x, h, C)
            y = np.dot(self.lstm.Wy, h) + self.lstm.by
            probs = Activation.softmax(y).flatten()
            
            # Check for numerical issues
            if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
                # Fall back to uniform distribution if numerical issues
                probs = np.ones(len(probs)) / len(probs)
            
            # Apply temperature
            if temperature != 1.0:
                probs = np.power(probs, 1.0 / temperature)
                probs_sum = np.sum(probs)
                if probs_sum > 0:
                    probs = probs / probs_sum  # Re-normalize
                else:
                    probs = np.ones(len(probs)) / len(probs)
            
            # Ensure probabilities sum to 1 (fix floating point errors)
            probs = probs / np.sum(probs)
            
            current_idx = np.random.choice(range(self.lstm.input_size), p=probs)
            generated_indices.append(current_idx)

        return seed_text + self.tokenizer.decode(generated_indices)