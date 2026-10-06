import numpy as np
from .Activation import Activation

class Generator:
    def __init__(self, lstm, tokenizer):
        self.lstm = lstm
        self.tokenizer = tokenizer

    def _step(self, index, h, C):
        """Feed one character; return the new state and the logits for the next one."""
        x = self.lstm.one_hot(index)
        h, C, _ = self.lstm.cell.forward(x, h, C)
        logits = np.dot(self.lstm.Wy, h) + self.lstm.by
        return h, C, logits

    def generate(self, seed_text, length, temperature=1.0):
        if not seed_text:
            raise ValueError("seed_text must not be empty")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        encoded = self.tokenizer.encode(seed_text)
        h = np.zeros((self.lstm.hidden_size, 1))
        C = np.zeros((self.lstm.hidden_size, 1))
        generated_indices = []

        # Warm up on the seed. Every seed character is fed exactly once; the
        # logits left over from the last one predict the first new character.
        for idx in encoded:
            h, C, logits = self._step(idx, h, C)

        # Generate new characters
        for _ in range(length):
            probs = Activation.softmax(logits).flatten()

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

            probs = probs / np.sum(probs)

            current_idx = np.random.choice(range(self.lstm.input_size), p=probs)
            generated_indices.append(current_idx)

            # Advance the state with the character that was just sampled.
            h, C, logits = self._step(current_idx, h, C)

        return seed_text + self.tokenizer.decode(generated_indices)
