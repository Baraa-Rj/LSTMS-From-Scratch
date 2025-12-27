from Cell import Cell
import numpy as np
from Activation import Activation
class Lstm:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = Cell(input_size, hidden_size)

        scale = 1 / (input_size + hidden_size)
        self.Wy = np.random.randn(input_size, hidden_size) * scale
        self.by = np.zeros((input_size, 1))
    def one_hot(self, index):
        x = np.zeros((self.input_size, 1))
        x[index] = 1
        return x
    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        C = np.zeros((self.hidden_size, 1))
        caches = []
        hs = []
        ys = []
        for idx in inputs:
            x = self.one_hot(idx)
            h, C, cache = self.cell.forward(x, h, C)
            y = np.dot(self.Wy, h) + self.by
            hs.append(h)
            ys.append(y)
            caches.append(cache)
        return ys, hs, caches
    def backward(self, dYs, caches):
        dWy = np.zeros_like(self.Wy)
        dby = np.zeros_like(self.by)
        dh_next = np.zeros((self.hidden_size, 1))
        dC_next = np.zeros((self.hidden_size, 1))

        # Initialize cell gradient accumulators
        cell_grads = {
            "dWf": np.zeros_like(self.cell.Wf),
            "dWi": np.zeros_like(self.cell.Wi),
            "dWC": np.zeros_like(self.cell.WC),
            "dWo": np.zeros_like(self.cell.Wo),
            "dbf": np.zeros_like(self.cell.bf),
            "dbi": np.zeros_like(self.cell.bi),
            "dbC": np.zeros_like(self.cell.bC),
            "dbo": np.zeros_like(self.cell.bo)
        }

        for t in reversed(range(len(dYs))):
            dY = dYs[t]
            cache = caches[t]

            dWy += np.dot(dY, cache["h_new"].T)
            dby += dY

            dh = np.dot(self.Wy.T, dY) + dh_next

            dx_t, dh_next, dC_next, grads = self.cell.backward(dh, dC_next, cache)
            
            # Accumulate cell gradients across timesteps
            for key in cell_grads:
                cell_grads[key] += grads[key]

        return dWy, dby, cell_grads
    
    def compute_loss(self, outputs, target_indices):
        total_loss = 0
        dYs = []  # Store gradients for backpropagation
        
        for logits, target_idx in zip(outputs, target_indices):
            probs = Activation.softmax(logits)
            
            correct_char_prob = probs[target_idx, 0]
            
            # Clamp probability to valid range
            correct_char_prob = np.clip(correct_char_prob, 1e-10, 1.0 - 1e-10)
            
            loss = -np.log(correct_char_prob)
            total_loss += loss
            
            # Gradient of cross-entropy loss w.r.t logits (softmax + CE derivative)
            dY = probs.copy()
            dY[target_idx] -= 1
            dYs.append(dY)
        
        avg_loss = total_loss / len(target_indices)
        return avg_loss, dYs