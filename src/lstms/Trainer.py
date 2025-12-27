import numpy as np
class Trainer:

    def __init__(self,lstm,learning_rate=0.001,clip_value=5.0):
        self.lstm = lstm
        self.learning_rate = learning_rate
        self.clip_value = clip_value

    def clip_gradients(self,grads):
        if isinstance(grads, dict):
            for key in grads:
                np.clip(grads[key], -self.clip_value, self.clip_value, out=grads[key])
        else:
            np.clip(grads, -self.clip_value, self.clip_value, out=grads)
        return grads
    def update_weights(self,dWy,dby,cell_grads):
        dWy = self.clip_gradients(dWy)
        dby = self.clip_gradients(dby)
        cell_grads = self.clip_gradients(cell_grads)

        self.lstm.Wy -= self.learning_rate * dWy
        self.lstm.by -= self.learning_rate * dby

        self.lstm.cell.Wf -= self.learning_rate * cell_grads["dWf"]
        self.lstm.cell.bf -= self.learning_rate * cell_grads["dbf"]

        self.lstm.cell.Wi -= self.learning_rate * cell_grads["dWi"]
        self.lstm.cell.bi -= self.learning_rate * cell_grads["dbi"]

        self.lstm.cell.WC -= self.learning_rate * cell_grads["dWC"]
        self.lstm.cell.bC -= self.learning_rate * cell_grads["dbC"]

        self.lstm.cell.Wo -= self.learning_rate * cell_grads["dWo"]
        self.lstm.cell.bo -= self.learning_rate * cell_grads["dbo"]

    def train_step(self, inputs, targets):
        """Train on a single sequence
        
        Args:
            inputs: list of character indices (input sequence)
            targets: list of character indices (target sequence)
            
        Returns:
            loss: average loss for this sequence
        """
        # Forward pass
        outputs, hs, caches = self.lstm.forward(inputs)
        
        # Compute loss and gradients
        loss, dYs = self.lstm.compute_loss(outputs, targets)
        
        # Backward pass
        dWy, dby, cell_grads = self.lstm.backward(dYs, caches)
        
        # Update weights
        self.update_weights(dWy, dby, cell_grads)
        
        return loss
    
    def train(self, inputs_list, targets_list, epochs, print_every=1):
      
        losses = []
        num_sequences = len(inputs_list)
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            for inputs, targets in zip(inputs_list, targets_list):
                loss = self.train_step(inputs, targets)
                epoch_loss += loss
            
            avg_loss = epoch_loss / num_sequences
            losses.append(avg_loss)
            
            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return losses