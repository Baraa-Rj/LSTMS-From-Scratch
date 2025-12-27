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

    def train_step(self,inputs,target,epochs,print_every = 10):
        losses = []
        num_samples = len(inputs)
        for epoch in range(epochs):
            total_loss = 0
            for i in range(num_samples):
                input_seq = inputs[i]
                target_idx = target[i]

                ys, hs, caches = self.lstm.forward(input_seq)

                exp_scores = np.exp(ys[-1] - np.max(ys[-1]))
                probs = exp_scores / np.sum(exp_scores)

                loss = -np.log(probs[target_idx][0])
                total_loss += loss

                dY = probs.copy()
                dY[target_idx] -= 1

                dYs = [np.zeros_like(y) for y in ys]
                dYs[-1] = dY

                dWy, dby, cell_grads = self.lstm.backward(dYs, caches)

                self.update_weights(dWy, dby, cell_grads)

            avg_loss = total_loss / num_samples
            losses.append(avg_loss)

            if (epoch + 1) % print_every == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
        return losses