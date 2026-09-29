import numpy as np
from .Activation import Activation
class Cell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        scale  = 1 / (input_size + hidden_size)
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.bf = np.zeros((hidden_size, 1))

        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.bi = np.zeros((hidden_size, 1))

        self.WC = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.bC = np.zeros((hidden_size, 1))

        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.bo = np.zeros((hidden_size, 1))   

    def forward(self, x_t, h_prev, C_prev):
        concat = np.vstack((h_prev, x_t))

        f_t = Activation.sigmoid(np.dot(self.Wf, concat) + self.bf)
        i_gate = Activation.sigmoid(np.dot(self.Wi, concat) + self.bi)
        c_candidate = Activation.tanh(np.dot(self.WC, concat) + self.bC)
        o_gate = Activation.sigmoid(np.dot(self.Wo, concat) + self.bo)

        C_t = f_t * C_prev + i_gate * c_candidate
        h_new = o_gate * Activation.tanh(C_t)

        cache = {"x_t": x_t, "h_prev": h_prev, "C_prev": C_prev,
                 "f_t": f_t, "i_gate": i_gate, "c_candidate": c_candidate,
                 "o_gate": o_gate, "C_t": C_t, "h_new": h_new, "concat": concat}
        return h_new, C_t, cache
    def backward(self, dh_next, dC_next, cache):

        x_t = cache["x_t"]
        h_prev = cache["h_prev"]
        C_prev = cache["C_prev"]
        f_t = cache["f_t"]
        i_gate = cache["i_gate"]
        c_candidate = cache["c_candidate"]
        o_gate = cache["o_gate"]
        C_t = cache["C_t"]
        concat = cache["concat"]
        
        do_gate = dh_next * Activation.tanh(C_t)
        do_gate_raw = do_gate * o_gate * (1 - o_gate)

        dC_t = dC_next + dh_next * o_gate * (1 - Activation.tanh(C_t)**2)

        df_gate = dC_t * C_prev
        df_gate_raw = df_gate * f_t * (1 - f_t)

        di_gate = dC_t * c_candidate
        di_gate_raw = di_gate * i_gate * (1 - i_gate)

        dC_candidate_gate = dC_t * i_gate
        dC_candidate_raw = dC_candidate_gate * (1 - c_candidate**2)

        dconcat = (np.dot(self.Wf.T, df_gate_raw) +
                   np.dot(self.Wi.T, di_gate_raw) +
                   np.dot(self.WC.T, dC_candidate_raw) +
                   np.dot(self.Wo.T, do_gate_raw))
        
        dh_prev = dconcat[:self.hidden_size, :]
        dx_t = dconcat[self.hidden_size:, :]
        
        dC_prev = dC_t * f_t

        grads = {
            "dWf": np.dot(df_gate_raw, concat.T),
            "dbf": df_gate_raw,
            "dWi": np.dot(di_gate_raw, concat.T),
            "dbi": di_gate_raw,
            "dWC": np.dot(dC_candidate_raw, concat.T),
            "dbC": dC_candidate_raw,
            "dWo": np.dot(do_gate_raw, concat.T),
            "dbo": do_gate_raw
        }
        return dx_t, dh_prev, dC_prev, grads 