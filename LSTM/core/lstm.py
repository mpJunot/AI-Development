import numpy as np
from core.activations import Sigmoid, Tanh, ReLU, LeakyReLU, ELU

class LSTM:
    def __init__(self, input_size, hidden_size, activation='tanh'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W = np.random.randn(4 * hidden_size, input_size) * 0.01
        self.U = np.random.randn(4 * hidden_size, hidden_size) * 0.01
        self.b = np.zeros((4 * hidden_size, 1))
        self.dW = np.zeros_like(self.W)
        self.dU = np.zeros_like(self.U)
        self.db = np.zeros_like(self.b)
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()
        if activation == 'tanh':
            self.gate_activation = Tanh()
        elif activation == 'relu':
            self.gate_activation = ReLU()
        elif activation == 'leaky_relu':
            self.gate_activation = LeakyReLU()
        elif activation == 'elu':
            self.gate_activation = ELU()
        else:
            raise ValueError('Unknown activation')
        self.activation = activation

    def forward(self, x, h_prev, c_prev):
        seq_len, batch_size, _ = x.shape
        h = np.zeros((seq_len, batch_size, self.hidden_size))
        c = np.zeros((seq_len, batch_size, self.hidden_size))
        gates = np.zeros((seq_len, batch_size, 4 * self.hidden_size))
        self.cache = {'x': x, 'h': [], 'c': [], 'gates': [], 'i': [], 'f': [], 'o': [], 'g': [], 'c_prev': [], 'h_prev': []}
        for t in range(seq_len):
            x_t = x[t].T  # (input_size, batch_size)
            h_prev_T = h_prev.T  # (hidden_size, batch_size)
            gates_t = (self.W @ x_t + self.U @ h_prev_T + self.b).T  # (batch_size, 4*hidden_size)
            gates[t] = gates_t
            i, f, o, g = np.split(gates_t, 4, axis=1)
            i_act = self.sigmoid.forward(i)
            f_act = self.sigmoid.forward(f)
            o_act = self.sigmoid.forward(o)
            g_act = self.gate_activation.forward(g)
            c_t = f_act * c_prev + i_act * g_act
            h_t = o_act * self.tanh.forward(c_t)
            h[t] = h_t
            c[t] = c_t
            self.cache['h'].append(h_t)
            self.cache['c'].append(c_t)
            self.cache['gates'].append(gates_t)
            self.cache['i'].append(i_act)
            self.cache['f'].append(f_act)
            self.cache['o'].append(o_act)
            self.cache['g'].append(g_act)
            self.cache['c_prev'].append(c_prev)
            self.cache['h_prev'].append(h_prev)
            h_prev = h_t
            c_prev = c_t
        self.cache['h'] = np.array(self.cache['h'])
        self.cache['c'] = np.array(self.cache['c'])
        self.cache['gates'] = np.array(self.cache['gates'])
        self.cache['i'] = np.array(self.cache['i'])
        self.cache['f'] = np.array(self.cache['f'])
        self.cache['o'] = np.array(self.cache['o'])
        self.cache['g'] = np.array(self.cache['g'])
        self.cache['c_prev'] = np.array(self.cache['c_prev'])
        self.cache['h_prev'] = np.array(self.cache['h_prev'])
        return h, c

    def backward(self, dh, dc):
        cache = self.cache
        x = cache['x']
        h = cache['h']
        c = cache['c']
        gates = cache['gates']
        i = cache['i']
        f = cache['f']
        o = cache['o']
        g = cache['g']
        c_prev = cache['c_prev']
        h_prev = cache['h_prev']
        seq_len, batch_size, _ = x.shape
        dW = np.zeros_like(self.W)
        dU = np.zeros_like(self.U)
        db = np.zeros_like(self.b)
        dh_prev = np.zeros((batch_size, self.hidden_size))
        dc_prev = np.zeros((batch_size, self.hidden_size))
        dx = np.zeros_like(x)
        for t in reversed(range(seq_len)):
            tanh_c = self.tanh.forward(c[t])
            do = dh[t] * tanh_c
            do = self.sigmoid.backward(do)
            dc_t = dh[t] * o[t] * (1 - tanh_c ** 2) + dc_prev
            di = dc_t * g[t]
            di = self.sigmoid.backward(di)
            df = dc_t * c_prev[t]
            df = self.sigmoid.backward(df)
            dg = dc_t * i[t]
            dg = self.gate_activation.backward(dg)
            dGates = np.concatenate([di, df, do, dg], axis=1)  # (batch_size, 4*hidden_size)
            x_t = x[t].T
            h_prev_T = h_prev[t].T
            dW += dGates.T @ x_t.T
            dU += dGates.T @ h_prev_T.T
            db += np.sum(dGates, axis=0, keepdims=True).T
            dx[t] = (dGates @ self.W).reshape(batch_size, self.input_size)
            dh_prev = (dGates @ self.U).reshape(batch_size, self.hidden_size)
            dc_prev = dc_t * f[t]
        for grad in [dW, dU, db]:
            np.clip(grad, -5, 5, out=grad)
        self.dW = dW
        self.dU = dU
        self.db = db
        return dx, dh_prev, dc_prev

    def get_params_and_grads(self):
        return {
            'W': (self.W, self.dW),
            'U': (self.U, self.dU),
            'b': (self.b, self.db)
        }
