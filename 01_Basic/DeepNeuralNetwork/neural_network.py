import numpy as np
from tqdm import tqdm


class Activation:
    def __init__(self):
        pass

    @staticmethod
    def relu(x):
        # return np.max(0, x)
        return np.vectorize(lambda x_: max(0, x_))(x)

    @staticmethod
    def softmax(x):
        s = np.exp(x - np.expand_dims(x.max(axis=1), axis=1))

        return s / np.expand_dims(s.sum(axis=1), axis=1)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


class Derivative:
    def __init(self):
        pass

    @staticmethod
    def relu(x):
        return np.vectorize(lambda x_: 0 if x_ < 0 else 1)(x)
        # return 0 if x < 0 else 1

    @staticmethod
    def softmax(x):
        # s = x.reshape(-1, 1)
        # return np.diagflat(s) - np.dot(s, s.T)
        return Derivative.sigmoid(x)

    @staticmethod
    def sigmoid(x):
        return x * (1 - x)


class Neuron:
    def __init__(self, n_input, n_output, activation='sigmoid'):
        self.w = np.random.randn(n_input + 1, n_output) * np.sqrt(2 / (n_input + n_output))

        if activation == 'relu':
            self.activation = Activation.relu
            self.derivative = Derivative.relu
        elif activation == 'softmax':
            self.activation = Activation.softmax
            self.derivative = Derivative.softmax
        else:
            self.activation = Activation.sigmoid
            self.derivative = Derivative.sigmoid

    def forward(self, x, append_bias=True):
        if append_bias:
            x = Neuron.append_bias(x)
        h = self.activation(np.matmul(x, self.w))

        return h

    def backward(self, x, y, append_bias=True):
        if append_bias:
            x = Neuron.append_bias(x)
        h = self.activation(np.matmul(x, self.w))
        delta_out = (y - h) * self.derivative(h)

        wn = np.matmul(x.T, delta_out)

        return wn, delta_out, h

    def train(self, x, y, learning_rate=0.01, append_bias=True):
        wn, d_out, h = self.backward(x, y, append_bias=append_bias)

        loss = np.sum(np.abs(h - y))

        self.w = self.w + wn * learning_rate

        return loss

    @staticmethod
    def append_bias(x):
        return np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)


class ANN:
    def __init__(self, layer=[(), 'relu']):
        self.layers = []
        for l in layer:
            self.layers.append(Neuron(l[0][0], l[0][1], activation=l[1]))

    def forward(self, x):
        h = self.layers[0].forward(x)

        for i in range(1, len(self.layers)):
            h = self.layers[i].forward(h, append_bias=False)

        return h

    def train(self, x, y, learning_rate=0.01):
        xn = Neuron.append_bias(x)

        hs = []
        hs.append(self.layers[0].forward(xn, append_bias=False))
        for i in range(1, len(self.layers)):
            hs.append(self.layers[i].forward(hs[-1], append_bias=False))

        ws = []
        d_outs = []
        wn, d_out, h = self.layers[-1].backward(hs[-2], y, append_bias=False)
        ws.append(wn)
        d_outs.append(d_out)

        for i in range(len(self.layers) - 2, 0, -1):
            d_out = np.matmul(d_outs[0], self.layers[i + 1].w.T) * self.layers[i].derivative(hs[i])
            wn = np.matmul(hs[i - 1].T, d_out)

            d_outs.insert(0, d_out)
            ws.insert(0, wn)

        d_out = np.matmul(d_outs[0], self.layers[1].w.T) * self.layers[0].derivative(hs[0])
        wn = np.matmul(xn.T, d_out)

        ws.insert(0, wn)
        d_outs.insert(0, d_out)

        for i in range(len(self.layers)):
            self.layers[i].w += ws[i] * learning_rate

        loss = np.sum(np.abs(y - hs[-1]))

        return loss

    def fit(self, x, y, learning_rate=0.01, epoch=1000, batch_size=1000, return_history=False):
        losses = []

        for i in range(epoch):
            loss = 0
            total_batch = x.shape[0] // batch_size
            with tqdm(total=total_batch) as pbar:
                for j in range(total_batch):
                    s_idx = j * batch_size
                    e_idx = (j + 1) * batch_size
                    x_batch = x[s_idx:e_idx]
                    y_batch = y[s_idx:e_idx]
                    loss += self.train(x_batch, y_batch, learning_rate=learning_rate)

                    pbar.set_description("Epoch: {:02d}/{:02d} , Loss: {:.3f}".format(i + 1, epoch, loss / (j + 1)/batch_size))
                    pbar.update()

            loss /= total_batch / batch_size
            losses.append(loss)

        if return_history:
            return losses
        elif losses:
            return losses[-1]
        else:
            return None
