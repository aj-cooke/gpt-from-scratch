import numpy as np

from activation import Activation_Softmax


class Layer_Dense:
    """
    Feed forward neural network layer. AKA fully connected
    layer or feed forward
    """
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class Embedding:
    """
    Embedding matrix for generative pretrained transformer.
    Represents one trainable vector of dimension embed_dim for each
    token in the corpus.
    """
    def __init__(self, vocab_size, embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.weights = np.random.randn(vocab_size, embed_dim)

    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs @ self.weights

    def backward(self, dvalues):
        batch_size, seq_len, _ = dvalues.shape

        inputs_flat = self.inputs.reshape(-1, self.vocab_size)
        dvalues_flat = dvalues.reshape(-1, self.embed_dim)

        self.dweights = inputs_flat.T @ dvalues_flat

        self.dinputs = dvalues @ self.weights.T  # usually not needed as embedding is first layer


class PositionalEncoding:
    """
    Layer without trainable parameters that applies cos or sin
    periodic function to each dimension of embedding vectors
    """
    def __init__(self, seq_len, embed_dim):
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        pos = np.arange(seq_len)[:, np.newaxis]         # (seq_len, 1)
        i = np.arange(embed_dim)[np.newaxis, :]         # (1, embed_dim)
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / embed_dim)
        angle_rads = pos * angle_rates

        self.output = np.zeros((seq_len, embed_dim))
        self.output[:, 0::2] = np.sin(angle_rads[:, 0::2])  # even dimensions
        self.output[:, 1::2] = np.cos(angle_rads[:, 1::2])  # odd dimensions

    def forward(self, inputs):
        self.inputs = inputs
        self.output = self.output + self.inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()


class Attention:
    """
    See: https://arxiv.org/abs/1706.03762
    First layer in a transformer block. Creates Q, K, and V
    matrices that attempt to capture "queries", "keys", and "values"
    of tokens.
    """
    def __init__(self, embed_dim):
        self.embed_dim = embed_dim
        self.Qw = np.random.randn(embed_dim, embed_dim)
        self.Kw = np.random.randn(embed_dim, embed_dim)
        self.Vw = np.random.randn(embed_dim, embed_dim)
        self.softmax = Activation_Softmax()

    def forward(self, inputs):
        self.inputs = inputs
        B, T, E = inputs.shape
        self.B, self.T, self.E = B, T, E
        scale = 1 / np.sqrt(self.embed_dim)

        self.Q = inputs @ self.Qw
        self.K = inputs @ self.Kw
        self.V = inputs @ self.Vw

        self.scores = self.Q @ self.K.transpose(0, 2, 1) * scale

        mask = np.triu(np.ones((T, T)), k=1) * -1e9 # no looking ahead!
        self.scores += mask

        self.softmax.forward(self.scores)
        self.attn_weights = self.softmax.output

        self.output = self.attn_weights @ self.V

    def backward(self, dvalues):
        B, T, E = self.B, self.T, self.E
        scale = 1 / np.sqrt(E)

        d_attn_weights = dvalues @ self.V.transpose(0, 2, 1)
        dV = self.attn_weights.transpose(0, 2, 1) @ dvalues

        self.softmax.backward(d_attn_weights)
        dscores = self.softmax.dinputs

        dQ = dscores @ self.K  * scale
        dK = dscores.transpose(0, 2, 1) @ self.Q * scale

        self.dQw = self.inputs.transpose(0, 2, 1) @ dQ
        self.dKw = self.inputs.transpose(0, 2, 1) @ dK
        self.dVw = self.inputs.transpose(0, 2, 1) @ dV

        self.dQw = np.sum(self.dQw, axis=0)
        self.dKw = np.sum(self.dKw, axis=0)
        self.dVw = np.sum(self.dVw, axis=0)

        d_inputs_Q = dQ @ self.Qw.T
        d_inputs_K = dK @ self.Kw.T
        d_inputs_V = dV @ self.Vw.T

        self.dinputs = d_inputs_Q + d_inputs_K + d_inputs_V


class LayerNorm:
    """
    Normalize outputs to easier-to-handle values between layers.
    Different from batch normalizing as location and scale parameters
    are tunable.
    """
    def __init__(self, embed_dim, eps=1e-5):
        self.eps = eps
        self.gamma = np.ones((1, 1, embed_dim))
        self.beta = np.zeros((1, 1, embed_dim))

    def forward(self, inputs):
        self.inputs = inputs

        self.mean = np.mean(inputs, axis=-1, keepdims=True)
        self.var = np.var(inputs, axis=-1, keepdims=True)

        self.std = np.sqrt(self.var + self.eps)
        self.norm = (inputs - self.mean) / self.std
        self.output = self.gamma * self.norm + self.beta

    def backward(self, dvalues):
        B, T, E = dvalues.shape
        
        self.dgamma = np.sum(dvalues * self.norm, axis=(0,1), keepdims=True)
        self.dbeta = np.sum(dvalues, axis=(0,1), keepdims=True)

        dnorm = dvalues * self.gamma

        dx = (1. / self.std) * (
            dnorm
            - np.mean(dnorm, axis=-1, keepdims=True)
            - self.norm * np.mean(dnorm * self.norm, axis=-1, keepdims=True)
        )

        self.dinputs = dx


class TransformerBlock:
    """
    Putting all the pieces together for a transformer.
    Attention -> LayerNorm -> 2 Dense layers -> LayerNorm
    """
    def __init__(self, embed_dim, ff_hidden_dim):
        self.attn = Attention(embed_dim)
        self.norm1 = LayerNorm(embed_dim)

        self.ff1 = Layer_Dense(embed_dim, ff_hidden_dim)
        self.ff2 = Layer_Dense(ff_hidden_dim, embed_dim)
        self.norm2 = LayerNorm(embed_dim)

    def forward(self, inputs):
        self.inputs = inputs

        self.attn.forward(self.inputs)                   
        self.res1 = self.inputs + self.attn.output # LayerNorm and Add
        self.norm1.forward(self.res1)

        self.ff1.forward(self.norm1.output.reshape(self.norm1.output.shape[0] * self.norm1.output.shape[1], self.norm1.output.shape[2]))
        self.relu_mask = self.ff1.output > 0
        self.ff1.output = np.maximum(0, self.ff1.output)

        self.ff2.forward(self.ff1.output)
        self.res2 = self.norm1.output + self.ff2.output.reshape(self.norm1.output.shape)
        self.norm2.forward(self.res2)

        self.output = self.norm2.output

    def backward(self, dvalues):
        self.norm2.backward(dvalues)

        flattened_dvalues = self.norm2.dinputs.reshape(self.norm2.dinputs.shape[0] * self.norm2.dinputs.shape[1],
                                               self.norm2.dinputs.shape[2])
        self.ff2.backward(flattened_dvalues)

        dff1 = self.ff2.dinputs
        dff1[~self.relu_mask] = 0

        self.ff1.backward(dff1)

        dnorm1 = self.norm2.dinputs + self.ff1.dinputs.reshape(self.norm2.dinputs.shape)

        self.norm1.backward(dnorm1)
        dres1 = self.norm1.dinputs

        self.attn.backward(dres1)
        self.dinputs = self.attn.dinputs + self.inputs

