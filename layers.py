import numpy as np


class Layer_Dense:
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

class Embedding: # maybe refactor as integer token and lookup
    def __init__(self, vocab_size, embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.weights = np.random.randn(vocab_size, embed_dim)

    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs @ self.weights

    def backward(self, dvalues):
        # dvalues: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, _ = dvalues.shape

        # Flatten both inputs and dvalues
        inputs_flat = self.inputs.reshape(-1, self.vocab_size)      # (B*T, V)
        dvalues_flat = dvalues.reshape(-1, self.embed_dim)          # (B*T, E)

        # Matrix multiply to get gradient of weights: (V, E)
        self.dweights = inputs_flat.T @ dvalues_flat

        # Optional: gradient w.r.t. inputs (usually not needed for embeddings)
        self.dinputs = dvalues @ self.weights.T  # (B, T, V)


class PositionalEncoding:
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
    def __init__(self, embed_dim):
        self.embed_dim = embed_dim
        self.Qw = np.random.randn(embed_dim, embed_dim)
        self.Kw = np.random.randn(embed_dim, embed_dim)
        self.Vw = np.random.randn(embed_dim, embed_dim)
        self.softmax = Activation_Softmax()

    def forward(self, inputs):
        self.inputs = inputs  # (B, T, E)
        B, T, E = inputs.shape
        self.B, self.T, self.E = B, T, E
        scale = 1 / np.sqrt(self.embed_dim)

        # Project to Q, K, V
        self.Q = inputs @ self.Qw  # (B, T, E)
        self.K = inputs @ self.Kw  # (B, T, E)
        self.V = inputs @ self.Vw  # (B, T, E)

        # Compute attention scores
        self.scores = self.Q @ self.K.transpose(0, 2, 1) * scale  # (B, T, T)

        # Causal mask (broadcasted over batch)
        mask = np.triu(np.ones((T, T)), k=1) * -1e9
        self.scores += mask

        # Softmax
        self.softmax.forward(self.scores)
        self.attn_weights = self.softmax.output  # (B, T, T)

        # Weighted sum of V
        self.output = self.attn_weights @ self.V  # (B, T, E)

    def backward(self, dvalues):  # (B, T, E)
        B, T, E = self.B, self.T, self.E
        scale = 1 / np.sqrt(E)

        # ----- d(attn_weights @ V) → dAttnWeights and dV -----
        d_attn_weights = dvalues @ self.V.transpose(0, 2, 1)  # (B, T, E) @ (B, E, T)
        dV = self.attn_weights.transpose(0, 2, 1) @ dvalues    # (B, T, E)

        # ----- Softmax backward -----
        self.softmax.backward(d_attn_weights)  # sets self.softmax.dinputs
        dscores = self.softmax.dinputs  # (B, T, T)

        # ----- d(scores = Q @ K.T) -----
        dQ = dscores @ self.K  * scale  # (B, T, E)
        dK = dscores.transpose(0, 2, 1) @ self.Q * scale  # (B, T, E)

        # ----- d(Q = inputs @ Qw), etc. -----
        self.dQw = self.inputs.transpose(0, 2, 1) @ dQ  # (B, E, E) → reduce
        self.dKw = self.inputs.transpose(0, 2, 1) @ dK
        self.dVw = self.inputs.transpose(0, 2, 1) @ dV

        # Sum over batch to get total gradients
        self.dQw = np.sum(self.dQw, axis=0)  # (E, E)
        self.dKw = np.sum(self.dKw, axis=0)
        self.dVw = np.sum(self.dVw, axis=0)

        # ----- dInputs -----
        d_inputs_Q = dQ @ self.Qw.T  # (B, T, E)
        d_inputs_K = dK @ self.Kw.T
        d_inputs_V = dV @ self.Vw.T

        self.dinputs = d_inputs_Q + d_inputs_K + d_inputs_V  # (B, T, E)


class TransformerBlock:
    def __init__(self, embed_dim, ff_hidden_dim):
        self.attn = Attention(embed_dim)
        self.norm1 = LayerNorm(embed_dim)

        self.ff1 = Layer_Dense(embed_dim, ff_hidden_dim)
        self.ff2 = Layer_Dense(ff_hidden_dim, embed_dim)
        self.norm2 = LayerNorm(embed_dim)

    def forward(self, inputs):
        self.inputs = inputs  # (B, T, E)

        # ---- Self-Attention ----
        self.attn.forward(self.inputs)                    # attn.output: (B, T, E)
        self.res1 = self.inputs + self.attn.output        # residual
        self.norm1.forward(self.res1)                     # norm1.output: (B, T, E)

        # ---- Feedforward ----
        self.ff1.forward(self.norm1.output.reshape(self.norm1.output.shape[0] * self.norm1.output.shape[1], self.norm1.output.shape[2]))
        self.relu_mask = self.ff1.output > 0
        self.ff1.output = np.maximum(0, self.ff1.output)

        self.ff2.forward(self.ff1.output)                 # ff2.output: (B, T, E)
        self.res2 = self.norm1.output + self.ff2.output.reshape(self.norm1.output.shape)   # residual
        self.norm2.forward(self.res2)

        self.output = self.norm2.output

    def backward(self, dvalues):
        # ---- Norm2 Backward ----
        self.norm2.backward(dvalues)                      # sets norm2.dinputs

        # ---- Feedforward Backward ----
        flattened_dvalues = self.norm2.dinputs.reshape(self.norm2.dinputs.shape[0] * self.norm2.dinputs.shape[1],
                                               self.norm2.dinputs.shape[2])  # (B*T, E)
        self.ff2.backward(flattened_dvalues)   # sets ff2.dinputs: (B*T, H)

        dff1 = self.ff2.dinputs                # (B*T, H)
        dff1[~self.relu_mask] = 0              #  shapes match now

        self.ff1.backward(dff1)                           # sets ff1.dinputs

        dnorm1 = self.norm2.dinputs + self.ff1.dinputs.reshape(self.norm2.dinputs.shape)                 # residual connection

        # ---- Norm1 Backward ----
        self.norm1.backward(dnorm1)
        dres1 = self.norm1.dinputs

        # ---- Attention Backward ----
        self.attn.backward(dres1)
        self.dinputs = self.attn.dinputs + self.inputs    # final residual connection

