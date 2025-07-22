import numpy as np

from tokenization import Tokenizer
from layers import Embedding, PositionalEncoding, TransformerBlock, Layer_Dense
from activation import Activation_Softmax
from loss_functions import Loss_CategoricalCrossentropy
from optimizers import Optimizer_SGD


def next_number(prevn, vocab_size, rand_add):
    return ((np.sum(prevn) + np.random.poisson(rand_add,1)) % vocab_size) + 1

def simulate_sequence(n, sl, vocab_size, window, rand_add):
    sequence = np.empty(n * sl, dtype=int)
    sequence[:window] = np.random.randint(0, vocab_size, size=window)  # seed

    for i in range(window, n * sl):
        sequence[i] = next_number(sequence[i-window:i], vocab_size-1, rand_add)

    return sequence.reshape(n, sl)


if __name__=='__main__':
    np.random.seed(42)
    N = 10
    SL = 4
    WINDOW = 1
    RAND_ADD = 1
    EMBED_DIM = 32
    DENSE_NEURONS = 64
    EPOCHS = 1000
    LR = 1e-4
    DECAY = 5e-3
    MOMENTUM = 1e-2
    
    corpus = ['red blue green orange pink yellow']
    tok = Tokenizer()
    tok.fit(corpus)
    
    X_tokens = simulate_sequence(N, SL, tok.vocab_size, WINDOW, RAND_ADD)

    X_in = X_tokens[:, :-1]
    SL -= 1 # in reality it is 1 shorter
    y = X_tokens[:, 1:]
    X = tok.sparse_tokenize(X_in)
    print(y)

    embed = Embedding(len(tok.word_token), EMBED_DIM)
    position = PositionalEncoding(SL, EMBED_DIM)
    transformer1 = TransformerBlock(EMBED_DIM, DENSE_NEURONS)
    output = Layer_Dense(EMBED_DIM, tok.vocab_size)
    output_activation = Activation_Softmax()
    loss_function = Loss_CategoricalCrossentropy()
    optimizer = Optimizer_SGD(learning_rate = LR, decay=DECAY, momentum=MOMENTUM)

    for epoch in range(EPOCHS+1):
        embed.forward(X)
        position.forward(embed.output)
        transformer1.forward(position.output)
        output.forward(transformer1.output.reshape(N * SL, EMBED_DIM))
        output_activation.forward(output.output)
        logits = output_activation.output.reshape(-1, tok.vocab_size)
        targets = y.reshape(-1)
        loss = loss_function.calculate(logits, targets)
        predictions = np.argmax(logits, axis=-1).reshape(N, SL)
        accuracy = np.mean(predictions == y)

        if not epoch % 10:
            print(f'epoch: {epoch}, '+
                  f'acc: {accuracy:.3f}, ' +
                  f'loss: {loss:.3f}, ' +
                  f'lr: {optimizer.current_learning_rate}')

        loss_function.backward_with_softmax(logits, targets)
        output.backward(loss_function.dinputs)
        transformer1.backward(output.dinputs.reshape(N, SL, EMBED_DIM))
        position.backward(transformer1.dinputs)
        embed.backward(position.dinputs)

        optimizer.pre_update_params()
        optimizer.update_weights(embed)
        optimizer.update_attention(transformer1.attn)
        optimizer.update_layernorm(transformer1.norm1)
        optimizer.update_params(transformer1.ff1)
        optimizer.update_params(transformer1.ff2)
        optimizer.update_layernorm(transformer1.norm2)
        optimizer.update_params(output)
        optimizer.post_update_params()

    print(tok.int_detokenize(predictions)[:10])
