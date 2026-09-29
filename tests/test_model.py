import numpy as np
import pytest

from lstms.Activation import Activation
from lstms.Generator import Generator
from lstms.Lstm import Lstm
from lstms.Tokenizer import Tokenizer
from lstms.Trainer import Trainer

CORPUS = "hello world, hello lstm. "


def summed_loss(lstm, inputs, targets):
    """Sum of per-step cross-entropy losses (what the dYs are gradients of)."""
    outputs, _, _ = lstm.forward(inputs)
    total = 0.0
    for logits, target in zip(outputs, targets):
        total -= np.log(Activation.softmax(logits)[target, 0])
    return total


def test_gradients_match_numerical_gradients():
    rng = np.random.default_rng(0)
    np.random.seed(0)
    vocab, hidden = 5, 4
    lstm = Lstm(input_size=vocab, hidden_size=hidden)
    # Larger random weights and non-zero biases exercise every path.
    for name in ["Wf", "Wi", "WC", "Wo", "bf", "bi", "bC", "bo"]:
        param = getattr(lstm.cell, name)
        param[...] = rng.normal(scale=0.5, size=param.shape)
    lstm.Wy[...] = rng.normal(scale=0.5, size=lstm.Wy.shape)
    lstm.by[...] = rng.normal(scale=0.5, size=lstm.by.shape)

    inputs = [0, 3, 1, 4, 2, 2]
    targets = [3, 1, 4, 2, 2, 0]

    outputs, _, caches = lstm.forward(inputs)
    _, dYs = lstm.compute_loss(outputs, targets)
    dWy, dby, cell_grads = lstm.backward(dYs, caches)

    analytic = {"Wy": dWy, "by": dby}
    params = {"Wy": lstm.Wy, "by": lstm.by}
    for name in ["Wf", "Wi", "WC", "Wo", "bf", "bi", "bC", "bo"]:
        analytic[name] = cell_grads["d" + name]
        params[name] = getattr(lstm.cell, name)
    assert len(params) == 10

    eps = 1e-5
    for name, param in params.items():
        numeric = np.zeros_like(param)
        for idx in np.ndindex(param.shape):
            original = param[idx]
            param[idx] = original + eps
            plus = summed_loss(lstm, inputs, targets)
            param[idx] = original - eps
            minus = summed_loss(lstm, inputs, targets)
            param[idx] = original
            numeric[idx] = (plus - minus) / (2 * eps)
        a = analytic[name]
        rel_error = np.linalg.norm(a - numeric) / (np.linalg.norm(a) + np.linalg.norm(numeric))
        assert rel_error < 1e-6, f"{name}: relative error {rel_error:.2e}"


def test_tokenizer_round_trip():
    tokenizer = Tokenizer(CORPUS)
    encoded = tokenizer.encode(CORPUS)
    assert all(0 <= i < tokenizer.vocab_size for i in encoded)
    assert tokenizer.decode(encoded) == CORPUS
    assert tokenizer.vocab_size == len(set(CORPUS))


def make_sequences(tokenizer, text, seq_length):
    encoded = tokenizer.encode(text)
    inputs = [encoded[i:i + seq_length] for i in range(len(encoded) - seq_length)]
    targets = [encoded[i + 1:i + seq_length + 1] for i in range(len(encoded) - seq_length)]
    return inputs, targets


def test_loss_decreases_on_tiny_corpus():
    np.random.seed(1)
    tokenizer = Tokenizer(CORPUS)
    inputs, targets = make_sequences(tokenizer, CORPUS * 2, seq_length=5)
    lstm = Lstm(input_size=tokenizer.vocab_size, hidden_size=16)
    trainer = Trainer(lstm, learning_rate=0.1, clip_value=5.0)
    losses = trainer.train(inputs, targets, epochs=15, print_every=100)
    assert all(np.isfinite(losses))
    assert losses[-1] < 0.7 * losses[0]


def test_generate_returns_seed_plus_n_vocab_characters():
    np.random.seed(2)
    tokenizer = Tokenizer(CORPUS)
    lstm = Lstm(input_size=tokenizer.vocab_size, hidden_size=8)
    generator = Generator(lstm, tokenizer)
    for temperature in (0.5, 1.0, 1.5):
        text = generator.generate("hel", 20, temperature=temperature)
        assert text.startswith("hel")
        assert len(text) == len("hel") + 20
        assert set(text) <= set(tokenizer.chars)
