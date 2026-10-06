# Character-Level LSTM from Scratch

A character-level language model built on a single-layer LSTM, written in NumPy
without a deep-learning framework. The forward pass, the backward pass
(backpropagation through time), the optimizer step and the sampler are all
implemented by hand, and the hand-derived gradients are checked against
numerical gradients in the test suite.

This is an educational implementation. It exists to show how an LSTM works end
to end, not to produce good text or to compete with PyTorch.

## What is implemented

| Module | What it does |
|---|---|
| `Cell.py` | One LSTM step: forget, input, candidate and output gates. `forward` returns the new hidden and cell state plus a cache; `backward` returns gradients for all eight cell tensors and for the previous state. |
| `Lstm.py` | Unrolls the cell over a sequence, projects each hidden state to vocabulary logits, computes softmax cross-entropy, and runs backpropagation through time over the whole sequence. |
| `Trainer.py` | Stochastic gradient descent, one sequence per update, with element-wise gradient clipping. `evaluate` reports loss without updating weights. |
| `Generator.py` | Samples text one character at a time, with a temperature parameter. |
| `Tokenizer.py` | Character-to-index mapping built from the training text. |
| `Activation.py` | Sigmoid, tanh and a numerically stable softmax. |
| `__main__.py` | Command-line training script: data split, training loop, evaluation, samples. |

NumPy is the only runtime dependency. There is no automatic differentiation.

## The model

For input `x_t` (a one-hot character), previous hidden state `h_{t-1}` and
previous cell state `C_{t-1}`, with `z_t = [h_{t-1}; x_t]`:

```
f_t  = sigmoid(W_f z_t + b_f)        forget gate
i_t  = sigmoid(W_i z_t + b_i)        input gate
C~_t = tanh(W_C z_t + b_C)           candidate cell state
o_t  = sigmoid(W_o z_t + b_o)        output gate

C_t  = f_t * C_{t-1} + i_t * C~_t    cell state
h_t  = o_t * tanh(C_t)               hidden state
y_t  = W_y h_t + b_y                 logits over the vocabulary
```

Each gate matrix has shape `(hidden, hidden + vocab)`. With the default
configuration below (vocabulary 65, hidden size 64) the model has
**37,505 parameters**: 4 x (64 x 129 + 64) = 33,280 in the cell, plus
65 x 64 + 65 = 4,225 in the output layer.

Weights are drawn from a standard normal and scaled by
`1 / (vocab + hidden)`; biases start at zero.

## Training

- **Data.** The text is cut into windows of `seq_length` characters. The target
  for a window is the same window shifted one character ahead, so the model
  predicts the next character at every position.
- **State.** Hidden and cell state start at zero for every window, so
  gradients flow through at most `seq_length` steps.
- **Loss.** Softmax cross-entropy per character, in nats. Reported losses are
  means per character. The gradient used for the update is the gradient of the
  loss summed over the window.
- **Backward pass.** `Lstm.backward` walks the window from the last step to
  the first, carrying `dh` and `dC` backwards and accumulating the gradients
  of the shared weights across time steps.
- **Update.** Plain SGD: each gradient is clipped element-wise to
  `[-clip, clip]` and subtracted, scaled by the learning rate. One window per
  update; windows are visited in the same order every epoch.
- **Evaluation.** The last 10% of the text is held out. Training windows are
  sampled from the first 90% and held-out windows from the last 10%, so no
  held-out character is seen during training.

## Gradient check

`tests/test_model.py::test_gradients_match_numerical_gradients` compares the
analytic gradients from `Lstm.backward` with central-difference estimates
(`eps = 1e-5`) for **every element of all ten parameter tensors**: `W_f`,
`W_i`, `W_C`, `W_o`, `b_f`, `b_i`, `b_C`, `b_o`, `W_y` and `b_y`. It uses a
small model (vocabulary 5, hidden size 4, 185 parameters in total), random
weights and a six-step sequence, and requires a relative error below `1e-6`
for each tensor.

The largest relative error measured is `3.4e-09` (on `W_f`).

## Results

All numbers in this section come from one run of the code in this repository
with its default settings:

```bash
python -m lstms
```

| Setting | Value |
|---|---|
| Corpus | `input.txt`, Tiny Shakespeare: 1,115,394 characters, 65 distinct |
| Training windows | 10,000, sampled from the first 90% of the text |
| Held-out windows | 2,000, sampled from the last 10% |
| Sequence length | 10 characters |
| Hidden size | 64 |
| Optimizer | SGD, learning rate 0.01, clipping 5.0, one window per update |
| Epochs | 10 |
| Seed | 0 |
| Environment | Python 3.13.16, NumPy 2.5.3, one CPU core |
| Wall time | 2 min 51 s |

Loss per character, in nats (a uniform guess over 65 characters scores
`ln 65 = 4.1744`):

| Epoch | Training | Held-out |
|---:|---:|---:|
| before training | | 4.1744 |
| 1 | 3.0866 | 2.6032 |
| 2 | 2.4407 | 2.3267 |
| 3 | 2.2748 | 2.2155 |
| 4 | 2.1855 | 2.1658 |
| 5 | 2.1260 | 2.1337 |
| 6 | 2.0794 | 2.1105 |
| 7 | 2.0409 | 2.0941 |
| 8 | 2.0084 | 2.0817 |
| 9 | 1.9799 | 2.0717 |
| 10 | 1.9542 | 2.0636 |

The training column is the mean loss recorded while the weights were being
updated during that epoch; the held-out column is measured after the epoch.
A held-out loss of 2.0636 nats is 2.98 bits per character, a perplexity of
about 7.9.

Samples from the trained model, 120 characters each, all seeded with `ROMEO:`:

Temperature 0.5

```
ROMEO:
May, be is a fore of this of the been of deest the forter I then your hour peelfing we come the the bresentless in the 
```

Temperature 1.0

```
ROMEO:
And your a the sulate
The all your from', murd
Bulliest let.

KINGBULE:
Say, coulf tendee.

HSRILOMEM:
Paroyce
M'sr no 
```

Temperature 1.5

```
ROMEO:
Vhores upoozblaked exhse:
ThiscOGtO
Pray lordARf, anCfecl;wer;
Are:
A
vord, miscafeRit: culmany's quvipuising;, is of e
```

What this shows: after ten short epochs the model has picked up the layout of
a play script (a speaker name in capitals, a colon, a new line), common short
words and plausible letter sequences. Most longer words are not real words,
and nothing is coherent beyond a few characters. Lower temperature gives more
repetitive, more word-like output; higher temperature gives more varied and
more broken output. The model is small and far from converged, as the
still-falling loss shows.

Running the command twice with the same seed produced identical output on the
machine above. Other platforms or NumPy builds may differ in the last digits.

## Installation

```bash
git clone https://github.com/Baraa-Rj/LSTMS-From-Scratch
cd LSTMS-From-Scratch
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
```

Python 3.10 or newer.

## Running the tests

```bash
python -m pytest
```

23 tests:

- the gradient check described above;
- loss decreases when training on a tiny corpus;
- `evaluate` returns the mean per-character loss and leaves the weights unchanged;
- the tokenizer round-trips text;
- generation returns the seed plus exactly the requested number of in-vocabulary characters;
- generation feeds each seed character once (a regression test for an earlier bug);
- generation rejects an empty seed, a seed with characters outside the vocabulary, and a non-positive temperature;
- training and held-out windows come from disjoint parts of the text, and targets are the inputs shifted by one character;
- the training script runs end to end and is reproducible for a fixed seed;
- the package declares its NumPy dependency and every module imports.

Continuous integration runs the suite on Python 3.11, 3.12 and 3.13.

## Training

Run from the repository root, because the script reads `input.txt` from the
current directory by default:

```bash
python -m lstms                                    # the run reported above
python -m lstms --epochs 20 --hidden-size 128      # a larger, slower run
python -m lstms --input my_text.txt --seed 1       # another corpus
```

| Option | Default | Meaning |
|---|---|---|
| `--input` | `input.txt` | training text file |
| `--epochs` | 10 | passes over the training windows |
| `--seq-length` | 10 | characters per window |
| `--hidden-size` | 64 | LSTM hidden units |
| `--learning-rate` | 0.01 | SGD step size |
| `--clip` | 5.0 | element-wise gradient clipping threshold |
| `--train-windows` | 10000 | training windows sampled from the text |
| `--heldout-windows` | 2000 | held-out windows used for evaluation |
| `--seed` | 0 | seed for window sampling, weight initialisation and sampling |
| `--sample-seed` | `ROMEO:` | text the samples start from |
| `--sample-length` | 120 | characters generated per sample |

## Generating text

The training script prints three samples at temperatures 0.5, 1.0 and 1.5 when
it finishes. Trained weights are not saved to disk, so generation from Python
means training in the same process:

```python
import numpy as np

from lstms.Generator import Generator
from lstms.Lstm import Lstm
from lstms.Tokenizer import Tokenizer
from lstms.Trainer import Trainer

text = open("input.txt", encoding="utf-8").read()[:20000]
tokenizer = Tokenizer(text)
encoded = tokenizer.encode(text)

seq_length = 10
starts = range(0, len(encoded) - seq_length, seq_length)
inputs = [encoded[i:i + seq_length] for i in starts]
targets = [encoded[i + 1:i + seq_length + 1] for i in starts]

np.random.seed(0)
lstm = Lstm(input_size=tokenizer.vocab_size, hidden_size=64)
Trainer(lstm, learning_rate=0.01, clip_value=5.0).train(inputs, targets, epochs=5)

print(Generator(lstm, tokenizer).generate("ROMEO:", length=200, temperature=0.8))
```

`temperature` rescales the predicted distribution before sampling: values below
1 sharpen it, values above 1 flatten it.

## Repository structure

```
src/lstms/
    Activation.py    sigmoid, tanh, softmax
    Cell.py          LSTM cell, forward and backward
    Lstm.py          sequence unrolling, loss, backpropagation through time
    Tokenizer.py     character vocabulary
    Trainer.py       SGD with gradient clipping, evaluation
    Generator.py     temperature sampling
    __main__.py      training script (python -m lstms)
tests/               pytest suite, including the gradient check
input.txt            Tiny Shakespeare corpus
.github/workflows/   CI: tests on Python 3.11, 3.12 and 3.13
```

## Limitations

- **Small and slow.** One layer, one sequence per update, pure Python loops
  over time steps, CPU only. The reported run trains on 10,000 windows, about
  1% of the windows available in the corpus.
- **Short context during training.** State is reset for every 10-character
  window, so the model is never trained to carry information further than that.
- **Basic optimisation.** Plain SGD with a fixed learning rate; no momentum,
  Adam, learning-rate schedule, mini-batching or per-epoch shuffling.
- **Modest results.** The samples above are representative. The model is not
  trained to convergence and no hyperparameter search was done.
- **No checkpoints.** Trained weights are not saved or loaded.
- **Character level only.** No subword tokenisation, no embeddings: inputs are
  one-hot vectors.

## References

- S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory", Neural Computation, 1997.
- C. Olah, "Understanding LSTM Networks", 2015.
- A. Karpathy, "The Unreasonable Effectiveness of Recurrent Neural Networks", 2015. The Tiny Shakespeare corpus used here comes from that work.
