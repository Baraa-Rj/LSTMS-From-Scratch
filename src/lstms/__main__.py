"""Train the character-level LSTM on a text file and print generated samples.

Run from the repository root:

    python -m lstms                 # defaults, reproducible (seed 0)
    python -m lstms --epochs 20 --hidden-size 128 --seed 1
"""
import argparse
import math
import random

import numpy as np

from .Tokenizer import Tokenizer
from .Lstm import Lstm
from .Trainer import Trainer
from .Generator import Generator

DEFAULT_SAMPLE_SEED = "ROMEO:"
SAMPLE_TEMPERATURES = (0.5, 1.0, 1.5)
HELDOUT_FRACTION = 0.1


def build_windows(encoded, starts, seq_length):
    """Input/target pairs: the target is the input shifted one character ahead."""
    inputs_list = [encoded[i:i + seq_length] for i in starts]
    targets_list = [encoded[i + 1:i + seq_length + 1] for i in starts]
    return inputs_list, targets_list


def prepare_training_data(text, seq_length, train_windows, heldout_windows, rng=random):
    """Split the text by position, then sample windows from each part.

    The last HELDOUT_FRACTION of the characters supplies the held-out windows
    and everything before it supplies the training windows, so no held-out
    character is ever seen during training.
    """
    tokenizer = Tokenizer(text)
    encoded = tokenizer.encode(text)

    split = int(len(encoded) * (1 - HELDOUT_FRACTION))
    train_starts = range(0, split - seq_length)
    heldout_starts = range(split, len(encoded) - seq_length)
    if len(train_starts) < 1 or len(heldout_starts) < 1:
        raise ValueError("text is too short for the requested sequence length")

    train = build_windows(
        encoded, rng.sample(train_starts, min(train_windows, len(train_starts))), seq_length)
    heldout = build_windows(
        encoded, rng.sample(heldout_starts, min(heldout_windows, len(heldout_starts))), seq_length)
    return train, heldout, tokenizer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m lstms",
        description="Train a character-level LSTM (pure NumPy) and sample text from it.")
    parser.add_argument("--input", default="input.txt", help="training text file")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seq-length", type=int, default=10,
                        help="characters per training window")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--clip", type=float, default=5.0,
                        help="element-wise gradient clipping threshold")
    parser.add_argument("--train-windows", type=int, default=10000,
                        help="number of training windows sampled from the text")
    parser.add_argument("--heldout-windows", type=int, default=2000,
                        help="number of held-out windows used for evaluation")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for window sampling, weight init and generation")
    parser.add_argument("--sample-seed", default=None,
                        help=f"text to start generation from (default: {DEFAULT_SAMPLE_SEED!r})")
    parser.add_argument("--sample-length", type=int, default=120,
                        help="characters to generate per sample")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 50)
    print("Character-Level LSTM Training")
    print("=" * 50)
    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    print("\n1. Preparing data...")
    (inputs_list, targets_list), (heldout_inputs, heldout_targets), tokenizer = \
        prepare_training_data(text, args.seq_length, args.train_windows, args.heldout_windows)
    print(f"   Corpus: {len(text)} characters, vocabulary of {tokenizer.vocab_size}")
    print(f"   Training windows: {len(inputs_list)} (from the first {1 - HELDOUT_FRACTION:.0%} of the text)")
    print(f"   Held-out windows: {len(heldout_inputs)} (from the last {HELDOUT_FRACTION:.0%})")
    print(f"   Sequence length: {args.seq_length}")

    print("\n2. Initializing LSTM...")
    lstm = Lstm(input_size=tokenizer.vocab_size, hidden_size=args.hidden_size)
    trainer = Trainer(lstm, learning_rate=args.learning_rate, clip_value=args.clip)
    print(f"   Hidden size: {args.hidden_size}")
    print(f"   Learning rate: {args.learning_rate}, gradient clipping: {args.clip}")
    print(f"   Seed: {args.seed}")

    print("\n3. Training (per-character cross-entropy, in nats)...")
    print(f"   Uniform-guess baseline: {math.log(tokenizer.vocab_size):.4f}")
    print(f"   Before training: held-out {trainer.evaluate(heldout_inputs, heldout_targets):.4f}")
    for epoch in range(1, args.epochs + 1):
        train_loss = trainer.train(inputs_list, targets_list, epochs=1, print_every=None)[0]
        heldout_loss = trainer.evaluate(heldout_inputs, heldout_targets)
        print(f"   Epoch {epoch}/{args.epochs}: train {train_loss:.4f}, held-out {heldout_loss:.4f}")

    print("\n4. Generating text samples...")
    generator = Generator(lstm, tokenizer)
    sample_seed = args.sample_seed or DEFAULT_SAMPLE_SEED
    if not set(sample_seed) <= set(tokenizer.chars):
        sample_seed = text[:len(DEFAULT_SAMPLE_SEED)]
    for temperature in SAMPLE_TEMPERATURES:
        sample = generator.generate(sample_seed, args.sample_length, temperature=temperature)
        print(f"\n--- temperature {temperature} ---")
        print(sample)

    print("\n" + "=" * 50)
    print("Done")
    print("=" * 50)


if __name__ == "__main__":
    main()
