"""End-to-end checks on the training script (python -m lstms)."""
import random

import pytest

from lstms.__main__ import HELDOUT_FRACTION, main, prepare_training_data

TEXT = "ROMEO: but soft, what light through yonder window breaks? " * 8


def test_training_and_heldout_windows_come_from_disjoint_parts_of_the_text():
    # The last 10% of this text is all "B", so the origin of every window is visible.
    text = "A" * 270 + "B" * 30
    assert int(len(text) * (1 - HELDOUT_FRACTION)) == 270
    (inputs, targets), (heldout_inputs, heldout_targets), tokenizer = prepare_training_data(
        text, 5, train_windows=50, heldout_windows=20, rng=random.Random(0))
    a, b = tokenizer.encode("AB")

    assert len(inputs) == len(targets) == 50
    assert len(heldout_inputs) == len(heldout_targets) == 20
    assert all(set(window) == {a} for window in inputs + targets)
    assert all(set(window) == {b} for window in heldout_inputs + heldout_targets)


def test_targets_are_the_inputs_shifted_one_character_ahead():
    (inputs, targets), _, _ = prepare_training_data(
        TEXT, 5, train_windows=30, heldout_windows=5, rng=random.Random(0))
    assert all(len(i) == len(t) == 5 for i, t in zip(inputs, targets))
    assert all(i[1:] == t[:-1] for i, t in zip(inputs, targets))


def test_prepare_training_data_rejects_text_shorter_than_a_window():
    with pytest.raises(ValueError, match="too short"):
        prepare_training_data("short", 10, train_windows=5, heldout_windows=5)


def run_main(tmp_path, capsys, seed):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(TEXT, encoding="utf-8")
    main(["--input", str(corpus), "--epochs", "2", "--seq-length", "5", "--hidden-size", "8",
          "--train-windows", "40", "--heldout-windows", "10", "--sample-length", "15",
          "--seed", str(seed)])
    return capsys.readouterr().out


def test_main_trains_evaluates_and_samples(tmp_path, capsys):
    out = run_main(tmp_path, capsys, seed=0)
    assert "Epoch 2/2: train" in out and "held-out" in out
    for temperature in (0.5, 1.0, 1.5):
        assert f"--- temperature {temperature} ---" in out
    assert out.count("ROMEO:") >= 3  # every sample starts from the seed text


def test_main_is_reproducible_for_a_fixed_seed(tmp_path, capsys):
    first = run_main(tmp_path, capsys, seed=0)
    second = run_main(tmp_path, capsys, seed=0)
    other = run_main(tmp_path, capsys, seed=1)
    assert first == second
    assert first != other
