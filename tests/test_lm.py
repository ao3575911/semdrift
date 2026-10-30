from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("transformers")

from semdrift.cli.app import main

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A random 2-layer GPT-2 with a BPE tokenizer trained on the fixture: no network."""
    import torch
    from tokenizers import ByteLevelBPETokenizer
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    out = tmp_path_factory.mktemp("tiny")
    text = (FIXTURES / "mini.jsonl").read_text() * 20
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator([text], vocab_size=300, special_tokens=["<|endoftext|>"])
    tok = PreTrainedTokenizerFast(tokenizer_object=bpe, eos_token="<|endoftext|>")  # type: ignore[no-untyped-call]
    tok.save_pretrained(out)
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=len(tok), n_positions=16, n_embd=16, n_layer=2, n_head=2)
    GPT2LMHeadModel(cfg).save_pretrained(out)  # type: ignore[no-untyped-call]
    return out


def test_missing_model_is_model_error(workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ingest", str(FIXTURES / "mini.jsonl"), "--pipeline", "basic"]) == 0
    assert main(["lm", "score", "--model", "not-pulled"]) == 4
    assert "semdrift lm pull" in capsys.readouterr().err


def test_score_top_words(workspace: Path, tiny_model: Path) -> None:
    assert main(["ingest", str(FIXTURES / "mini.jsonl"), "--pipeline", "basic"]) == 0
    assert main(["slice", "define", "early", "year<1900"]) == 0
    m = ["--model", str(tiny_model), "--stride", "5"]
    assert main(["--json", "lm", "score", "--slices", "early", *m]) == 0
    assert main(["--json", "lm", "top", "-n", "3", *m]) == 0
    assert main(["--json", "lm", "words", "--min-count", "1", *m]) == 0
    assert any((workspace / ".semdrift" / "lm").glob("*.parquet"))


def test_word_alignment_conserves_bits(tiny_model: Path) -> None:
    from semdrift.lm.model import load
    from semdrift.lm.surprisal import token_bits, word_bits

    lm = load(str(tiny_model))
    text = "The cat sat on the mat, and the dog sat too."
    words = text.replace(",", " ,").replace(".", " .").split()
    spans, pos = [], 0
    for w in words:
        start = text.index(w, pos)
        spans.append((start, start + len(w)))
        pos = start + len(w)
    bits, ends = token_bits(lm, text, stride=4)  # window 16 forces several windows
    assert not np.isnan(bits).any()
    assert (bits > 0).all()
    wb = word_bits(spans, bits, ends)
    assert wb.sum() == pytest.approx(bits.sum())
    assert (wb > 0).all()


def test_window_matches_full_context_when_it_fits(tiny_model: Path) -> None:
    from semdrift.lm.model import load
    from semdrift.lm.surprisal import token_bits

    lm = load(str(tiny_model))
    a, _ = token_bits(lm, "the cat sat", stride=15)
    b, _ = token_bits(lm, "the cat sat", stride=3)
    np.testing.assert_allclose(a, b, rtol=1e-5)
