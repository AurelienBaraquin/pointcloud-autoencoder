"""Vérifie que le code de src/ est importable depuis les tests."""

import src


def test_src_is_importable() -> None:
    assert src.__doc__ is not None
