"""Unit tests for domain exception hierarchy."""

from __future__ import annotations

import pytest

from certainaity.exceptions import (
    CertainaityError,
    CorruptImageError,
    FeatureExtractionError,
    FileTooLargeError,
    ImageTooSmallError,
    UnsupportedFormatError,
    WeightsNotFoundError,
)

_SUBCLASSES = [
    UnsupportedFormatError,
    FileTooLargeError,
    ImageTooSmallError,
    CorruptImageError,
    WeightsNotFoundError,
    FeatureExtractionError,
]


class TestHierarchy:
    def test_base_is_exception(self) -> None:
        assert issubclass(CertainaityError, Exception)

    def test_all_subclasses_inherit_base(self) -> None:
        for cls in _SUBCLASSES:
            assert issubclass(cls, CertainaityError), f"{cls.__name__} missing base"

    def test_subclasses_are_mutually_distinct(self) -> None:
        for i, a in enumerate(_SUBCLASSES):
            for b in _SUBCLASSES[i + 1 :]:
                assert not issubclass(a, b), f"{a.__name__} is subclass of {b.__name__}"

    def test_base_catches_all_subclasses(self) -> None:
        for cls in _SUBCLASSES:
            with pytest.raises(CertainaityError):
                raise cls("test")

    def test_specific_handler_not_caught_by_sibling(self) -> None:
        with pytest.raises(CorruptImageError):
            try:
                raise CorruptImageError("bad")
            except FileTooLargeError:
                pass

    def test_message_preserved(self) -> None:
        msg = "file is 999 bytes over limit"
        exc = FileTooLargeError(msg)
        assert msg in str(exc)

    def test_can_chain_exceptions(self) -> None:
        original = ValueError("raw decode failed")
        wrapped = CorruptImageError("decoding failed")
        wrapped.__cause__ = original
        assert wrapped.__cause__ is original
