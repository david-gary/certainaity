"""Domain exceptions used throughout Certainaity."""


class CertainaityError(Exception):
    """Base class for all Certainaity errors."""


class UnsupportedFormatError(CertainaityError):
    """Raised when the submitted image format is not supported."""


class FileTooLargeError(CertainaityError):
    """Raised when the submitted file exceeds the size limit."""


class ImageTooSmallError(CertainaityError):
    """Raised when the image dimensions are below the minimum threshold."""


class CorruptImageError(CertainaityError):
    """Raised when the image cannot be decoded."""


class WeightsNotFoundError(CertainaityError):
    """Raised when model weight files are missing from the weights/ directory."""


class FeatureExtractionError(CertainaityError):
    """Raised when a handcrafted feature extractor fails unexpectedly."""
