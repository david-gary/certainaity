"""Domain exceptions used throughout ForenScope."""


class ForenScopeError(Exception):
    """Base class for all ForenScope errors."""


class UnsupportedFormatError(ForenScopeError):
    """Raised when the submitted image format is not supported."""


class FileTooLargeError(ForenScopeError):
    """Raised when the submitted file exceeds the size limit."""


class ImageTooSmallError(ForenScopeError):
    """Raised when the image dimensions are below the minimum threshold."""


class CorruptImageError(ForenScopeError):
    """Raised when the image cannot be decoded."""


class WeightsNotFoundError(ForenScopeError):
    """Raised when model weight files are missing from the weights/ directory."""


class FeatureExtractionError(ForenScopeError):
    """Raised when a handcrafted feature extractor fails unexpectedly."""
