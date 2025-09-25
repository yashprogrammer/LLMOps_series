
from .cutom_logger import CustomLogger as _CustomLogger  # backward compat
try:
    from .cutom_logger import CustomLogger
except Exception:
    CustomLogger = _CustomLogger

# Expose a global structlog-style logger used across the codebase
GLOBAL_LOGGER = CustomLogger().get_logger(__name__)
