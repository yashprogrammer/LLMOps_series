# logger/__init__.py
from .cutom_logger import CustomLogger
# Create a single shared logger instance
GLOBAL_LOGGER = CustomLogger().get_logger("doc_chat")