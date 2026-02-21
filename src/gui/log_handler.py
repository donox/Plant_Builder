"""Qt logging handler that emits a signal for each log record."""
from __future__ import annotations

import logging

from PySide2 import QtCore


class _SignalEmitter(QtCore.QObject):
    log_record_emitted = QtCore.Signal(str, int)  # (formatted_message, levelno)


class QtLogHandler(logging.Handler):
    """Logging handler that forwards records to a Qt signal."""

    def __init__(self):
        super().__init__()
        self._emitter = _SignalEmitter()
        self.log_record_emitted = self._emitter.log_record_emitted
        self.setFormatter(logging.Formatter("%(levelname)-8s %(name)s: %(message)s"))

    def emit(self, record):
        try:
            self._emitter.log_record_emitted.emit(self.format(record), record.levelno)
        except Exception:
            self.handleError(record)
