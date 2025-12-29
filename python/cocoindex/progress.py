"""Progress tracking system for CocoIndex with file-based IPC support.

This module provides both same-process progress callbacks and cross-process
progress reporting via file-based IPC (JSONL format).

Key Components:
- Global progress callbacks for same-process operations
- ProgressReporter class for cross-process file-based IPC
- Thread-safe progress tracking
"""

import threading
import json
import datetime
from typing import Callable, Dict
from pathlib import Path
import os

# Global progress callback registry (same-process only)
_progress_callbacks: Dict[str, Callable[[str, int, int], None]] = {}
_progress_lock = threading.Lock()

# Debug logging - use relative path from current working directory
# This assumes the script is run from the project root (standard practice)
_logs_dir = Path(os.getcwd()) / "logs"
_logs_dir.mkdir(parents=True, exist_ok=True)
_debug_log_path = _logs_dir / "progress_debug.log"


def _debug_log(msg: str):
    """Write debug log message to progress_debug.log."""
    try:
        _debug_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(_debug_log_path, "a", encoding="utf-8") as f:
            timestamp = datetime.datetime.now().isoformat()
            f.write(f"{timestamp} | {msg}\n")
            f.flush()
    except Exception:
        pass  # Don't crash on logging errors


def set_progress_callback(operation_name: str, callback: Callable[[str, int, int], None]) -> None:
    """Set a progress callback for a specific operation.

    Args:
        operation_name: Name of the operation (e.g., "embedding", "chunking")
        callback: Function that takes (message: str, current: int, total: int)

    Note: This only works within the same process. For cross-process progress,
    use ProgressReporter class instead.
    """
    with _progress_lock:
        _progress_callbacks[operation_name] = callback
        _debug_log(f"✅ set_progress_callback: {operation_name}")


def clear_progress_callback(operation_name: str) -> None:
    """Clear a progress callback for a specific operation.

    Args:
        operation_name: Name of the operation to clear
    """
    with _progress_lock:
        if operation_name in _progress_callbacks:
            del _progress_callbacks[operation_name]
            _debug_log(f"❌ clear_progress_callback: {operation_name}")


def report_progress(operation_name: str, message: str, current: int, total: int) -> None:
    """Report progress for an operation (same-process only).

    Args:
        operation_name: Name of the operation
        message: Progress message
        current: Current item count
        total: Total item count

    Note: This only works within the same process. Worker processes spawned by
    multiprocessing won't have access to callbacks registered in the main process.
    """
    with _progress_lock:
        callback = _progress_callbacks.get(operation_name)

    if callback:
        try:
            callback(message, current, total)
            _debug_log(f"📊 report_progress: {operation_name} {current}/{total} - {message}")
        except Exception as e:
            _debug_log(f"❌ report_progress error: {e}")
            pass


class ProgressReporter:
    """Helper class for reporting progress in batched operations.

    Uses file-based IPC for cross-process progress reporting. This works across
    process boundaries, unlike the global callback system.

    Usage:
        reporter = ProgressReporter("embedding", total_items=1000)
        for batch in batches:
            process_batch(batch)
            reporter.update(len(batch), "Processing embeddings")
    """

    def __init__(self, operation_name: str, total_items: int, flow_name: str = "default"):
        """Initialize progress reporter.

        Args:
            operation_name: Name of the operation (e.g., "embedding")
            total_items: Total number of items to process
            flow_name: Name of the flow this operation belongs to (e.g., "code", "unity")
        """
        self.operation_name = operation_name
        self.total_items = total_items
        self.current_item = 0
        self.flow_name = flow_name
        # Use the same logs directory initialized at module level
        self.progress_file = _logs_dir / "progress_ipc.jsonl"

        _debug_log(f"✅ ProgressReporter created: {operation_name} flow={flow_name} (total={total_items})")

    def start(self, message: str = "Starting"):
        """Report operation started.

        Args:
            message: Start message
        """
        try:
            progress_data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "operation_name": self.operation_name,
                "flow_name": self.flow_name,
                "message": message,
                "total": self.total_items,
                "type": "operation_started"
            }
            with open(self.progress_file, "a") as f:
                f.write(json.dumps(progress_data) + "\n")
                f.flush()
            _debug_log(f"✅ ProgressReporter.start: {self.operation_name}")
        except Exception as e:
            _debug_log(f"❌ ProgressReporter.start error: {e}")
            pass  # Don't crash on IPC errors

    def update(self, items_processed: int, message: str = "Processing"):
        """Update progress by number of items processed.

        Args:
            items_processed: Number of items processed in this update
            message: Progress message
        """
        self.current_item += items_processed

        # Write progress to file for IPC (works across processes)
        try:
            progress_data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "operation_name": self.operation_name,
                "flow_name": self.flow_name,
                "message": message,
                "current": self.current_item,
                "total": self.total_items,
                "type": "operation_progress"
            }
            with open(self.progress_file, "a") as f:
                f.write(json.dumps(progress_data) + "\n")
                f.flush()
            _debug_log(f"✅ ProgressReporter.update: {self.operation_name} {self.current_item}/{self.total_items}")
        except Exception as e:
            _debug_log(f"❌ ProgressReporter.update error: {e}")
            pass  # Don't crash on IPC errors

    def complete(self, message: str = "Completed"):
        """Report operation completed.

        Args:
            message: Completion message
        """
        try:
            progress_data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "operation_name": self.operation_name,
                "flow_name": self.flow_name,
                "message": message,
                "total_processed": self.current_item,
                "type": "operation_completed"
            }
            with open(self.progress_file, "a") as f:
                f.write(json.dumps(progress_data) + "\n")
                f.flush()
            _debug_log(f"✅ ProgressReporter.complete: {self.operation_name}")
        except Exception as e:
            _debug_log(f"❌ ProgressReporter.complete error: {e}")
            pass  # Don't crash on IPC errors
