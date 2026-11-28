"""Isolate mitmproxy's module-global context for embedded session threads.

mitmproxy 12 assumes a single Master per process. Its ctx module is read by
connection dispatch and addons, not just startup. Only threads inside our
scope use isolated bindings; unrelated users retain the original module state.
"""
from contextlib import contextmanager
from threading import local, Lock
from types import ModuleType
from mitmproxy import ctx

_local = local()
_install_lock = Lock()
_FIELDS = frozenset(("master", "options", "log"))

class _SessionContext(ModuleType):
    def __getattribute__(self, name):
        values = getattr(_local, "values", None)
        if name in _FIELDS and values is not None:
            if name not in values:
                raise RuntimeError("mitmproxy context accessed before session initialization")
            return values[name]
        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        values = getattr(_local, "values", None)
        if name in _FIELDS and values is not None:
            values[name] = value
        else:
            super().__setattr__(name, value)

@contextmanager
def session_context():
    with _install_lock:
        if not isinstance(ctx, _SessionContext):
            ctx.__class__ = _SessionContext
    previous = getattr(_local, "values", None)
    _local.values = {}
    try:
        yield
    finally:
        _local.values = previous
