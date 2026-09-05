"""Architecture registry shim.

traiNNer resolves `network_g.type` through its own ARCH_REGISTRY, so an
architecture is only trainable if it is registered *there*. The vendored arch
modules originally carried a local no-op stub:

    class _R:
        def register(self):
            return lambda f: f

which silently made every `@ARCH_REGISTRY.register()` do nothing. Anything
decorated with it was invisible to traiNNer, so no config in configs/ could
be trained -- the failure looked like a missing architecture rather than a
registration that never happened.

This module resolves the real registry when traiNNer is importable, and falls
back to a no-op only so the eval and export scripts -- which build models
directly and never consult the registry -- keep working in environments where
traiNNer is not installed. `USING_REAL_REGISTRY` says which one you got, so a
caller can fail loudly rather than train against a phantom.
"""

try:
    from traiNNer.utils.registry import ARCH_REGISTRY  # type: ignore

    USING_REAL_REGISTRY = True
except ImportError:  # pragma: no cover - depends on the environment
    class _NoOpRegistry:
        """Stand-in used only when traiNNer is absent."""

        def register(self, *args, **kwargs):
            def _identity(obj):
                return obj

            return _identity

    ARCH_REGISTRY = _NoOpRegistry()
    USING_REAL_REGISTRY = False


def require_real_registry() -> None:
    """Raise if architectures would be registered into the no-op stub.

    Call this from any training entry point. Registering into the stub is
    indistinguishable from success until traiNNer reports an unknown
    architecture much later, with a message that points nowhere useful.
    """
    if not USING_REAL_REGISTRY:
        raise RuntimeError(
            "traiNNer is not importable, so architectures cannot be "
            "registered and no config can be trained. Install the training "
            "extra (`uv sync --extra train`) or use the container image."
        )
