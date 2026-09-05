"""userDevice RTUS — real-time 2x upscaling.

Importing this package registers the RTUS architectures with traiNNer (when
traiNNer is installed), so `network_g: {type: rtmosr_ea_film}` in a config
resolves. See `_registry` for why that is not automatic.
"""

from userdevice_rtus._registry import (
    ARCH_REGISTRY,
    USING_REAL_REGISTRY,
    require_real_registry,
)
from userdevice_rtus.rtmosr_ea_vendored import RTMoSREA
from userdevice_rtus.rtmosr_vendored import RTMoSR

__all__ = [
    "ARCH_REGISTRY",
    "USING_REAL_REGISTRY",
    "require_real_registry",
    "RTMoSR",
    "RTMoSREA",
    "TIERS",
    "build_tier",
    "rtmosr_ea_film",
    "rtmosr_ea_film_sd",
]

# --- Tier geometry: THE single source of truth -----------------------------
#
# These shapes are not free parameters. They are what the released checkpoints
# were trained at, and a mismatch produces a state_dict that will not load.
# Before this table existed the same numbers were repeated in five scripts
# with no canonical definition; anything that builds an RTUS model should read
# them from here.
#
#   rtmosr_ea_film     d48 tier -- the 1080p model
#   rtmosr_ea_film_sd  d64 tier -- the <=720p model ("_sd" in config filenames)

_COMMON = {
    "scale": 2,
    "ffn_expansion": 2.0,
    "unshuffle_mod": True,
    "dccm": True,
    "se": True,
}

TIERS = {
    "rtmosr_ea_film": {**_COMMON, "dim": 48, "n_blocks": 3},
    "rtmosr_ea_film_sd": {**_COMMON, "dim": 64, "n_blocks": 6},
    "rtmosr_l": {**_COMMON, "dim": 32, "n_blocks": 2},
}


def build_tier(arch: str, **overrides):
    """Build a model by tier name, using the canonical geometry.

    This is what every eval and export path should call instead of repeating
    dim/n_blocks literals.
    """
    if arch not in TIERS:
        raise ValueError(
            f"unknown arch {arch!r} — expected one of {sorted(TIERS)}"
        )
    kwargs = {**TIERS[arch], **overrides}
    cls = RTMoSR if arch == "rtmosr_l" else RTMoSREA
    return cls(**kwargs)


@ARCH_REGISTRY.register()
def rtmosr_ea_film(**kwargs) -> RTMoSREA:
    """d48 tier — the 1080p model, and the only tier that fits 1080p->4K."""
    return RTMoSREA(**{**TIERS["rtmosr_ea_film"], **kwargs})


@ARCH_REGISTRY.register()
def rtmosr_ea_film_sd(**kwargs) -> RTMoSREA:
    """d64 tier — the <=720p model; spends the headroom low-res inputs leave."""
    return RTMoSREA(**{**TIERS["rtmosr_ea_film_sd"], **kwargs})
