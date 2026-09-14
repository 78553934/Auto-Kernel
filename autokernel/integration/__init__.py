"""Small, dependency-light bridge between AutoKernel and SOLAR.

The bridge deliberately lives outside either upstream project.  It can be
copied into an AutoKernel checkout, while SOLAR remains an optional provider
of architecture-aware predictions.
"""

__all__ = ["solar_bridge", "theory_advisor"]
