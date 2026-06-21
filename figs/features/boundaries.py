"""Deviant-supercell / surface-boundary features (raw scalars).

Surface boundaries — outflow, warm/cold fronts, drylines — locally enhance
low-level vorticity and storm-relative inflow and are where deviant (often
left-of-mean or right-of-mean) supercells and tornadoes concentrate. We flag them
with the horizontal GRADIENT MAGNITUDE of the 2 m temperature and dewpoint
(thermal / moisture boundaries) and the 10 m wind CONVERGENCE and total
DEFORMATION (the kinematic signature of a boundary). All are direction-invariant
raw scalars on the FIGS grid, computed from surface fields we already fetch —
cache-safe and added without further spatial treatment.
"""

from __future__ import annotations

import numpy as np

from .kinematics import _grid_spacing

# scale per-metre horizontal derivatives to per-100 km so the feature magnitudes
# sit in a friendly O(0.1–10) range (trees are scale-free, this is just for sanity).
_PER_100KM = 1.0e5


_SCALAR_GRAD_FIELDS = ("t2m", "td2m")  # scalars: signed x/y gradient components + magnitude


def boundary_feature_names() -> list[str]:
    """Names of the boundary point features (for the Tier-1 spatial set). All
    candidates are listed; ``compute_features`` only smooths the ones produced for
    a given state (those whose source surface fields were present)."""
    names: list[str] = []
    for f in _SCALAR_GRAD_FIELDS:                        # thermal / moisture boundaries
        names += [f"bnd_{f}_gradx", f"bnd_{f}_grady", f"bnd_{f}_gradmag"]
    # 10 m WIND VECTOR gradient: the four signed velocity-gradient components, plus
    # the rotation-invariant kinematic combos (convergence + deformation).
    names += ["bnd_u10_gradx", "bnd_u10_grady", "bnd_v10_gradx", "bnd_v10_grady",
              "bnd_conv10", "bnd_vort10", "bnd_def_stretch", "bnd_def_shear", "bnd_def10"]
    return names


def _grad(field: np.ndarray):
    """Signed horizontal gradient components (per 100 km): (d/dx east, d/dy north)."""
    dx, dy = _grid_spacing()
    gx = np.gradient(field, dx, axis=1) * _PER_100KM     # eastward (column) derivative
    gy = np.gradient(field, dy, axis=0) * _PER_100KM     # northward (row) derivative
    return gx, gy


def boundary_features(sfc: dict) -> dict[str, np.ndarray]:
    """Surface-boundary features keyed ``bnd_*``:

      * ``bnd_<f>_gradx`` / ``_grady`` / ``_gradmag``  signed east/north gradient +
        magnitude, for the scalar fields f in {t2m (thermal), td2m (moisture/dryline)}
      * ``bnd_u10_gradx`` / ``_grady`` / ``bnd_v10_gradx`` / ``_grady``  the four signed
        components of the 10 m WIND-VECTOR gradient ∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y (wind
        shifts / confluence keep their direction, unlike a wind-SPEED gradient)
      * ``bnd_conv10``       10 m convergence (max(-div, 0); boundary forcing)
      * ``bnd_vort10``       10 m relative vorticity (∂v/∂x − ∂u/∂y; misocyclones/shear)
      * ``bnd_def_stretch``  stretching deformation (∂u/∂x − ∂v/∂y)
      * ``bnd_def_shear``    shearing  deformation (∂v/∂x + ∂u/∂y)
      * ``bnd_def10``        total deformation magnitude (frontogenetic signature)
    """
    out: dict[str, np.ndarray] = {}

    def _add_grad(name: str, field: np.ndarray):
        gx, gy = _grad(np.asarray(field, dtype=float))
        out[f"bnd_{name}_gradx"] = gx
        out[f"bnd_{name}_grady"] = gy
        out[f"bnd_{name}_gradmag"] = np.sqrt(gx**2 + gy**2)

    if sfc.get("t2m") is not None:
        _add_grad("t2m", sfc["t2m"])
    if sfc.get("td2m") is not None:
        _add_grad("td2m", sfc["td2m"])

    u, v = sfc.get("u10"), sfc.get("v10")
    if u is not None and v is not None:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        dudx, dudy = _grad(u)                            # already scaled per 100 km
        dvdx, dvdy = _grad(v)
        out["bnd_u10_gradx"] = dudx
        out["bnd_u10_grady"] = dudy
        out["bnd_v10_gradx"] = dvdx
        out["bnd_v10_grady"] = dvdy
        out["bnd_conv10"] = np.maximum(-(dudx + dvdy), 0.0)  # convergence (positive part)
        out["bnd_vort10"] = dvdx - dudy                      # relative vorticity
        stretch = dudx - dvdy
        shear = dvdx + dudy
        out["bnd_def_stretch"] = stretch
        out["bnd_def_shear"] = shear
        out["bnd_def10"] = np.sqrt(stretch**2 + shear**2)
    return out
