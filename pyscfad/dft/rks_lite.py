# Copyright 2025-2026 The PySCFAD Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Lightweight restricted Kohn-Sham DFT, built on :mod:`~pyscfad.scf.hf_lite`.

:func:`pyscfad.dft.rks.get_veff`/`energy_elec` are already plain,
duck-typed functions of a mean-field object (not methods tied to the dense
:class:`~pyscfad.scf.hf.RHF`), and :mod:`pyscfad.dft.numint`/
:mod:`pyscfad.gto.eval_gto` already work directly on a
:class:`~pyscfad.gto.MoleLite` (confirmed directly: ``numint.NumInt().nr_rks``
runs and returns a sane exc/vxc on a tiny He/STO-3G example) -- so this class
only needs to supply the grid/numint plumbing :func:`get_veff` expects
(``grids``, ``_numint``, ``xc``, ``do_nlc``, ``initialize_grids``) on top of
:class:`~pyscfad.scf.hf_lite.SCF`, reusing its ``get_j``/``get_jk`` (already
correct for packed ERI storage) and its SP2-sparsification-capable
``scf()``/``kernel()`` entirely unchanged.

Scope of this first version: pure/hybrid GGA or LDA functionals (whatever
:mod:`pyscfad.dft.numint`/libxc support through ``nr_rks``), no NLC, no
density fitting -- matching the "lite" ecosystem's own small-system,
packed-integral focus. Grids are built once per instance from
``mol.to_pyscf()`` via plain (non-differentiable) ``pyscf.dft.gen_grid`` --
grid *point* positions do depend on the nuclear geometry, but that
dependence is not auto-differentiated here (same simplification the
existing ``pyscfad.dft.rks.RKS`` makes for its own grids).
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from pyscf import __config__
from pyscf.dft import gen_grid
from pyscf.lib import param

from pyscfad.scf import hf_lite
from pyscfad.dft import numint
from pyscfad.dft.rks import get_veff as _rks_get_veff
from pyscfad.dft.rks import energy_elec as _rks_energy_elec

if TYPE_CHECKING:
    from pyscfad.gto import MoleLite


class RKSLite(hf_lite.SCF):
    """Restricted Kohn-Sham DFT on a :class:`~pyscfad.gto.MoleLite`.

    Attributes
    ----------
    xc : str
        exchange-correlation functional (libxc name), e.g. ``"LDA,VWN"``.
    nlc : str
        non-local correlation functional; ``""`` (default) disables it.
    """

    xc: str = "LDA,VWN"
    nlc: str = ""
    small_rho_cutoff: float = 1e-7

    def __init__(self, mol: "MoleLite", xc: str = "LDA,VWN", **kwargs):
        super().__init__(mol, **kwargs)
        # hf_lite.SCF.__init__ doesn't call pyscf.scf.hf.SCF.__init__, so
        # a few attributes get_veff needs (beyond what hf_lite.SCF itself
        # uses) are never set -- fill in the ones rks.get_veff reads.
        self.max_memory = param.MAX_MEMORY
        self.direct_scf = True
        self.xc = xc
        self._numint = numint.NumInt()
        self.grids = gen_grid.Grids(mol.to_pyscf())
        self.grids.level = getattr(__config__, "dft_rks_RKS_grids_level", 3)
        self._grids_built = False

    def do_nlc(self) -> bool:
        return bool(self.nlc) or self._numint.libxc.is_nlc(self.xc)

    def initialize_grids(self, mol=None, dm=None):
        # built once per instance, not geometry-dependent across calls --
        # matches the small/fixed-geometry-per-mf use pattern sp2.py's
        # scf() and update_dm() already assume (a fresh mf per energy
        # evaluation, same as every other hf_lite.SCF user in this
        # codebase: see tests/test_scf.py::test_rhf_lite_packed_nuc_grad).
        if not self._grids_built:
            self.grids.build()
            self._grids_built = True

    get_veff = _rks_get_veff
    energy_elec = _rks_energy_elec
