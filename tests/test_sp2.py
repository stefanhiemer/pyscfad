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

"""Tests for pyscfad.scf.sp2.

Deliberately tiny (H2/STO-3G) -- fast, memory-light regression checks.
"""
import jax
import jax.numpy as jnp

from pyscfad.gto import MoleLite
from pyscfad.scf import hf_lite
from pyscfad.scf import sp2


def _make_mf(coords):
    mol = MoleLite(("H", "H"), coords, basis="sto3g", verbose=0)
    mf = hf_lite.SCFLite(mol)
    mf.use_sp2 = True
    mf.init_guess = "hcore"
    mf.diis = "anderson"
    return mol, mf


def test_sp2_scf_h2_sto3g_energy():
    mol, mf = _make_mf(jnp.array([[0., 0., 0.], [0., 0., 0.74]]))
    conv, e_tot, mo_energy, mo_coeff, mo_occ = sp2.scf(mf)

    mf_ref = MoleLite(("H", "H"), [[0., 0., 0.], [0., 0., 0.74]],
                      basis="sto3g", verbose=0).to_pyscf().RHF()
    mf_ref.kernel()
    assert bool(conv)
    assert abs(float(e_tot) - mf_ref.e_tot) < 1e-8


def test_sp2_scf_h2_sto3g_gradient():
    """sp2.scf's gradient (via custom_root) matches finite differences.

    Regression test for a bug where scf()'s tail reused _scf_implicit's
    AUX outputs (vhf, fock) as if they were differentiable. JAX's
    custom_root gives aux an explicit *zero* tangent by design
    (jax._src.lax.control_flow.solves._root_jvp: "solution_dot +=
    _map(ad_util.zeros_like_jaxval, aux)") -- the energy built from them
    was numerically correct but silently missing the implicit/Pulay-type
    gradient response (autodiff gave dE/dz=-0.9408 for this exact system;
    the correct value, matching finite differences and
    pyscfad.scf.hf_lite.kernel's own non-SP2 path on the same system, is
    -1.1405). Fixed by recomputing vhf/fock fresh from the properly-
    differentiated primal `dm`, mirroring hf_lite.kernel's own
    (already-correct) tail. A trivial standalone
    jax.lax.custom_root + pyscfad.scipy.sparse.linalg.gmres_const_atol
    fixed-point example confirmed the machinery itself was never at
    fault -- this was a usage bug in how scf() consumed its own
    _scf_implicit call.
    """
    def energy(z):
        coords = jnp.array([[0., 0., 0.], [0., 0., z]])
        _, mf = _make_mf(coords)
        return sp2.scf(mf)[1]

    g = jax.grad(energy)(0.74)
    eps = 1e-5
    fd = (energy(0.74 + eps) - energy(0.74 - eps)) / (2 * eps)
    assert abs(float(g) - float(fd)) < 1e-6
