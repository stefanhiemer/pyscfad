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

"""Tests for pyscfad.scf.sp2, including the group-LASSO shell-sparsification
extension (group_soft_threshold / mcweeny_purify_ao / sp2.scf(lam=...)).

Deliberately tiny systems throughout (H2/STO-3G, toy 6x6 matrices) -- these
are meant to be fast, memory-light correctness/regression checks, not
production-scale basis-set calculations.
"""
import numpy
import pytest
import jax
import jax.numpy as jnp

from pyscfad.gto import MoleLite
from pyscfad.scf import hf_lite
from pyscfad.scf import sp2


def _toy_dm(seed=0, n=6):
    rng = numpy.random.default_rng(seed)
    a = rng.standard_normal((n, n))
    return (a + a.T) / 2


def test_group_soft_threshold_shrinks_and_zeros():
    """Block norms shrink monotonically with lam and hit exactly zero once
    lam exceeds a block's own (lam=0) norm -- the basic group-LASSO
    shrinkage behaviour the whole sparsification scheme depends on.
    """
    shell_slices = [slice(0, 2), slice(2, 4), slice(4, 6)]
    dm = jnp.asarray(_toy_dm())
    norms0 = [float(jnp.linalg.norm(dm[sl, :])) for sl in shell_slices]

    out_small = sp2.group_soft_threshold(dm, shell_slices, 0.0)
    assert numpy.allclose(out_small, dm)

    lam_big = max(norms0) * 10
    out_big = sp2.group_soft_threshold(dm, shell_slices, lam_big)
    for sl in shell_slices:
        assert float(jnp.linalg.norm(out_big[sl, :])) < 1e-10
        assert float(jnp.linalg.norm(out_big[:, sl])) < 1e-10


def test_group_soft_threshold_differentiable():
    """Gradient matches finite differences in the smooth interior, and
    stays finite (no NaN) exactly at a block's own threshold crossing --
    the "ReLU-like kink" risk flagged in roadmap.md, checked directly.
    """
    shell_slices = [slice(0, 2), slice(2, 4), slice(4, 6)]
    dm0 = _toy_dm()
    norms0 = [float(jnp.linalg.norm(jnp.asarray(dm0)[sl, :])) for sl in shell_slices]
    x0 = jnp.asarray(dm0.ravel())

    def f(dm_flat, lam):
        dm = dm_flat.reshape(6, 6)
        dm = (dm + dm.T) / 2
        return jnp.sum(sp2.group_soft_threshold(dm, shell_slices, lam) ** 2)

    # interior region: strictly between 0 and the smallest block norm
    lam_interior = min(norms0) * 0.5
    val, grad = jax.value_and_grad(f)(x0, lam_interior)
    eps = 1e-6
    x0_np = numpy.asarray(x0)
    fd = numpy.zeros_like(x0_np)
    for i in range(x0_np.size):
        xp = x0_np.copy(); xp[i] += eps
        xm = x0_np.copy(); xm[i] -= eps
        fd[i] = (float(f(jnp.asarray(xp), lam_interior))
                - float(f(jnp.asarray(xm), lam_interior))) / (2*eps)
    assert numpy.abs(fd - numpy.asarray(grad)).max() < 1e-6

    # exactly at a threshold crossing
    val, grad = jax.value_and_grad(f)(x0, norms0[1])
    assert numpy.all(numpy.isfinite(numpy.asarray(grad)))


def _idempotent_dm(s1e, nocc, seed):
    """Build an exactly D S D = D, Tr(D S) = 2*nocc density for a given
    overlap -- i.e. a genuine doubly-occupied RHF-like density, not an
    arbitrary symmetric matrix. Representative of mcweeny_purify_ao's
    actual input: a lightly shrinkage-perturbed, already-SP2-purified
    density, not a cold-start trial (see its docstring on why a plain
    random matrix is not in McWeeny's basin of convergence).
    """
    n = s1e.shape[-1]
    w, u = numpy.linalg.eigh(numpy.asarray(s1e))
    s_mhalf = (u * w**-0.5) @ u.T
    rng = numpy.random.default_rng(seed)
    q, _ = numpy.linalg.qr(rng.standard_normal((n, n)))
    c_occ = s_mhalf @ q[:, :nocc]
    return jnp.asarray(2. * c_occ @ c_occ.T)


def test_mcweeny_purify_ao_idempotent():
    """Re-idempotizes a shrinkage-perturbed, already-near-idempotent trial
    back onto the doubly-occupied-convention manifold (D S eigenvalues
    {0,2}, not {0,1} -- see mcweeny_purify_ao's own docstring for why the
    naive {0,1}-convention formula diverges here) and preserves trace.
    """
    n = 6
    rng = numpy.random.default_rng(1)
    b = rng.standard_normal((n, n))
    s1e = jnp.asarray(b @ b.T + n * numpy.eye(n))

    dm_idem = _idempotent_dm(s1e, nocc=2, seed=2)
    # mimic one group_soft_threshold step: scale part of the matrix down,
    # breaking idempotency without destroying the overall scale
    dm_trial = dm_idem.at[:3, :].multiply(0.7).at[:, :3].multiply(0.7)

    dm_pure = sp2.mcweeny_purify_ao(dm_trial, s1e, 4.0, n_iter=25)
    ds = dm_pure @ s1e
    assert float(jnp.linalg.norm(ds @ ds - 2. * ds)) < 1e-8
    assert abs(float(jnp.trace(ds)) - 4.0) < 1e-8

    # an already-idempotent (doubly-occupied) matrix is a fixed point,
    # not a divergence -- this is exactly the case that caught the
    # {0,1}-vs-{0,2} convention bug during development
    dm_fixed = sp2.mcweeny_purify_ao(dm_idem, s1e, 4.0, n_iter=10)
    assert float(jnp.linalg.norm(dm_fixed - dm_idem)) < 1e-8


@pytest.mark.parametrize("lam", [None, 0.0, 1e-3, 1e-2])
def test_sp2_scf_h2_sto3g_energy(lam):
    """sp2.scf reproduces the reference RHF energy for H2/STO-3G, with the
    sparsification machinery inactive (lam=None, the pre-existing default
    path, unmodified) and with it active but not yet strong enough to
    shrink anything observable at this scale (lam=0 is an exact no-op by
    construction; the larger lam values are below where either of this
    2-shell system's two (both essential, non-redundant) shells would
    actually be dropped).
    """
    symbols = ("H", "H")
    coords = numpy.array([[0., 0., 0.], [0., 0., 0.74]])

    mol = MoleLite(symbols, coords, basis="sto3g", verbose=0)
    mf = hf_lite.SCFLite(mol)
    mf.use_sp2 = True
    mf.init_guess = "hcore"
    mf.diis = "anderson"

    kwargs = {}
    if lam is not None:
        kwargs["lam"] = lam
        kwargs["shell_slices"] = sp2.shell_ao_slices(mol)
    conv, e_tot, mo_energy, mo_coeff, mo_occ = sp2.scf(mf, **kwargs)

    mf_ref = MoleLite(symbols, coords, basis="sto3g", verbose=0).to_pyscf().RHF()
    mf_ref.kernel()
    assert bool(conv)
    assert abs(float(e_tot) - mf_ref.e_tot) < 1e-6


@pytest.mark.parametrize("lam", [None, 0.0, 1e-3])
def test_sp2_scf_h2_sto3g_gradient(lam):
    """sp2.scf's gradient (via custom_root) matches finite differences.

    This used to fail (see git history / roadmap.md for the full writeup):
    scf()'s tail was reusing _scf_implicit's AUX outputs (vhf, fock) as if
    they were differentiable. JAX's custom_root gives aux an explicit
    *zero* tangent by design (jax._src.lax.control_flow.solves._root_jvp:
    "solution_dot += _map(ad_util.zeros_like_jaxval, aux)") -- the energy
    built from them was numerically correct but silently missing the
    implicit/Pulay-type gradient response. Fixed by recomputing vhf/fock
    fresh from the properly-differentiated primal `dm`, mirroring
    pyscfad.scf.hf_lite.kernel's own (already-correct) tail. Confirmed via
    a trivial standalone custom_root+gmres_const_atol example that the
    machinery itself was never at fault -- this was purely a usage bug in
    how scf() consumed its own _scf_implicit call.

    Parametrized over lam (None/0.0/1e-3) to check this holds through the
    sparsification path too, not just the original unsparsified one.
    """
    symbols = ("H", "H")

    def energy(z):
        coords = jnp.array([[0., 0., 0.], [0., 0., z]])
        mol = MoleLite(symbols, coords, basis="sto3g", verbose=0)
        mf = hf_lite.SCFLite(mol)
        mf.use_sp2 = True
        mf.init_guess = "hcore"
        mf.diis = "anderson"
        kwargs = {}
        if lam is not None:
            kwargs["lam"] = lam
            kwargs["shell_slices"] = sp2.shell_ao_slices(mol)
        return sp2.scf(mf, **kwargs)[1]

    g = jax.grad(energy)(0.74)
    eps = 1e-5
    fd = (energy(0.74 + eps) - energy(0.74 - eps)) / (2 * eps)
    assert abs(float(g) - float(fd)) < 1e-6


def test_sp2_scf_sparsifies_a_redundant_shell():
    """End-to-end: as lam ramps up, two of three He shells get driven to
    an *exactly* zero AO block while the third compensates (grows), and
    the energy degrades smoothly and monotonically -- the group-LASSO
    "shells drop out one at a time, continuously in lam" path this whole
    mechanism is for.

    Must inspect `_scf_implicit`'s own primal `dm` directly, not
    `sp2.scf`'s returned `mo_coeff`/`mo_occ` (reconstructed via a fresh
    diagonalization of ``h1e + veff(dm)`` in scf()'s own "extra cycle" --
    h1e alone still couples to a "dropped" shell's AOs since it doesn't
    depend on dm, so that reconstruction does not, and is not meant to,
    preserve the shrunk blocks exactly; it exists only to recover full-
    precision gradients, see scf()'s own comment).
    """
    basis = {"He": [[0, [0.3, 1.0]], [0, [1.0, 1.0]], [0, [3.0, 1.0]]]}
    shell_slices = [slice(0, 1), slice(1, 2), slice(2, 3)]

    def converged_sparse_dm(lam):
        mol = MoleLite(("He",), [[0., 0., 0.]], basis=basis, verbose=0)
        mf = hf_lite.SCFLite(mol)
        mf.use_sp2 = True
        mf.init_guess = "hcore"
        mf.diis = "anderson"
        s1e = mf.get_ovlp(mol)
        dm0 = mf.get_init_guess(mol, mf.init_guess, s1e=s1e)
        h1e = mf.get_hcore(mol, s1e=s1e)
        vhf0 = mf.get_veff(mol, dm0, s1e=s1e)
        e0 = mf.energy_tot(dm0, h1e, vhf0)
        dm, _, _, _ = sp2._scf_implicit(
            mf, dm0, h1e, s1e, vhf0, e0, 1e-10, 3.16e-6,
            lam=lam, shell_slices=shell_slices, purify_iter=30)
        e = mf.energy_tot(dm, h1e, mf.get_veff(mol, dm, s1e=s1e))
        return dm, float(e)

    mf_ref = MoleLite(("He",), [[0., 0., 0.]], basis=basis, verbose=0).to_pyscf().RHF()
    mf_ref.kernel()

    dm0, e0 = converged_sparse_dm(0.0)
    assert abs(e0 - mf_ref.e_tot) < 1e-6
    norms0 = [float(numpy.linalg.norm(numpy.asarray(dm0)[sl, :])) for sl in shell_slices]
    assert all(n > 0.1 for n in norms0)  # all three shells contribute at lam=0

    dm1, e1 = converged_sparse_dm(0.5)
    norms1 = [float(numpy.linalg.norm(numpy.asarray(dm1)[sl, :])) for sl in shell_slices]
    # shells 1 and 2 (the tighter two) are exactly gone; shell 0 (most
    # diffuse) carries the whole doubly-occupied density alone
    assert norms1[1] < 1e-8
    assert norms1[2] < 1e-8
    assert abs(norms1[0] - 2.0) < 1e-6
    # energy visibly, but not catastrophically, worse than the 3-shell ref
    assert e1 > e0
    assert e1 - e0 < 1.0
