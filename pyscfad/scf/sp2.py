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
SP2
"""
from __future__ import annotations
from typing import Any
from functools import partial

import numpy
from jax.lax import cond, while_loop, custom_root

from pyscfad import numpy as np
from pyscfad import lib
from pyscfad.lib import logger
from pyscfad.scf.anderson import Anderson
#from pyscfad.tools.linear_solver import gen_gmres
from pyscfad.scipy.sparse.linalg import gmres_const_atol

from pyscfad.scf.hf import SCF
Array = Any

def gershgorin_real_bounds(a):
    a = np.asarray(a)
    centers = a.diagonal()
    radii = np.sum(np.abs(a), axis=1) - np.abs(centers)
    lower_bounds = centers.real - radii
    upper_bounds = centers.real + radii
    return np.min(lower_bounds), np.max(upper_bounds)

def update_sp2_dm(dm, dm2, traceP, tracePP, nelectron):
    def _lower(dm, dm2):
        return dm2

    def _raise(dm, dm2):
        return 2 * dm - dm2

    dm_new = cond(
        np.greater(traceP, nelectron),
        _lower,
        _raise,
        dm, dm2
    )
    return dm_new

#def dot2(x):
#    x0 = x.astype(np.float32)
#    x1 = (x - x0).astype(np.float32)
#    a = np.dot(x0, x0).astype(np.float64)
#    b = np.dot(x0, x1).astype(np.float64)
#    #c = np.dot(x1, x1).astype(np.float32)
#    return a + b + b.T

#def dot2(a):
#    a_diag = np.diag(a)
#    a_offdiag = (a - np.diag(a_diag)).astype(np.float32)
#    a_offdiag2 = np.dot(a_offdiag, a_offdiag)
#    a_diag2 = a_diag * a_diag
#    a_diag_single = a_diag.astype(np.float32)
#    tmp = a_diag_single[:,None] * a_offdiag2 + a_offdiag2 * a_diag_single[None,:]
#    return np.diag(a_diag2) + tmp + a_offdiag2

def dot2(a):
    return np.dot(a, a)

def _sp2(dm, nelectron, max_cycle=50, tol=1e-7):
    def cond_fun(value):
        cycle, _, _, traceP, tracePP = value
        return (cycle < max_cycle) & (np.abs(tracePP-traceP) > tol)

    def body_fun(value):
        cycle, dm, dm2, traceP, tracePP = value
        dm_new = update_sp2_dm(dm, dm2, traceP, tracePP, nelectron)
        dm2_new = dot2(dm_new)
        traceP_new = np.trace(dm_new)
        tracePP_new = np.trace(dm2_new)
        return cycle+1, dm_new, dm2_new, traceP_new, tracePP_new

    dm2 = dot2(dm)
    traceP = np.trace(dm)
    tracePP = np.trace(dm2)
    init_val = (0, dm, dm2, traceP, tracePP)
    dm = while_loop(cond_fun, body_fun, init_val)[1]
    return dm

def sp2(h, nelectron, max_cycle=50, tol=1e-7):
    n = h.shape[-1]
    theta = nelectron / n
    theta_bar = 1 - theta

    mu = np.trace(h) / n
    e_min, e_max = gershgorin_real_bounds(h)

    beta = theta / (e_max - mu)
    beta_bar = theta_bar / (mu - e_min)

    beta1 = theta
    beta2 = np.minimum(beta, beta_bar)

    dm0 = -beta2 * h
    dm0 += (beta1 + beta2 * mu) * np.eye(n)

    dm = _sp2(dm0, nelectron, max_cycle=max_cycle, tol=tol)
    return dm

def _ao2mo(a, x):
    return np.dot(x.T.conj(), np.dot(a, x))

def _mo2ao(a, x):
    return np.dot(x, np.dot(a, x.T.conj()))

def shell_ao_slices(mol) -> list:
    """AO index slice per shell (``mol.ao_loc_nr()``), one entry per shell.

    Each slice spans a whole shell's AOs, i.e. all ``2l+1`` m-components
    together -- the unit :func:`group_soft_threshold` must gate as one
    group, or spherical symmetry within the shell breaks.
    """
    ao_loc = mol.ao_loc_nr()
    return [slice(int(ao_loc[i]), int(ao_loc[i+1])) for i in range(mol.nbas)]

def group_soft_threshold(dm: Array, shell_slices: list, lam: float) -> Array:
    """Group-soft-threshold (group-LASSO proximal step) of a density matrix.

    Each shell gets one scalar gate, applied to its *entire* AO row and
    column jointly (``dm_out[i,j] = g[shell(i)] * g[shell(j)] * dm[i,j]``)
    so an inactive shell's whole row+column vanishes -- not just its
    diagonal block -- which is what actually removes it from the energy
    and from every occupied natural orbital. The gate is

        g_s = relu(1 - lambda / ||dm[shell_s, :]||_F)

    the standard group-LASSO shrinkage, zeroing a shell exactly once its
    row-block Frobenius norm drops below ``lambda``. A small eps inside
    the sqrt keeps the gradient finite even when a block is already
    exactly zero (``sqrt(x)`` alone has an infinite derivative at
    ``x=0``); verified by direct finite-difference check, including
    exactly at a threshold crossing.

    Parameters
    ----------
    dm : array
        AO-basis density matrix (or one spin channel of it), symmetric.
    shell_slices : list of slice
        one entry per shell, from :func:`shell_ao_slices`.
    lam : float
        shrinkage strength (same units as the density matrix's own
        Frobenius-norm scale, i.e. of order the occupation number).

    Returns
    -------
    array
        the shrunk density matrix, same shape as ``dm``.
    """
    eps = 1e-24
    gates = []
    for sl in shell_slices:
        block_sqnorm = np.sum(dm[sl, :] ** 2)
        block_norm = np.sqrt(block_sqnorm + eps)
        gate = np.maximum(0., 1. - lam / block_norm)
        gates.append(np.broadcast_to(gate, (sl.stop - sl.start,)))
    g = np.concatenate(gates)
    return (g[:,None] * g[None,:]) * dm

def mcweeny_purify_ao(dm: Array, s1e: Array, nelectron: float,
                      n_iter: int = 20) -> Array:
    """AO-basis McWeeny purification, re-idempotizing a trial density.

    :func:`group_soft_threshold` breaks ``D = D S D`` exactly (shrinking
    a shell's block is not, in general, an idempotency-preserving move),
    so the shrunk density needs projecting back onto the idempotency
    manifold before it can seed the next Fock build. Re-uses McWeeny's
    classic non-orthogonal-basis recursion ``D' = 3 D S D - 2 D S D S D``
    (pushes the eigenvalues of ``D S`` towards ``{0,1}``) rather than
    SP2's own polynomial purification (:func:`sp2`/`_sp2`), since that
    one already assumes an *orthogonalized* working basis (built from the
    AO->MO transform ``x``) -- shell blocks only have a clean meaning in
    the AO basis itself, so re-purification has to happen there too, in
    its own natural (non-orthogonal) form.

    A plain rescale to the target trace first keeps the iteration in the
    basin where it actually converges to the intended occupation, rather
    than to the nearest *wrong* integer.

    McWeeny's recursion has its fixed points at ``D S`` eigenvalues of
    ``{0, 1}`` (a singly-occupied-orbital convention). This project's own
    density matrices are the doubly-occupied RHF convention instead
    (``D S`` eigenvalues ``{0, 2}``, see :func:`update_dm`'s ``* 2`` after
    ``_mo2ao``) -- feeding that convention directly into the plain
    formula is not just "a worse fixed point", it puts every occupied
    eigenvalue on the *unstable* branch of ``f(x) = 3x**2 - 2x**3``
    (``f(2) = -4``: one iteration already overshoots and flips sign, and
    it diverges exponentially from there -- caught by direct numerical
    check before this landed). Purifying the half-scaled, singly-occupied
    matrix instead and doubling the result afterward fixes this exactly.

    Unlike :func:`sp2`/`_sp2` (which is globally convergent by
    construction, via its own Gershgorin-bound-scaled starting point),
    plain McWeeny iteration only converges from a trial density already
    reasonably close to idempotent -- a trial whose ``D S`` eigenvalues
    stray far outside ``[0, 2]`` can converge to the trivial ``D=0``
    fixed point instead, or diverge (verified directly: an arbitrary
    random symmetric trial matrix does this; a lightly-perturbed, already
    near-idempotent one does not). This is fine for its one intended use
    here -- the input is always a shrinkage-perturbed *already-SP2-
    purified* density, not an arbitrary matrix -- but it is not a
    drop-in substitute for :func:`sp2` on a cold-start trial.

    Parameters
    ----------
    dm : array
        trial AO-basis density matrix (e.g. straight out of
        :func:`group_soft_threshold`), doubly-occupied convention.
    s1e : array
        AO overlap matrix.
    nelectron : float
        target electron count for this density/spin-channel
        (``Tr(D S)`` after purification).
    n_iter : int
        number of McWeeny iterations; a fixed, static count (this runs
        inside a JAX-traced loop) rather than a tolerance-based stop.

    Returns
    -------
    array
        the re-idempotized density matrix, same shape as ``dm``.
    """
    half = dm / 2.
    tr = np.trace(np.dot(half, s1e))
    half = half * ((nelectron / 2.) / tr)
    for _ in range(n_iter):
        ds = np.dot(half, s1e)
        dsd = np.dot(ds, half)
        half = 3. * dsd - 2. * np.dot(ds, dsd)
    return half * 2.

def update_dm(
    mf: SCF,
    h1e: Array,
    s1e: Array,
    vhf: Array,
    dm: Array,
    cycle: int,
    diis: Any,
    fock: Array,
    x: Array,
    e_tot: float,
    lam: float | None = None,
    shell_slices: list | None = None,
    purify_iter: int = 20,
) -> tuple[float, float, Array, Array, Array, float, Any]:
    """Single SP2 SCF step, with optional group-LASSO shell sparsification.

    ``lam``/``shell_slices`` are ``None`` by default, reproducing the
    original (pre-sparsification) behaviour exactly. When ``lam`` is
    given, the density this step proposes is passed through
    :func:`group_soft_threshold` (dropping low-weight shells' AO blocks
    to exactly zero) and then :func:`mcweeny_purify_ao` (restoring
    idempotency broken by the shrinkage) before the veff/energy/Fock for
    *this* cycle are built -- so everything downstream (convergence
    check, next cycle's Fock) is consistent with the shrunk density, not
    the pre-shrinkage one.
    """
    log = logger.new_logger(mf)

    mol = mf.mol
    dm_last = dm
    last_hf_e = e_tot
    fock_last = fock

    h = _ao2mo(fock_last, x)
    nocc = mf.tot_electrons // 2 #FIXME
    dm_orth = sp2(h, nocc)
    dm = _mo2ao(dm_orth, x) * 2

    if lam is not None:
        dm = group_soft_threshold(dm, shell_slices, lam)
        dm = mcweeny_purify_ao(dm, s1e, 2*nocc, n_iter=purify_iter)

    vhf = mf.get_veff(mol, dm, dm_last, vhf, s1e=s1e)
    e_tot = mf.energy_tot(dm, h1e, vhf)

    de = e_tot-last_hf_e
    norm_ddm = np.linalg.norm(dm-dm_last)
    log.info("cycle= %d E= %.15g  delta_E= %4.3g  |ddm|= %4.3g",
             cycle+1, e_tot, de, norm_ddm)

    fock = mf.get_fock(h1e, s1e, vhf, dm, cycle+1, diis, fock_last=fock_last)
    del log
    return de, norm_ddm, dm, vhf, fock, e_tot, diis

def _scf(
    mf: SCF,
    dm: Array,
    h1e: Array,
    s1e: Array,
    vhf: Array,
    e_tot: float,
    conv_tol: float,
    conv_tol_dm: float,
    lam: float | None = None,
    shell_slices: list | None = None,
    purify_iter: int = 20,
):
    fock = mf.get_fock(h1e, s1e, vhf, dm)
    _, x = mf.eig(fock, s1e)

    def cond_fun(value):
        cycle, de, norm_ddm = value[:3]
        return (cycle < mf.max_cycle) & ((abs(de) > conv_tol) | (norm_ddm > conv_tol_dm))

    def body_fun(value):
        cycle, _, _, dm, vhf, fock, e_tot, diis = value
        de, norm_ddm, dm, vhf, fock, e_tot, diis = \
            update_dm(mf, h1e, s1e, vhf, dm, cycle, diis, fock, x, e_tot,
                     lam=lam, shell_slices=shell_slices, purify_iter=purify_iter)
        return cycle+1, de, norm_ddm, dm, vhf, fock, e_tot, diis

    if isinstance(mf.diis, str) and mf.diis.lower() == "anderson":
        diis = Anderson(
            lib.pack_tril(fock),
            space=mf.diis_space,
            damp=mf.diis_damp,
            start_cycle=mf.diis_start_cycle
        )
    else:
        diis = None
    init_val = (0, e_tot, 1e3, dm, vhf, fock, e_tot, diis)
    cycle, _, _, dm, vhf, fock, e_tot, _ = while_loop(cond_fun, body_fun, init_val)
    return dm, (vhf, fock, e_tot)

def _scf_implicit(
    mf: SCF,
    dm: Array,
    h1e: Array,
    s1e: Array,
    vhf: Array,
    e_tot: float,
    conv_tol: float,
    conv_tol_dm: float,
    lam: float | None = None,
    shell_slices: list | None = None,
    purify_iter: int = 20,
):
    oracle = lambda fn, dm0: _scf(mf, dm0, h1e, s1e, vhf, e_tot, conv_tol, conv_tol_dm,
                                  lam=lam, shell_slices=shell_slices, purify_iter=purify_iter)

    def root_fn(dm):
        # must define the SAME fixed-point map the oracle actually
        # iterates (including the shrink+repurify step when lam is
        # given) -- custom_root's implicit-function-theorem gradient is
        # only correct for the equation whose root dm actually is; a
        # mismatch here would silently compute the gradient of the
        # unsparsified map at a point that isn't its root.
        vhf = mf.get_veff(mf.mol, dm, s1e=s1e)
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm_new = mf.make_rdm1(mo_coeff, mo_occ)
        del mo_energy, mo_occ
        if lam is not None:
            dm_new = group_soft_threshold(dm_new, shell_slices, lam)
            dm_new = mcweeny_purify_ao(dm_new, s1e, mf.tot_electrons,
                                       n_iter=purify_iter)
        return dm_new - dm

    #solver = gen_gmres()
    solver = partial(gmres_const_atol,
                 tol=1e-6, atol=1e-6, maxiter=30,
                 solve_method="batched", restart=20)
    def tangent_solve(g, dm_bar):
        return solver(g, dm_bar)[0]

    dm_cnvg, (vhf_cnvg, fock_cnvg, e_tot_cnvg) = \
        custom_root(root_fn, dm, oracle, tangent_solve, has_aux=True)
    return dm_cnvg, vhf_cnvg, fock_cnvg, e_tot_cnvg

def scf(
    mf: SCF,
    conv_tol: float = 1e-10,
    conv_tol_dm: float = None,
    dm0: Array | None = None,
    lam: float | None = None,
    shell_slices: list | None = None,
    purify_iter: int = 20,
):
    """SP2-purification SCF, with optional group-LASSO shell sparsification.

    Parameters
    ----------
    lam : float, optional
        group-LASSO shrinkage strength (see :func:`group_soft_threshold`).
        ``None`` (default) reproduces the original, unsparsified SP2 SCF
        exactly. Should be ramped up over an outer optimization loop
        rather than treated as a free design variable -- see
        ``roadmap.md`` in the pyscfad-routines project for why.
    shell_slices : list of slice, optional
        one entry per shell (:func:`shell_ao_slices`); required when
        ``lam`` is given.
    purify_iter : int
        number of McWeeny re-purification iterations after each
        shrinkage step (:func:`mcweeny_purify_ao`).
    """
    log = logger.new_logger(mf)
    if conv_tol_dm is None:
        conv_tol_dm = numpy.sqrt(conv_tol)
        log.info("Set dm conv threshold to %g", conv_tol_dm)
    if lam is not None and shell_slices is None:
        shell_slices = shell_ao_slices(mf.mol)

    mol = mf.mol
    s1e = mf.get_ovlp(mol)

    if dm0 is None:
        dm = mf.get_init_guess(mol, mf.init_guess, s1e=s1e)
    else:
        dm = dm0

    h1e = mf.get_hcore(mol, s1e=s1e)
    vhf = mf.get_veff(mol, dm, s1e=s1e)
    e_tot = mf.energy_tot(dm, h1e, vhf)
    log.info("init E= %.15g", e_tot)

    scf_conv = False
    mo_energy = mo_coeff = mo_occ = None

    dm, _, _, e_tot = _scf_implicit(
        mf,
        dm,
        h1e,
        s1e,
        vhf,
        e_tot,
        conv_tol,
        conv_tol_dm,
        lam=lam,
        shell_slices=shell_slices,
        purify_iter=purify_iter,
    )
    # vhf/fock from _scf_implicit are custom_root's AUX outputs -- JAX
    # gives aux an explicit *zero* tangent by design (see
    # jax._src.lax.control_flow.solves._root_jvp: "solution_dot +=
    # _map(ad_util.zeros_like_jaxval, aux)"). Reusing them directly here
    # would silently drop the implicit/Pulay-type gradient response and
    # return a WRONG gradient while leaving the forward energy correct --
    # this is exactly the bug this file's gradient test used to catch
    # (confirmed by direct comparison against finite differences and
    # against pyscfad.scf.hf_lite.kernel's own tail, which already avoids
    # it by recomputing vhf/fock here the same way). Recompute them fresh
    # from the properly-differentiated primal `dm` instead. `e_tot` from
    # `_scf_implicit` is still fine to keep using below -- it only feeds
    # `abs(e_tot-last_hf_e)` for the boolean convergence flag, which
    # doesn't need a gradient.
    vhf = mf.get_veff(mol, dm, s1e=s1e)
    fock = mf.get_fock(h1e, s1e, vhf, dm)

    mo_energy, mo_coeff = mf.eig(fock, s1e)
    mo_occ = mf.get_occ(mo_energy, mo_coeff)
    dm, dm_last = mf.make_rdm1(mo_coeff, mo_occ), dm
    # the "extra cycle" below re-derives dm/e_tot from a fresh
    # diagonalization of the converged Fock purely to recover full-
    # precision gradients lost by implicit differentiation (same pattern
    # as pyscfad.scf.hf.kernel's own extra cycle) -- h1e alone still
    # couples to a sparsified shell's AOs (it doesn't depend on dm), so
    # without re-applying shrink+repurify here this step would silently
    # undo the sparsification on the very last step before e_tot/mo_occ
    # are returned.
    if lam is not None:
        dm = group_soft_threshold(dm, shell_slices, lam)
        dm = mcweeny_purify_ao(dm, s1e, mf.tot_electrons, n_iter=purify_iter)
    vhf = mf.get_veff(mol, dm, dm_last, vhf, s1e=s1e)
    e_tot, last_hf_e = mf.energy_tot(dm, h1e, vhf), e_tot

    norm_ddm = np.linalg.norm(dm - dm_last)
    conv_tol = conv_tol * 10
    conv_tol_dm = conv_tol_dm * 3
    scf_conv = (abs(e_tot-last_hf_e) < conv_tol) & (norm_ddm < conv_tol_dm)
    log.info("Extra cycle  E= %.15g  delta_E= %4.3g  |ddm|= %4.3g",
             e_tot, e_tot-last_hf_e, norm_ddm)

    del log
    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ
