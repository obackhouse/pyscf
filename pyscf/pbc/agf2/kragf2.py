# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
#
# Author: Oliver J. Backhouse <olbackhouse@gmail.com>
#         George H. Booth <george.booth@kcl.ac.uk>
#

'''
k-point adapted Auxiliary second-order Green's function perturbation theory
'''

import time
import copy
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__
from pyscf import ao2mo
from pyscf.pbc import tools
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.agf2 import kragf2_ao2mo, _kagf2
from pyscf.agf2 import aux, ragf2, _agf2, mpi_helper, chkfile as chkutil
from pyscf.agf2.chempot import binsearch_chempot, minimize_chempot
from pyscf.pbc.mp.kmp2 import get_nocc, get_nmo, get_frozen_mask

#TODO: change some allreduce to allgather 
#TODO: check aft, this may be broken?
#TODO: change agf2 object to gf2 and molecular code
#TODO: fix molecular code DIIS to use the Aux.moment function
#TODO: should we track convergence via etot?
#TODO: re-tests direct stuff - had to change conj

#NOTE: agf2.nmo is .max()'d, must all be the same, whilst agf2.nocc is per-kpoint and can be different


#NOTE: printing IPs and EAs doesn't work with ragf2 kernel - fix later for better inheritance 
def kernel(agf2, eri=None, gf=None, se=None, verbose=None, dump_chk=True):

    log = logger.new_logger(agf2, verbose)
    cput1 = cput0 = (time.clock(), time.time())
    name = agf2.__class__.__name__

    if eri is None: eri = agf2.ao2mo()
    if gf is None: gf = agf2.gf
    if se is None: se = agf2.se
    if verbose is None: verbose = agf2.verbose

    if gf is None:
        gf = agf2.init_gf(eri)

    if se is None:
        se = agf2.build_se(eri, gf)

    if dump_chk:
        agf2.dump_chk(gf=gf, se=se)

    if isinstance(agf2.diis, lib.diis.DIIS):
        diis = agf2.diis
    elif agf2.diis:
        diis = lib.diis.DIIS(agf2)
        diis.space = agf2.diis_space
        diis.min_space = agf2.diis_min_space
    else:
        diis = None

    e_init = agf2.energy_mp2(agf2.mo_energy, se)
    #e_init = agf2.energy_2body(gf, se) / 2
    log.info('E(init) = %.16g  E_corr(init) = %.16g', e_init+eri.e_hf, e_init)

    e_1b = eri.e_hf
    e_2b = e_init

    e_prev = 0.0
    se_prev = None
    converged = False
    for niter in range(1, agf2.max_cycle+1):
        if agf2.damping != 0.0:
            se_prev = copy.deepcopy(se)

        # one-body terms
        gf, se, fock_conv = agf2.fock_loop(eri, gf, se)
        e_1b = agf2.energy_1body(eri, gf)

        # two-body terms
        se = agf2.build_se(eri, gf, se_prev=se_prev)
        se = agf2.run_diis(se, diis)
        e_2b = agf2.energy_2body(gf, se)

        if dump_chk:
            agf2.dump_chk(gf=gf, se=se)

        e_tot = e_1b + e_2b

        ip = [x[0] for x in agf2.get_ip(gf, nroots=1)[0]]
        ea = [x[0] for x in agf2.get_ea(gf, nroots=1)[0]]

        log.info('cycle = %3d  E(%s) = %.15g  E_corr(%s) = %.15g  dE = %.9g',
                 niter, name, e_tot, name, e_tot-eri.e_hf, e_tot-e_prev)
        log.info('E_1b = %.15g  E_2b = %.15g', e_1b, e_2b)
        for kx in range(agf2.nkpts):
            log.info('k-point %d  IP = %.15g  EA = %.15g', kx, ip[kx], ea[kx])
        cput1 = log.timer('%s iter'%name, *cput1)

        if abs(e_tot - e_prev) < agf2.conv_tol:
            converged = True
            break

        e_prev = e_tot

    if dump_chk:
        agf2.dump_chk(gf=gf, se=se)

    log.timer('%s'%name, *cput0)

    return converged, e_1b, e_2b, gf, se


#TODO: we probably need to deal with padded energies here? maybe only in the first iteration..?
def build_se_part(agf2, eri, gf_occ, gf_vir, os_factor=1.0, ss_factor=1.0):
    ''' Builds either the auxiliaries of the occupied self-energy at
        each k-point, or virtual if :attr:`gf_occ` and :attr:`gf_vir` 
        are swapped.

    Args:
        eri : _ChemistsERIS
            Electronic repulsion integrals
        gf_occ : list of GreensFunction
            Occupied Green's function at each k-point
        gf_vir : list of GreensFunction
            Virtual Green's function at each k-point

    Kwargs:
        os_factor : float
            Opposte-spin factor for spin-component-scaled (SCS)
            calculations. Default 1.0
        ss_factor : float
            Same-spin factor for spin-component-scaled (SCS)
            calculations. Default 1.0

    Returns:
        list of :class:`SelfEnergy`
    '''

    assert type(gf_occ[0]) is aux.GreensFunction
    assert type(gf_vir[0]) is aux.GreensFunction

    cput0 = (time.clock(), time.time())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    nkpts = agf2.nkpts
    nmo = agf2.nmo
    tol = agf2.weight_tol
    facs = dict(os_factor=os_factor, ss_factor=ss_factor)

    khelper = agf2.khelper
    kconserv = khelper.kconserv

    if isinstance(eri.eri, tuple):
        fmo2qmo = kragf2_ao2mo._make_qmo_eris_direct
    else:
        fmo2qmo = kragf2_ao2mo._make_qmo_eris_incore

    vv = np.zeros((nkpts, nmo, nmo), dtype=np.complex128)
    vev = np.zeros((nkpts, nmo, nmo), dtype=np.complex128)

    #FIXME: this probably isn't true...
    # constraining kj via conservation instead of ka means that we
    # don't have to do any padding tricks, as kx,ki,ka are all in
    # independent spaces
    #FIXME the sizes can still be different i think :(
    for kxia in mpi_helper.nrange(nkpts**3):
        kxi, ka = divmod(kxia, nkpts)
        kx, ki = divmod(kxi, nkpts)
        kj = kconserv[kx,ki,ka]

        ci, ei, ni = gf_occ[ki].coupling, gf_occ[ki].energy, gf_occ[ki].naux
        cj, ej, nj = gf_occ[kj].coupling, gf_occ[kj].energy, gf_occ[kj].naux
        ca, ea, na = gf_vir[ka].coupling, gf_vir[ka].energy, gf_vir[ka].naux

        if isinstance(eri.eri, (tuple, list)):
            qxi, qja = fmo2qmo(agf2, eri, (ci,cj,ca), (kx,ki,kj,ka))
            qxj, qia = fmo2qmo(agf2, eri, (cj,ci,ca), (kx,kj,ki,ka))
            vv_k, vev_k = _kagf2.build_mats_kragf2_direct(qxi, qja, qxj, qia, ei, ej, ea, **facs)
            del qxi, qja, qxj, qia
        elif isinstance(eri.eri, (df.AFTDF)):
            #TODO make more efficient if this is good - ao2mo_7d-like function for custom kpts
            cx_ao = agf2.mo_coeff[kx]
            ci_ao = np.dot(agf2.mo_coeff[ki], ci)
            cj_ao = np.dot(agf2.mo_coeff[kj], cj)
            ca_ao = np.dot(agf2.mo_coeff[ka], ca)
            pija = eri.eri.get_mo_eri((cx_ao, ci_ao, cj_ao, ca_ao), kpts[[kx,ki,kj,ka]])
            pjia = eri.eri.get_mo_eri((cx_ao, cj_ao, ci_ao, ca_ao), kpts[[kx,kj,ki,ka]])
            vv_k, vev_k = _kagf2.build_mats_kragf2_incore(pija, pjia, ei, ej, ea, **facs)
            del pija, pjia
        else:
            pija = fmo2qmo(agf2, eri, (ci,cj,ca), (kx,ki,kj,ka))
            pjia = fmo2qmo(agf2, eri, (cj,ci,ca), (kx,kj,ki,ka))
            vv_k, vev_k = _kagf2.build_mats_kragf2_incore(pija, pjia, ei, ej, ea, **facs)
            del pija, pjia

        vv[kx] += vv_k
        vev[kx] += vev_k

    mpi_helper.barrier()
    for kx in range(nkpts):
        mpi_helper.allreduce_safe_inplace(vv[kx])
        mpi_helper.allreduce_safe_inplace(vev[kx])

    se = []
    for kx in range(nkpts):
        #TODO remove checks
        if (not np.allclose(vv[kx], vv[kx].T.conj())) or (not np.allclose(vev[kx], vev[kx].T.conj())):
            if mpi_helper.rank == 0:
                print('NOT HERM CONJ vv: %.12g vev: %.12g' % (np.max(np.absolute(vv[kx]-vv[kx].T.conj())), np.max(np.absolute(vev[kx]-vev[kx].T.conj()))))
            vv[kx] = 0.5 * (vv[kx] + vv[kx].T.conj())
            vev[kx] = 0.5 * (vev[kx] + vev[kx].T.conj())
        e, c = _agf2.cholesky_build(vv[kx], vev[kx], eps=1e-14)
        se_kx = aux.SelfEnergy(e, c, chempot=gf_occ[kx].chempot)
        se_kx.remove_uncoupled(tol=tol)
        se.append(se_kx)

    log.timer('se part', *cput0)

    return se


#TODO: can we adapt these k-space get_jk functions to point-group symmetry easily?
#TODO: maybe optimised with pyscf lib
#NOTE: should we fuse loops?
def get_jk_direct(agf2, eri, rdm1, with_j=True, with_k=True, madelung=None):
    ''' Get the J/K matrices.

    Args:
        eri : tuple of ndarray
            Electronic repulsion integrals (NOT as _ChemistsERIs) at
            each k-point
        rdm1 : list of 2D array
            Reduced density matrix at each k-point

    Kwargs:
        with_j : bool
            Whether to compute J. Default value is True
        with_k : bool
            Whether to compute K. Default value is True
        madelung : list of 2D array
            Result of tools.pbc.madelung(cell, kpts)

    Returns:
        tuple of ndarrays corresponding to J and K at each k-point,
        if either are not requested then they are set to None.
    '''

    nkpts = agf2.nkpts
    nmo = agf2.nmo
    naux = eri[0].shape[2]
    dtype = np.result_type(eri[0].dtype, eri[1].dtype, *[x.dtype for x in rdm1])

    qij, qkl = eri
    qij = qij.reshape(nkpts, nkpts, naux, nmo**2)
    qkl = qkl.reshape(nkpts, nkpts, nkpts, naux, nmo**2)

    vj = vk = None

    if with_j:
        vj = np.zeros((nkpts, nmo*nmo), dtype=dtype)
        buf = np.zeros((naux), dtype=dtype)

        for kik in mpi_helper.nrange(nkpts**2):
            ki, kk = divmod(kik, nkpts)
            kj = ki
            kl = agf2.khelper.kconserv[ki,kj,kk]
            buf = np.dot(qkl[ki,kj,kk], rdm1[kl].ravel(), out=buf)
            vj[ki] += np.dot(qij[ki,kj].T, buf)

        vj = vj.reshape(nkpts, nmo, nmo)

        mpi_helper.barrier()
        mpi_helper.allreduce_safe_inplace(vj)

    if with_k:
        vk = np.zeros((nkpts, nmo, nmo), dtype=dtype)
        buf = np.zeros((naux*nmo, nmo), dtype=dtype)

        for kik in mpi_helper.nrange(nkpts**2):
            ki, kk = divmod(kik, nkpts)
            kj = ki
            kl = agf2.khelper.kconserv[ki,kj,kk]
            buf = lib.dot(qij[ki,kl].reshape(-1, nmo), rdm1[kl].conj(), c=buf)
            buf = buf.reshape(-1, nmo, nmo).swapaxes(1,2).reshape(-1, nmo)
            vk[ki] = lib.dot(buf.T, qkl[ki,kl,kk].reshape(-1, nmo), c=vk[ki], beta=1).T.conj()

        mpi_helper.barrier()
        mpi_helper.allreduce_safe_inplace(vk)

        #NOTE: should keep_exxdiv even be considered here?
        if agf2.keep_exxdiv and agf2._scf.exxdiv == 'ewald':
            vk += get_ewald(agf2, rdm1, madelung=madelung)

    return vj, vk


#TODO: check
#TODO: optimise
def get_jk_incore(agf2, eri, rdm1, with_j=True, with_k=True, madelung=None):
    ''' Get the J/K matrices.

    Args:
        eri : ndarray
            Electronic repulsion integrals (NOT as _ChemistsERIs) at
            each k-point
        rdm1 : list of 2D array
            Reduced density matrix at each k-point

    Kwargs:
        with_j : bool
            Whether to compute J. Default value is True
        with_k : bool
            Whether to compute K. Default value is True
        madelung : list of 2D array
            Result of tools.pbc.madelung(cell, kpts)

    Returns:
        tuple of ndarrays corresponding to J and K at each k-point,
        if either are not requested then they are set to None.
    '''

    nkpts = len(rdm1)
    nmo = agf2.nmo
    dtype = np.result_type(eri.dtype, *[x.dtype for x in rdm1])
    
    eri = eri.reshape(nkpts, nkpts, nkpts, nmo, nmo, nmo, nmo)

    vj = vk = None

    if with_j:
        vj = np.zeros((nkpts, nmo, nmo), dtype=dtype)

        for kik in mpi_helper.nrange(nkpts**2):
            ki, kk = divmod(kik, nkpts)
            kj = ki
            kl = agf2.khelper.kconserv[ki,kj,kk]
            vj[ki] += lib.einsum('ijkl,lk->ij', eri[ki,kj,kk], rdm1[kl].conj())

        mpi_helper.barrier()
        mpi_helper.allreduce_safe_inplace(vj)

    if with_k:
        vk = np.zeros((nkpts, nmo, nmo), dtype=dtype)

        for kik in mpi_helper.nrange(nkpts**2):
            ki, kk = divmod(kik, nkpts)
            kj = ki
            kl = agf2.khelper.kconserv[ki,kj,kk]
            vk[ki] += lib.einsum('ilkj,lk->ij', eri[ki,kl,kk], rdm1[kl].conj())

        mpi_helper.barrier()
        mpi_helper.allreduce_safe_inplace(vk)

        #NOTE: should keep_exxdiv even be considered here?
        if agf2.keep_exxdiv and agf2._scf.exxdiv == 'ewald':
            vk += get_ewald(agf2, rdm1, madelung=madelung)

    return vj, vk


def get_jk(agf2, eri, rdm1, with_j=True, with_k=True, madelung=None):
    if isinstance(eri, (tuple, list)):
        vj, vk = get_jk_direct(agf2, eri, rdm1, with_j=with_j, with_k=with_k, madelung=madelung)
    else:
        vj, vk = get_jk_incore(agf2, eri, rdm1, with_j=with_j, with_k=with_k, madelung=madelung)
    for kx in range(len(rdm1)):
        #TODO remove checks
        if (not np.allclose(vj[kx], vj[kx].T.conj())) or (not np.allclose(vk[kx], vk[kx].T.conj())):
            if mpi_helper.rank == 0:
                print('NOT HERM CONJ j: %.12g k: %.12g' % (np.max(np.absolute(vj[kx]-vj[kx].T.conj())), np.max(np.absolute(vk[kx]-vk[kx].T.conj()))))
        vj[kx] = 0.5 * (vj[kx] + vj[kx].T.conj())
        vk[kx] = 0.5 * (vk[kx] + vk[kx].T.conj())

    return vj, vk
    

def get_ewald(agf2, rdm1, madelung=None):
    ''' Get the Ewald exchange contribution. The density matrix is
        assumed to be in MO basis, i.e. the overlap is identity.

    Args:
        rdm1 : list of 2D array
            Reduced density matrix at each k-point

    Kwargs:
        madelung : list of 2D array
            Result of tools.pbc.madelung(cell, kpts)

    Returns:
        ndarray of Ewald exchange contribution at each k-point
    '''

    if madelung is None:
        madelung = tools.pbc.madelung(agf2.cell, agf2.kpts)

    ewald = [madelung * x for x in rdm1]

    return ewald


def get_fock(agf2, eri, gf=None, rdm1=None, madelung=None):
    ''' Computes the physical space Fock matrix in MO basis at each
        k-point. If :attr:`rdm1` is not supplied, it is built from
        :attr:`gf`, which defaults the the mean-field Green's function

    Args:
        eri : ndarray
            Electronic repulsion integrals (NOT as _ChemistsERIs) at
            each k-point

    Kwargs:
        gf : list of Greensfunction
            Auxiliaries of the Green's function at each k-point
        rdm1 : list of 2D array
            Reduced density matrix at each k-point
        madelung : list of 2D array
            Result of tools.pbc.madelung(cell, kpts)

    Returns:
        ndarray of physical space Fock matrix at each k-point
    '''

    if rdm1 is None:
        rdm1 = agf2.make_rdm1(gf)

    vj, vk = agf2.get_jk(eri.eri, rdm1, madelung=madelung)
    fock = np.array(eri.h1e) + vj - 0.5 * vk

    return fock


#TODO: remove?
def adjust_occ(agf2, gf):
    ''' Modify the occupied energies of the Green's function according
        to the Madelung constant.
    '''

    if not agf2.keep_exxdiv:
        madelung = tools.madelung(agf2.cell, agf2.kpts)
        for kx in range(agf2.nkpts):
            occ = gf[kx].energy < gf[kx].chempot
            gf[kx].energy[occ] -= madelung

    return gf


#NOTE: padding should not affect the nelec, because the padded elements will be zero
#NOTE: how will the padding effect the projection of the QMOs into physical space?
def fock_loop(agf2, eri, gf, se, nelec_per_kpt=None):
    ''' Self-consistent loop for the density matrix via the HF self-
        consistent field in k-space.

    Args:
        eri : _ChemistsERIs
            Electronic repulsion integrals
        gf : list of GreensFunction
            Auxiliaries of the Green's function at each k-point
        se : list of SelfEnergy
            Auxiliaries of the self-energy at each k-point

    Kwargs:
        nelec_per_kpt : list of int
            Number of electrons at each k-point. If None, use
            :class:`agf2.get_nocc`.

    Returns:
        :class:`SelfEnergy`, :class:`GreensFunction` and a boolean
        indicating whether convergence was successful.
    '''

    assert type(gf[0]) is aux.GreensFunction
    assert type(se[0]) is aux.SelfEnergy

    cput0 = cput1 = (time.clock(), time.time())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    diis = lib.diis.DIIS(agf2)
    diis.space = agf2.diis_space
    diis.min_space = agf2.diis_min_space
    madelung = tools.madelung(agf2.cell, agf2.kpts)
    fock = agf2.get_fock(eri, gf, madelung=madelung)

    if nelec_per_kpt is None:
        nelec_per_kpt = np.array(get_nocc(agf2, per_kpoint=True)) * 2
    nmo = agf2.nmo
    nkpts = len(gf)
    converged = False
    opts = dict(tol=agf2.conv_tol_nelec, maxiter=agf2.max_cycle_inner)

    se = list(se)
    gf = list(gf)

    for niter1 in range(1, agf2.max_cycle_outer+1):
        for kx in range(nkpts):
            nelec = nelec_per_kpt[kx]
            se[kx], opt = minimize_chempot(se[kx], fock[kx], nelec, x0=se[kx].chempot, **opts)

        for niter2 in range(1, agf2.max_cycle_inner+1):
            nerr = 0

            for kx in range(nkpts):
                nelec = nelec_per_kpt[kx]
                w, v = se[kx].eig(fock[kx])
                se[kx].chempot, nerr_kpt = binsearch_chempot((w, v), nmo, nelec)
                nerr = nerr_kpt.real if abs(nerr_kpt.real) > nerr else nerr

                gf[kx] = aux.GreensFunction(w, v[:nmo], chempot=se[kx].chempot)
                gf[kx].remove_uncoupled(tol=agf2.weight_tol)

            fock = agf2.get_fock(eri, gf, madelung=madelung)
            rdm1 = agf2.make_rdm1(gf)
            fock = diis.update(fock, xerr=None)

            if niter2 > 1:
                drdm1 = np.array(rdm1) - np.array(rdm1_prev)
                derr_real = np.max(np.absolute(drdm1.real))
                derr_imag = np.max(np.absolute(drdm1.imag))
                derr = max(derr_real, derr_imag)
                if derr < agf2.conv_tol_rdm1:
                    break

            rdm1_prev = np.copy(rdm1)

        log.debug1('fock loop %d  cycles = %d  dN = %.3g  |ddm| = %.3g',
                   niter1, niter2, nerr, derr)
        cput1 = log.timer_debug1('fock loop %d'%niter1, *cput1)

        if derr < agf2.conv_tol_rdm1 and abs(nerr) < agf2.conv_tol_nelec:
            converged = True
            break

    log.info('fock converged = %s  dN = %.3g  |ddm| = %.3g', converged, nerr, derr)
    for kx in range(nkpts):
        log.info('      k-point %d  chempot = %.9g', kx, se[kx].chempot)
    log.timer('fock loop', *cput0)

    return gf, se, converged


def energy_1body(agf2, eri, gf):
    ''' Calculates the one-body energy according to the RHF form.

    Args:
        eri : _ChemistsERIS
            Electronic repulsion integrals
        gf : list of GreensFunction
            Auxiliaries of the Green's function at each k-point

    Returns:
        One-body energy
    '''

    e1b = 0.0

    rdm1 = agf2.make_rdm1(gf)
    fock = agf2.get_fock(eri, gf)

    for kx in range(agf2.nkpts):
        e1b += 0.5 * np.sum(rdm1[kx] * (eri.h1e[kx] + fock[kx])).real

    e1b /= len(gf)
    e1b += agf2.energy_nuc()

    return e1b


def energy_2body(agf2, gf, se):
    ''' Calculates the two-body energy using analytically integrated
        Galitskii-Migdal formula. The formula is symmetric and only
        one side needs to be calculated.

    Args:
        gf : list of GreensFunction
            Auxiliaries of the Green's function at each k-point
        se : list of SelfEnergy
            Auxiliaries of the self-energy at each k-point

    Returns:
        Two-body energy
    '''

    e2b = 0.0

    for g, s in zip(gf, se):
        e2b += ragf2.energy_2body(agf2, g, s)

    e2b /= len(gf)

    return e2b


def energy_mp2(agf2, mo_energy, se, return_mean=True):
    ''' Calculates the two-body energy using analytically integrated
        Galitskii-Migdal formula for an MP2 self-energy. Per the
        definition of one- and two-body partitioning in the Dyson
        equation, this result is half of :func:`energy_2body`.

    Args:
        gf : list of GreensFunction
            Auxiliaries of the Green's function at each k-point
        se : list of SelfEnergy
            Auxiliaries of the self-energy at each k-point

    Returns:
        MP2 energy
    '''

    assert isinstance(mo_energy, (list, tuple)) or mo_energy.ndim == 2

    emp2 = 0.0

    for mo, s in zip(mo_energy, se):
        emp2 += ragf2.energy_mp2(agf2, mo, s)

    emp2 /= len(se)

    return emp2


#TODO: memoize energy_nuc() in _ChemistsERIs
class KRAGF2(ragf2.RAGF2):
    #TODO: doc
    ''' Restricted AGF2 with canonical HF reference in k-space
    '''

    async_io = getattr(__config__, 'pbc_agf2_async_io', True)
    incore_complete = getattr(__config__, 'pbc_agf2_incore_complete', True)

    def __init__(self, mf, frozen=None, mo_energy=None, mo_coeff=None, mo_occ=None):

        if mo_energy is None: mo_energy = mf.mo_energy
        if mo_coeff is None:  mo_coeff  = mf.mo_coeff
        if mo_occ is None:    mo_occ    = mf.mo_occ

        if not (frozen is None or frozen == 0):
            raise ValueError('Frozen orbitals not supported for KAGF2')

        self.cell = mf.cell
        self._scf = mf
        self.kpts = mf.kpts
        self.khelper = kpts_helper.KptsHelper(self.cell, self.kpts)
        self.verbose = self.cell.verbose
        self.stdout = self.cell.stdout
        self.max_memory = mf.max_memory
        self.incore_complete = self.incore_complete or self.cell.incore_anyway
        self.with_df = mf.with_df

        self.conv_tol = getattr(__config__, 'pbc_agf2_conv_tol', 1e-7)
        self.conv_tol_rdm1 = getattr(__config__, 'pbc_agf2_conv_tol_rdm1', 1e-8)
        self.conv_tol_nelec = getattr(__config__, 'pbc_agf2_conv_tol_nelec', 1e-6)
        self.max_cycle = getattr(__config__, 'pbc_agf2_max_cycle', 50)
        self.max_cycle_outer = getattr(__config__, 'pbc_agf2_max_cycle_outer', 20)
        self.max_cycle_inner = getattr(__config__, 'pbc_agf2_max_cycle_inner', 50)
        self.weight_tol = getattr(__config__, 'pbc_agf2_weight_tol', 1e-11)
        self.diis = getattr(__config__, 'agf2_diis', True)
        self.diis_space = getattr(__config__, 'pbc_agf2_diis_space', 8)
        self.diis_min_space = getattr(__config__, 'pbc_agf2_diis_min_space', 1)
        self.os_factor = getattr(__config__, 'pbc_agf2_os_factor', 1.0)
        self.ss_factor = getattr(__config__, 'pbc_agf2_ss_factor', 1.0)
        self.damping = getattr(__config__, 'pbc_agf2_damping', 0.0)
        self.direct = getattr(__config__, 'pbc_agf2_direct', True)
        self.keep_exxdiv = getattr(__config__, 'pbc_agf2_keep_exxdiv', True)

        self.mo_energy = mo_energy
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.se = None
        self.gf = None
        self.e_1b = mf.e_tot
        self.e_2b = 0.0
        self.e_init = 0.0
        self.frozen = frozen
        self._nmo = None
        self._nocc = None
        self.converged = False
        self.chkfile = mf.chkfile
        self._keys = set(self.__dict__.keys())

    energy_1body = energy_1body
    energy_2body = energy_2body
    fock_loop = fock_loop
    build_se_part = build_se_part
    get_jk = get_jk

    def ao2mo(self, mo_coeff=None):
        ''' Get the electronic repulsion integrals in MO basis.
        '''
        #TODO FIXME 
        from pyscf.pbc import df
        if self.direct and type(self.with_df) == df.FFTDF:
            raise ValueError('direct + FFT broken')

        if self.direct:
            eri = kragf2_ao2mo._make_mo_eris_direct(self, mo_coeff)
            l, r = eri.eri
            mpi_helper.barrier()
            mpi_helper.allreduce_safe_inplace(l)
            mpi_helper.allreduce_safe_inplace(r)
            mpi_helper.barrier()
            eri.eri = (l / mpi_helper.size, r / mpi_helper.size)
        else:
            eri = kragf2_ao2mo._make_mo_eris_incore(self, mo_coeff)
            mpi_helper.barrier()
            mpi_helper.allreduce_safe_inplace(eri.eri)
            mpi_helper.barrier()
            eri.eri /= mpi_helper.size

        return eri

    def make_rdm1(self, gf=None):
        ''' Computes the one-body reduced density matrix in MO basis
            at each k-point.

        Kwargs:
            gf : list of GreensFunction
                Auxiliaries of the Green's function at each k-point

        Returns:
            ndarray of density matrix at each k-point
        '''

        if gf is None: gf = self.gf
        if gf is None: gf = self.init_gf()

        if isinstance(gf, (list, tuple)):
            rdm1 = [x.make_rdm1() for x in gf]
        else:
            rdm1 = gf.make_rdm1()

        return rdm1

    def get_fock(self, eri=None, gf=None, rdm1=None, madelung=None):
        ''' Computes the physical space Fock matrix in MO basis at
            each k-point.
        '''

        if eri is None: eri = self.ao2mo()
        if gf is None: gf = self.gf

        return get_fock(self, eri, gf=gf, rdm1=rdm1, madelung=madelung)

    def energy_mp2(self, mo_energy=None, se=None):
        if mo_energy is None: mo_energy = self.mo_energy
        if se is None: se = self.build_se(gf=self.gf)

        #TODO: should this be the adjusted/exxdiv energies?

        self.e_init = energy_mp2(self, self.mo_energy, se)

        return self.e_init

    #TODO: frozen
    def init_gf(self, eri=None):
        ''' Builds the Hartree-Fock Green's function.

        Kwargs:
            eri : _ChemistsERIs
                Electronic repulsion integrals

        Returns:
            :class:`GreensFunction`, :class:`SelfEnergy`
        '''

        if eri is None: eri = self.ao2mo()

        nkpts = self.nkpts
        nmo = self.nmo
        nocc = self.nocc
        energy = self.mo_energy
        coupling = np.eye(nmo)

        gf = []

        for kx in range(nkpts):
            #FIXME: I think that we need this because in k-space we can have fully occ or fully vir?
            if nocc[kx] == 0:
                chempot = energy[kx][0] - 1e-6
            elif nocc[kx] == nmo:
                chempot = energy[kx][-1] + 1e-6
            else:
                chempot = binsearch_chempot(np.diag(energy[kx]), nmo, nocc[kx]*2)[0]

            gf.append(aux.GreensFunction(energy[kx], coupling, chempot=chempot))

        return gf

    def build_se(self, eri=None, gf=None, os_factor=None, ss_factor=None, se_prev=None):
        ''' Builds the auxiliaries of the self-energy at each k-point.

        Args:
            eri : _ChemistsERIs
                Electronic repulsion integrals
            gf : list of GreensFunction
                Auxiliaries of the Green's function at each k-point

        Kwargs:
            os_factor : float
                Opposite-spin factor for spin-component-scaled (SCS)
                calculations. Default 1.0
            ss_factor : float
                Same-spin factor for spin-component-scaled (SCS)
                calculations. Default 1.0
            se_prev : list of SelfEnergy
                Previous self-energy at each k-point for damping.
                Default value is None.

        Returns:
            :class:`SelfEnergy` at each k-point
        '''

        if eri is None: eri = self.ao2mo()
        if gf is None: gf = self.gf
        if gf is None: gf = self.init_gf(eri)

        if os_factor is None: os_factor = self.os_factor
        if ss_factor is None: ss_factor = self.ss_factor

        facs = dict(os_factor=os_factor, ss_factor=ss_factor)
        gf_occ = [x.get_occupied() for x in gf]
        gf_vir = [x.get_virtual() for x in gf]

        for kx in range(self.nkpts):
            if gf_occ[kx].naux == 0 or gf_vir[kx].naux == 0:
                logger.warn(self, 'Attempting to build a self-energy with '
                                  'no (i,j,a) or (a,b,i) configurations at '
                                  'k-point %d', kx)

        #FIXME doesn't work
        #if not self.keep_exxdiv:
        #    madelung = tools.madelung(self.cell, self.kpts)
        #    for kx in range(self.nkpts):
        #        gf_occ[kx].energy -= madelung
        #        chempot = 0.5 * (gf_occ[kx].energy.max() + gf_vir[kx].energy.min())
        #        gf_occ[kx].chempot = gf_vir[kx].chempot = chempot

        se_occ = self.build_se_part(eri, gf_occ, gf_vir, **facs)
        se_vir = self.build_se_part(eri, gf_vir, gf_occ, **facs)
        se = [aux.combine(o, v) for o,v in zip(se_occ, se_vir)]

        if se_prev is not None and self.damping != 0.0:
            for kx in range(self.nkpts):
                se[kx].coupling *= np.sqrt(1.0-self.damping)
                se_prev[kx].coupling *= np.sqrt(self.damping)
                se[kx] = aux.combine(se[kx], se_prev[kx])

                #NOTE: I haven't properly ported over compressions to complex numbers so I will do this manually for now:
                se_occ, se_vir = se[kx].get_occupied(), se[kx].get_virtual()

                vv = np.dot(se_occ.coupling, se_occ.coupling.T.conj())
                vev = np.dot(se_occ.coupling * se_occ.energy[None], se_occ.coupling.T.conj())
                e, c = _agf2.cholesky_build(vv, vev, eps=1e-14)
                se_occ = aux.SelfEnergy(e, c, chempot=se_occ.chempot)
                
                vv = np.dot(se_vir.coupling, se_vir.coupling.T.conj())
                vev = np.dot(se_vir.coupling * se_vir.energy[None], se_vir.coupling.T.conj())
                e, c = _agf2.cholesky_build(vv, vev, eps=1e-14)
                se_vir = aux.SelfEnergy(e, c, chempot=se_vir.chempot)

                se[kx] = aux.combine(se_occ, se_vir)

        return se

    def run_diis(self, se, diis=None):
        ''' Runs the direct inversion of the iterative subspace for the
            self-energy.

        Args:
            se : list of SelfEnergy
                Auxiliaries of the self-energy at each k-point
            diis : lib.diis.DIIS
                DIIS object

        Returns:
            :class:`SelfEnergy` at each k-point
        '''

        if diis is None:
            return se

        se_occ = [x.get_occupied() for x in se]
        se_vir = [x.get_virtual() for x in se]

        vv_occ = [x.moment(0) for x in se_occ]
        vv_vir = [x.moment(0) for x in se_vir]

        vev_occ = [x.moment(1) for x in se_occ]
        vev_vir = [x.moment(1) for x in se_vir]

        dat = np.array([vv_occ, vv_vir, vev_occ, vev_vir])
        dat = diis.update(dat)
        vv_occ, vv_vir, vev_occ, vev_vir = dat

        se_out = []
        for kx in range(self.nkpts):
            chempot = se[kx].chempot
            se_occ = aux.SelfEnergy(*_agf2.cholesky_build(vv_occ[kx], vev_occ[kx], eps=1e-14), chempot=chempot)
            se_vir = aux.SelfEnergy(*_agf2.cholesky_build(vv_vir[kx], vev_vir[kx], eps=1e-14), chempot=chempot)
            se_out.append(aux.combine(se_occ, se_vir))

        return se_out


    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('conv_tol = %g', self.conv_tol)
        log.info('conv_tol_rdm1 = %g', self.conv_tol_rdm1)
        log.info('conv_tol_nelec = %g', self.conv_tol_nelec)
        log.info('max_cycle = %g', self.max_cycle)
        log.info('max_cycle_outer = %g', self.max_cycle_outer)
        log.info('max_cycle_inner = %g', self.max_cycle_inner)
        log.info('weight_tol = %g', self.weight_tol)
        log.info('diis_space = %d', self.diis_space)
        log.info('diis_min_space = %d', self.diis_min_space)
        log.info('os_factor = %g', self.os_factor)
        log.info('ss_factor = %g', self.ss_factor)
        log.info('damping = %g', self.damping)
        log.info('direct = %s', self.direct)
        log.info('keep_exxdiv = %s', self.keep_exxdiv)
        log.info('nmo = %s', self.nmo)
        log.info('nocc = %s', self.nocc)
        if self.frozen is not None:
            log.info('frozen orbitals = %s', self.frozen)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        return self

    def _finalize(self):
        ''' Hook for dumping results and clearing up the object.
        '''

        if self.converged:
            logger.info(self, '%s converged', self.__class__.__name__)
        else:
            logger.note(self, '%s not converged', self.__class__.__name__)

        ip = [x[0] for x in self.get_ip(self.gf, nroots=1)[0]]
        ea = [x[0] for x in self.get_ea(self.gf, nroots=1)[0]]

        logger.note(self, 'E(%s) = %.16g  E_corr = %.16g',
                    self.__class__.__name__, self.e_tot, self.e_corr)

        for kx in range(self.nkpts):
            logger.note(self, 'k-point %d  IP = %.16g  EA = %.16g  QP gap = %.16g', 
                        kx, ip[kx], ea[kx], ip[kx]+ea[kx])

        return self

    def kernel(self, eri=None, gf=None, se=None, dump_chk=True):
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()

        if eri is None: eri = self.ao2mo()
        if gf is None: gf = self.gf
        if se is None: se = self.se

        if gf is None:
            gf = self.init_gf(eri)

        if se is None:
            se = self.build_se(eri, gf)

        self.converged, self.e_1b, self.e_2b, self.gf, self.se = \
                kernel(self, eri=eri, gf=gf, se=se, verbose=self.verbose, dump_chk=dump_chk)

        self._finalize()

        return self.converged, self.e_1b, self.e_2b, self.gf, self.se

    def dump_chk(self, chkfile=None, key='kagf2', kpts=None, gf=None, se=None, nmom=None, mo_energy=None, mo_coeff=None, mo_occ=None):
        chkutil.dump_kagf2(self, chkfile, key,
                           kpts, gf, se, None,
                           mo_energy, mo_coeff, mo_occ)
        return self

    def update_from_chk_(self, chkfile=None, key='agf2'):
        if chkfile is None:
            chkfile = self.chkfile

        mol, agf2_dict = chkutil.load_kagf2(chkfile, key)
        self.__dict__.update(agf2_dict)

        return self

    update = update_from_chk = update_from_chk_


    def get_ip(self, gf, nroots=5):
        gf_occ = [x.get_occupied() for x in gf]
        e_ip = [list(-x.energy[-nroots:])[::-1] for x in gf_occ]
        v_ip = [list(x.coupling[:,-nroots:].T)[::-1] for x in gf_occ]
        return e_ip, v_ip

    #TODO: unpack vectors
    def ipagf2(self, nroots=5):
        ''' Find the (N-1) electron charged excitations at each k-point,
            corresponding to the largest :attr:`nroots` poles of the
            occupied Green's function.

        Kwargs:
            nroots : int
                Number of roots (poles) requested. Default 1.

        Returns:
            IP and transition moment at each k-point (1D array, 2D
            array) if :attr:`nroots` = 1, or array of IPs and moments
            (2D array, 3D array) if :attr:`nroots` > 1. The leftmost
            axis corresponds to the k-point.
        '''

        e_ip, v_ip = self.get_ip(self.gf, nroots=nroots)

        for kx in range(self.nkpts):
            logger.note(self, 'k-point %d  (%s)', kx, self.kpts[kx])
            for n, en, vn in zip(range(nroots), e_ip[kx], v_ip[kx]):
                qpwt = np.linalg.norm(vn)**2
                logger.note(self, 'IP energy level %d E = %.16g  QP weight = %0.6g', n, en, qpwt)

        e_ip = np.asarray(e_ip)
        v_ip = np.asarray(v_ip)

        if not self.keep_exxdiv:
            madelung = tools.madelung(self.cell, self.kpts)
            e_ip += madelung

        if nroots == 1:
            return e_ip.reshape(self.nkpts,), v_ip.reshape(self.nkpts, self.nmo)
        else:
            return e_ip, v_ip

    def get_ea(self, gf, nroots=5):
        gf_vir = [x.get_virtual() for x in gf]
        e_ea = [list(x.energy[:nroots]) for x in gf_vir]
        v_ea = [list(x.coupling[:,:nroots].T) for x in gf_vir]
        return e_ea, v_ea

    def eaagf2(self, nroots=5):
        ''' Find the (N+1) electron charge excitations at each k-point,
            corresponding to the smallest :attr:`nroots` poles of the
            virtual Green's function.

        Kwargs:
            See eaagf2()
        '''

        e_ea, v_ea = self.get_ea(self.gf, nroots=nroots)

        for kx in range(self.nkpts):
            logger.note(self, 'k-point %d  (%s)', kx, self.kpts[kx])
            for n, en, vn in zip(range(nroots), e_ea[kx], v_ea[kx]):
                qpwt = np.linalg.norm(vn)**2
                logger.note(self, 'EA energy level %d E = %.16g  QP weight = %0.6g', n, en, qpwt)

        e_ea = np.asarray(e_ea)
        v_ea = np.asarray(v_ea)

        if nroots == 1:
            return e_ea.reshape(self.nkpts,), v_ea.reshape(self.nkpts, self.nmo)
        else:
            return e_ea, v_ea

    get_nmo = get_nmo
    get_nocc = get_nocc
    get_frozen_mask = get_frozen_mask

    #NOTE: _nocc and _nmo memoize the per_kpoint=True or per_kpoint=False
    # value depending which is called first and then just always return it
    #@property
    #def nmo(self):
    #    if self._nmo is None:
    #        self._nmo = get_nmo(self)
    #    return self._nmo
    #@nmo.setter
    #def nmo(self, val):
    #    self._nmo = val

    #@property
    #def nocc(self):
    #    if self._nocc is None:
    #        self._nocc = get_nocc(self)
    #    return self._nocc
    #@nocc.setter
    #def nocc(self, val):
    #    self._nocc = val

    @property
    def nmo(self):
        _nmo = self.get_nmo(per_kpoint=True)
        if not all([x==_nmo[0] for x in _nmo]):
            raise NotImplementedError('Different nmo at each k-point') #TODO move
        return _nmo[0]

    @property
    def nocc(self):
        return self.get_nocc(per_kpoint=True)

    @property
    def nkpts(self):
        return len(self.kpts)

    @property
    def e_tot(self):
        return self.e_1b + self.e_2b

    @property
    def e_corr(self):
        return self.e_tot - self._scf.e_tot

    @property
    def qmo_energy(self):
        return [x.energy for x in self.gf]

    @property
    def qmo_coeff(self):
        ''' Gives the couplings in AO basis '''
        return [np.dot(mo, x.coupling) for mo,x in zip(self.mo_coeff, self.gf)]

    @property
    def qmo_occ(self):
        coeff = [x.get_occupied().coupling for x in self.gf]
        occ = [2.0 * np.linalg.norm(c, axis=0) ** 2 for c in coeff]
        vir = [np.zeros_like(x.get_virtual().energy) for x in self.gf]
        qmo_occ = [np.concatenate([o, v]) for o,v in zip(occ, vir)]
        return qmo_occ


if __name__ == '__main__':
    import warnings
    import h5py
    warnings.simplefilter('ignore', h5py.h5py_warnings.H5pyDeprecationWarning)
    from pyscf.pbc import gto, scf, cc, mp, df

    def test_eri(rhf):
        gf2 = KRAGF2(rhf)
        eri_df = gf2.ao2mo()
        gf2.direct = False
        eri = gf2.ao2mo()

        if isinstance(gf2.with_df, df.MDF):
            bra, ket = kragf2_ao2mo._make_ao_eris_direct_mdf(gf2, eri)
        elif isinstance(gf2.with_df, df.GDF):
            bra, ket = kragf2_ao2mo._make_ao_eris_direct_gdf(gf2, eri)
        elif isinstance(gf2.with_df, df.AFTDF):
            bra, ket = kragf2_ao2mo._make_ao_eris_direct_aftdf(gf2, eri)
        elif isinstance(gf2.with_df, df.FFTDF):
            bra, ket = kragf2_ao2mo._make_ao_eris_direct_fftdf(gf2, eri)
        ao0 = np.einsum('ablp,abclq->abcpq', bra, ket).reshape((gf2.nkpts,)*3 + (gf2.nmo,)*4)
        ao1 = rhf.with_df.ao2mo_7d(np.asarray([[np.eye(gf2.nmo),]*gf2.nkpts]*4), kpts=rhf.kpts) / len(rhf.kpts)

        eri0 = np.einsum('ablp,abclq->abcpq', eri_df.eri[0], eri_df.eri[1]).reshape((gf2.nkpts,)*3 + (gf2.nmo,)*4)
        eri1 = eri.eri
        eri2 = rhf.with_df.ao2mo_7d(np.asarray(rhf.mo_coeff)+0j, kpts=rhf.kpts) / len(rhf.kpts)

        ci = np.random.random((gf2.nmo, max(gf2.nocc)*2)) + 1.0j * np.random.random((gf2.nmo, max(gf2.nocc)*2)) 
        cj = np.random.random((gf2.nmo, max(gf2.nocc)*2)) + 1.0j * np.random.random((gf2.nmo, max(gf2.nocc)*2))  
        ca = np.random.random((gf2.nmo, (gf2.nmo-max(gf2.nocc))*2)) + 1.0j * np.random.random((gf2.nmo, (gf2.nmo-max(gf2.nocc))*2))
        qmo0 = np.einsum('pqrs,qi,rj,sa->pija', eri2[0,1,0], ci, cj.conj(), ca)
        qmo1_bra, qmo1_ket = kragf2_ao2mo._make_qmo_eris_direct(gf2, eri_df, (ci,cj,ca), [0,1,0,1])
        qmo1 = np.dot(qmo1_bra.T, qmo1_ket).reshape(qmo0.shape)
        qmo2 = kragf2_ao2mo._make_qmo_eris_incore(gf2, eri, (ci,cj,ca), [0,1,0,1])

        print('MO', np.allclose(eri0, eri2), np.allclose(eri1, eri2), np.linalg.norm(eri0-eri2), np.linalg.norm(eri1-eri2))
        print('AO', np.allclose(ao0, ao1), np.linalg.norm(ao0-ao1))
        print('QMO', np.allclose(qmo0, qmo2), np.allclose(qmo0, qmo1), np.linalg.norm(qmo0-qmo1), np.linalg.norm(qmo0-qmo2))

    def test_fock(rhf):
        gf2 = KRAGF2(rhf)
        eri_df = gf2.ao2mo()
        gf2.direct = False
        eri = gf2.ao2mo()

        vj0, vk0 = rhf.get_jk(dm_kpts=rhf.make_rdm1())
        vj0 = np.einsum('kpq,kpi,kqj->kij', vj0, np.asarray(rhf.mo_coeff).conj(), np.asarray(rhf.mo_coeff))
        vk0 = np.einsum('kpq,kpi,kqj->kij', vk0, np.asarray(rhf.mo_coeff).conj(), np.asarray(rhf.mo_coeff))

        vj1, vk1 = gf2.get_jk(eri.eri,    [x.make_rdm1() for x in gf2.init_gf(eri)])
        vj2, vk2 = gf2.get_jk(eri_df.eri, [x.make_rdm1() for x in gf2.init_gf(eri_df)])

        print('J', np.allclose(vj0, vj1), np.allclose(vj0, vj2), np.linalg.norm(vj0-vj1), np.linalg.norm(vj0-vj2))
        print('K', np.allclose(vk0, vk1), np.allclose(vk0, vk2), np.linalg.norm(vk0-vk1), np.linalg.norm(vk0-vk2))


    from ase.lattice import bulk
    from pyscf.pbc.tools import pyscf_ase
    from pyscf_cache import pyscf_cache

    pyscf_cache.apply_cache()

    cell = gto.C(atom='He 1 0 1; He 0 0 1', 
                 basis='6-31g', 
                 a=np.eye(3)*3, 
                 #mesh=[20,]*3,
                 #ke_cutoff=100,
                 precision=1e-8,
                 verbose=3 if mpi_helper.rank == 0 else 0)

    #cell = gto.C(unit = 'B',
    #             a = [[4.6298286730500005, 0.0, 0.0], [-2.3149143365249993, 4.009549246030899, 0.0], [0.0, 0.0, 25]],
    #             atom = 'C 0 0 0; C 0 2.67303283 0',
    #             mesh = [20,20,20],
    #             ke_cutoff = 200,
    #             dimension = 2,
    #             low_dim_ft_type = None,
    #             pseudo = 'gth-pade',
    #             verbose = 3 if mpi_helper.rank == 0 else 0,
    #             precision = 1e-6,
    #             basis = 'gth-szv')

    ase_atom = bulk('Si', 'diamond', a=5.43102)
    #ase_atom = bulk('C', 'diamond', a=5.43102)
    cell = pyscf_cache.Cell()
    cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
    cell.a = ase_atom.cell[:]
    cell.max_memory = 20000
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.precision = 1e-8
    cell.exp_to_discard = 0.1
    cell.verbose = 4 if mpi_helper.rank == 0 else 0
    cell.build()

    rhf = pyscf_cache.KRHF(cell)
    #rhf.with_df = pyscf_cache.MDF(cell)
    rhf.kpts = cell.make_kpts([2,1,1])
    rhf.with_df = df.GDF(cell, rhf.kpts)
    rhf.with_df.build()
    rhf.exxdiv = None

    rhf.run()

    test_eri(rhf)
    test_fock(rhf)
    
    #gf2a = KRAGF2(rhf)
    #gf2a.direct = False
    #gf2a.damping = 0.5
    #gf2a.max_cycle = 50
    #gf2a.run()

    gf2b = KRAGF2(rhf)
    gf2b.direct = False
    gf2b.conv_tol = 1e-5
    gf2b.max_cycle = 20
    gf2b.keep_exxdiv = False
    gf2b.damping = 0.0
    gf2b.run()

    #mp2 = mp.KMP2(rhf)
    #mp2.run()
    ##print(rhf.e_tot, mp2.e_corr, mp2.e_tot)
    ##print(gf2a.e_init, gf2a.e_1b, gf2a.e_2b, gf2a.e_tot, gf2a.converged)
    ##print(gf2b.e_init, gf2b.e_1b, gf2b.e_2b, gf2b.e_tot, gf2b.converged)

    #print('MP2:')
    #print(mp2.e_corr)
    ##print(gf2a.e_init)
    #print(gf2b.e_init)
    #print('GF2:')
    ##print(gf2a.e_1b, gf2a.e_2b, gf2a.e_corr, gf2a.e_tot, gf2a.converged)
    #print(gf2b.e_1b, gf2b.e_2b, gf2b.e_corr, gf2b.e_tot, gf2b.converged)

