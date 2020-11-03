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

import warnings
import h5py
warnings.simplefilter('ignore', h5py.h5py_warnings.H5pyDeprecationWarning)

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
from pyscf.pbc import tools, df
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.cc.ccsd import _adjust_occ
from pyscf.agf2 import aux, ragf2, _agf2, mpi_helper
from pyscf.agf2.chempot import binsearch_chempot, minimize_chempot
from pyscf.pbc.mp.kmp2 import get_nocc, get_nmo, get_frozen_mask, \
                              padding_k_idx, padded_mo_coeff, padded_mo_energy

#TODO: memory warning if direct=False
#TODO: change some allreduce to allgather 


#NOTE: printing IPs and EAs doesn't work with ragf2 kernel - fix later for better inheritance 
def kernel(agf2, eri=None, gf=None, se=None, verbose=None, dump_chk=True):

    log = logger.new_logger(agf2, verbose)
    cput1 = cput0 = (time.clock(), time.time())
    name = agf2.__class__.__name__

    if eri is None: eri = agf2.ao2mo()
    if gf is None: gf = self.gf
    if se is None: se = self.se
    if verbose is None: verbose = agf2.verbose

    if gf is None:
        gf = agf2.init_gf()
        gf_froz = agf2.init_gf(frozen=True)
    else:
        gf_froz = gf

    if se is None:
        se = agf2.build_se(eri, gf_froz)

    if dump_chk:
        agf2.dump_chk(gf=gf, se=se)

    e_init = agf2.energy_mp2(agf2.mo_energy, se)
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

    nkpts = len(gf_occ)
    nmo_per_kpt = [x.nphys for x in gf_occ]  # should be padded, but this supports point-group symmetry
    tol = agf2.weight_tol
    facs = dict(os_factor=os_factor, ss_factor=ss_factor)

    khelper = agf2.khelper
    kconserv = khelper.kconserv

    if isinstance(eri.eri, tuple):
        fmo2qmo = _make_qmo_eris_direct
    else:
        fmo2qmo = _make_qmo_eris_incore

    vv = [np.zeros((nmo, nmo), dtype=np.complex128) for nmo in nmo_per_kpt]
    vev = [np.zeros((nmo, nmo), dtype=np.complex128) for nmo in nmo_per_kpt]

    fpos = os_factor + ss_factor
    fneg = -ss_factor

    #NOTE: we can loop over x,i,j and determine a, or x,i,a and determine j. The former requires oo+vv
    # operations with o(o+1)/2 + v(v+1)/2 intermediates, the latter requires 2ia operations and intermediates.
    # I think the latter is better and also gives a more reliable platform for parallelism.

    if isinstance(eri.eri, (tuple, list)):
        for kxia in mpi_helper.nrange(nkpts**3):
            kxi, ka = divmod(kxia, nkpts)
            kx, ki = divmod(kxi, nkpts)
            kj = kconserv[kx,ki,ka]

            ci, ei, ni = gf_occ[ki].coupling, gf_occ[ki].energy, gf_occ[ki].naux
            cj, ej, nj = gf_occ[kj].coupling, gf_occ[kj].energy, gf_occ[kj].naux
            ca, ea, na = gf_vir[ka].coupling, gf_vir[ka].energy, gf_vir[ka].naux
            nmo = nmo_per_kpt[kx]

            qxi, qja = fmo2qmo(agf2, eri, (ci,cj,ca), (kx,ki,kj,ka))
            qxj, qia = fmo2qmo(agf2, eri, (cj,ci,ca), (kx,kj,ki,ka))
            naux = qxi.shape[0]
            eja = lib.direct_sum('j,a->ja', ej, -ea).ravel()

            #TODO: mesh can be large - block the dot products
            for i in range(ni):
                qx = np.array(qxi.reshape(naux, nmo, ni)[:,:,i])
                xija = lib.dot(qx.T.conj(), qja)
                xjia = lib.dot(qxj.T.conj(), np.array(qia[:,i*na:(i+1)*na]))
                xjia = xjia.reshape(nmo, nj*na)
                xjia = fpos * xija + fneg * xjia

                vv[kx] = lib.dot(xija, xjia.T.conj(), beta=1, c=vv[kx])

                eija = eja + ei[i]
                exija = xija * eija[None]

                vev[kx] = lib.dot(exija, xjia.T.conj(), beta=1, c=vev[kx])

    else:
        for kxia in mpi_helper.nrange(nkpts**3):
            kxi, ka = divmod(kxia, nkpts)
            kx, ki = divmod(kxi, nkpts)
            kj = kconserv[kx,ki,ka]

            ci, ei, ni = gf_occ[ki].coupling, gf_occ[ki].energy, gf_occ[ki].naux
            cj, ej, nj = gf_occ[kj].coupling, gf_occ[kj].energy, gf_occ[kj].naux
            ca, ea, na = gf_vir[ka].coupling, gf_vir[ka].energy, gf_vir[ka].naux
            nmo = nmo_per_kpt[kx]

            pija = fmo2qmo(agf2, eri, (ci,cj,ca), (kx,ki,kj,ka))
            pjia = fmo2qmo(agf2, eri, (cj,ci,ca), (kx,kj,ki,ka))
            eja = lib.direct_sum('j,a->ja', ej, -ea).ravel()

            for i in range(ni):
                xija = np.array(pija[:,i].reshape(nmo, -1))
                xjia = np.array(pjia[:,:,i].reshape(nmo, -1))
                xjia = fpos * xija + fneg * xjia

                vv[kx] = lib.dot(xija, xjia.T.conj(), beta=1, c=vv[kx])

                eija = eja + ei[i]
                exija = xija * eija[None]

                vev[kx] = lib.dot(exija, xjia.T.conj(), beta=1, c=vev[kx])


    mpi_helper.barrier()
    for kx in range(nkpts):
        mpi_helper.allreduce_safe_inplace(vv[kx])
        mpi_helper.allreduce_safe_inplace(vev[kx])


    #for kx in range(nkpts):
    #    for ki in range(nkpts):
    #        for ka in range(nkpts):
    #            kj = kconserv[kx,ki,ka]

    #            ci, ei = gf_occ[ki].coupling, gf_occ[ki].energy
    #            cj, ej = gf_occ[kj].coupling, gf_occ[kj].energy
    #            ca, ea = gf_vir[ka].coupling, gf_vir[ka].energy

    #            ni = ei.size
    #            nj = ej.size
    #            na = ea.size
    #            nmo = nmo_per_kpt[kx]

    #            if isinstance(eri.eri, (tuple, list)):
    #                qxi, qja = fmo2qmo(agf2, eri, (ci,cj,ca), (kx,ki,kj,ka))
    #                qxj, qia = fmo2qmo(agf2, eri, (cj,ci,ca), (kx,kj,ki,ka))
    #                naux = qxi.shape[0]
    #                eja = lib.direct_sum('j,a->ja', ej, -ea).ravel()

    #                for i in range(ni):
    #                    qx = np.array(qxi.reshape(naux, nmo, ni)[:,:,i])
    #                    xija = lib.dot(qx.T.conj(), qja)
    #                    xjia = lib.dot(qxj.T.conj(), np.array(qia[:,i*na:(i+1)*na]))
    #                    xjia = xjia.reshape(nmo, nj*na)

    #                    eija = eja + ei[i]

    #                    vv[kx] = lib.dot(xija, xija.T.conj(), alpha=fpos, beta=1, c=vv[kx])
    #                    vv[kx] = lib.dot(xija, xjia.T.conj(), alpha=fneg, beta=1, c=vv[kx])

    #                    exija = xija * eija[None]

    #                    vev[kx] = lib.dot(exija, xija.T.conj(), alpha=fpos, beta=1, c=vev[kx])
    #                    vev[kx] = lib.dot(exija, xjia.T.conj(), alpha=fneg, beta=1, c=vev[kx])

    #            else:
    #                pija = fmo2qmo(agf2, eri, (ci,cj,ca), (kx,ki,kj,ka))
    #                pjia = fmo2qmo(agf2, eri, (cj,ci,ca), (kx,kj,ki,ka))
    #                eja = lib.direct_sum('j,a->ja', ej, -ea).ravel()

    #                for i in range(ni):
    #                    xija = np.array(pija[:,i].reshape(nmo, -1))
    #                    xjia = np.array(pjia[:,:,i].reshape(nmo, -1))

    #                    eija = eja + ei[i]

    #                    vv[kx] = lib.dot(xija, xija.T.conj(), alpha=fpos, beta=1, c=vv[kx])
    #                    vv[kx] = lib.dot(xija, xjia.T.conj(), alpha=fneg, beta=1, c=vv[kx])

    #                    exija = xija * eija[None]

    #                    vev[kx] = lib.dot(exija, xija.T.conj(), alpha=fpos, beta=1, c=vev[kx])
    #                    vev[kx] = lib.dot(exija, xjia.T.conj(), alpha=fneg, beta=1, c=vev[kx])


    #####NOTE: DIIS hack to be removed or implemented properly - does this even help??
    if not hasattr(agf2, '_diis'):
        agf2._diis = [lib.diis.DIIS(agf2), lib.diis.DIIS(agf2)]
        for x in agf2._diis:
            x.space = 15
            x.min_space = 2
    vv, vev = agf2._diis[int(gf_occ[0].energy[0] >= gf_occ[0].chempot)].update(np.array([vv, vev]))
    ######################################################################################

    se = []
    for kx in range(nkpts):
        e, c = _agf2.cholesky_build(vv[kx], vev[kx])
        se_kx = aux.SelfEnergy(e, c, chempot=gf_occ[kx].chempot)
        se_kx.remove_uncoupled(tol=tol)
        se.append(se_kx)

    log.timer('se part', *cput0)

    return se


#TODO: can we adapt these k-space get_jk functions to point-group symmetry easily?
#TODO: maybe optimised with pyscf lib
#NOTE: should we fuse loops?
def get_jk_direct(agf2, eri, rdm1, with_j=True, with_k=True):
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

    Returns:
        tuple of ndarrays corresponding to J and K at each k-point,
        if either are not requested then they are set to None.
    '''

    nkpts = len(rdm1)
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

        #for ki in range(nkpts):
        #    kj = ki
        #    for kk in range(nkpts):
        for kik in mpi_helper.nrange(nkpts**2):
            ki, kk = divmod(kik, nkpts)
            kj = ki
            buf = np.dot(qkl[ki,kj,kk], rdm1[ki].ravel(), out=buf)
            vj[ki] += np.dot(qij[ki,kj].T.conj(), buf)

        vj = vj.reshape(nkpts, nmo, nmo)

        mpi_helper.barrier()
        mpi_helper.allreduce_safe_inplace(vj)

    if with_k:
        vk = np.zeros((nkpts, nmo, nmo), dtype=dtype)
        buf = np.zeros((naux*nmo, nmo), dtype=dtype)

        #for ki in range(nkpts):
        #    kj = ki
        #    for kk in range(nkpts):
        for kik in mpi_helper.nrange(nkpts**2):
            ki, kk = divmod(kik, nkpts)
            kj = ki
            kl = agf2.khelper.kconserv[ki,kj,kk]
            buf = lib.dot(qij[ki,kl].reshape(-1, nmo).conj(), rdm1[ki], c=buf)
            buf = buf.reshape(-1, nmo, nmo).swapaxes(1,2).reshape(-1, nmo)
            vk[ki] = lib.dot(buf.T, qkl[ki,kl,kk].reshape(-1, nmo), c=vk[ki], beta=1).T   #TODO: should that be .T.conj() ?

        mpi_helper.barrier()
        mpi_helper.allreduce_safe_inplace(vk)

    return vj, vk


#TODO: check
#TODO: optimise
def get_jk_incore(agf2, eri, rdm1, with_j=True, with_k=True):
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

        #for ki in range(nkpts):
        #    kj = ki
        #    for kk in range(nkpts):
        for kik in mpi_helper.nrange(nkpts**2):
            ki, kk = divmod(kik, nkpts)
            kj = ki
            vj[ki] += lib.einsum('ijkl,lk->ij', eri[ki,kj,kk], rdm1[ki].conj())

        mpi_helper.barrier()
        mpi_helper.allreduce_safe_inplace(vj)

    if with_k:
        vk = np.zeros((nkpts, nmo, nmo), dtype=dtype)

        #for ki in range(nkpts):
        #    kj = ki
        #    for kk in range(nkpts):
        for kik in mpi_helper.nrange(nkpts**2):
            ki, kk = divmod(kik, nkpts)
            kj = ki
            kl = agf2.khelper.kconserv[ki,kj,kk]
            vk[ki] += lib.einsum('ilkj,lk->ij', eri[ki,kl,kk], rdm1[ki].conj())

        mpi_helper.barrier()
        mpi_helper.allreduce_safe_inplace(vk)

    return vj, vk


def get_jk(agf2, eri, rdm1, with_j=True, with_k=True):
    if isinstance(eri, (tuple, list)):
        vj, vk = get_jk_direct(agf2, eri, rdm1, with_j=with_j, with_k=with_k)
    else:
        vj, vk = get_jk_incore(agf2, eri, rdm1, with_j=with_j, with_k=with_k)

    return vj, vk
    

def get_ewald(agf2, rdm1):
    ''' Get the Ewald exchange contribution. The density matrix is
        assumed to be in MO basis, i.e. the overlap is identity.

    Args:
        rdm1 : list of 2D array
            Reduced density matrix at each k-point

    Returns:
        ndarray of Ewald exchange contribution at each k-point
    '''

    madelung = tools.pbc.madelung(agf2.cell, agf2.kpts)
    ewald = [madelung * x for x in rdm1]

    return ewald


def get_fock(agf2, eri, gf=None, rdm1=None):
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

    Returns:
        ndarray of physical space Fock matrix at each k-point
    '''

    if rdm1 is None:
        rdm1 = agf2.make_rdm1(gf)

    vj, vk = agf2.get_jk(eri.eri, rdm1)
    #NOTE: AFTDF seemed to have relatively large hermiticity errors on the diagonal? #NOTE this may be fixed
    #assert np.all(np.absolute(np.einsum('kii->ki', rdm1).imag) < 1e-10)
    #assert np.all(np.absolute(np.einsum('kii->ki', vj).imag) < 1e-10)
    #assert np.all(np.absolute(np.einsum('kii->ki', vk).imag) < 1e-10)

    #NOTE: should keep_exxdiv be considered here?
    if agf2.keep_exxdiv and agf2._scf.exxdiv == 'ewald':
        vk += get_ewald(agf2, rdm1)

    fock = eri.h1e + vj - 0.5 * vk

    return fock


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
    fock = agf2.get_fock(eri, gf)

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
                w, v = se[kx].eig(fock[kx], chempot=0.0)
                se[kx].chempot, nerr_kpt = binsearch_chempot((w, v), nmo, nelec)
                nerr = nerr_kpt.real if abs(nerr_kpt.real) > nerr else nerr

                w, v = se[kx].eig(fock[kx])
                gf[kx] = aux.GreensFunction(w, v[:nmo], chempot=se[kx].chempot)
                gf[kx].remove_uncoupled(tol=agf2.weight_tol)

            fock = agf2.get_fock(eri, gf)
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
            raise ValueError('Frozen orbitals not support for KAGF2')

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
        self.diis_space = getattr(__config__, 'pbc_agf2_diis_space', 6)
        self.diis_min_space = getattr(__config__, 'pbc_agf2_diis_min_space', 1)
        self.os_factor = getattr(__config__, 'pbc_agf2_os_factor', 1.0)
        self.ss_factor = getattr(__config__, 'pbc_agf2_ss_factor', 1.0)
        self.damping = getattr(__config__, 'pbc_agf2_damping', 0.0)
        self.direct = getattr(__config__, 'pbc_agf2_direct', True)
        self.keep_exxdiv = getattr(__config__, 'pbc_agf2_keep_exxdiv', False)

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

        if self.direct:
            eri = _make_mo_eris_direct(self, mo_coeff)
        else:
            eri = _make_mo_eris_incore(self, mo_coeff)

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

    def get_fock(self, eri=None, gf=None, rdm1=None):
        ''' Computes the physical space Fock matrix in MO basis at
            each k-point.
        '''

        if eri is None: eri = self.ao2mo()
        if gf is None: gf = self.gf

        return get_fock(self, eri, gf=gf, rdm1=rdm1)

    def energy_mp2(self, mo_energy=None, se=None):
        if mo_energy is None: mo_energy = self.mo_energy
        if se is None: se = self.build_se(gf=self.gf)

        mo_energy = padded_mo_energy(self, mo_energy)

        self.e_init = energy_mp2(self, mo_energy, se)

        return self.e_init

    #TODO: perhaps this would be best c.f. frozen where we can have padded coupled to non-padded?
    #TODO: frozen
    def init_gf(self, frozen=False):
        ''' Builds the Hartree-Fock Green's function.

        Returns:
            :class:`GreensFunction`, :class:`SelfEnergy`
        '''

        energy = padded_mo_energy(self, self.mo_energy)
        coupling = np.eye(self.nmo)
        nocc = get_nocc(self, per_kpoint=True)

        gf = []

        for kx in range(self.nkpts):
            try:
                chempot = binsearch_chempot(np.diag(energy[kx]), self.nmo, nocc[kx]*2)[0]
            except IndexError as e:  #TODO: fix this, I think nocc=0 or nvir=0 is feasible in k-space?
                print(e)
                chempot = 0.0
            gf.append(aux.GreensFunction(energy[kx], coupling, chempot=chempot))

        return gf

    def build_gf(self, eri=None, gf=None, se=None):
        ''' Builds the auxiliaries of the Green's function by solving
            the Dyson equation at each k-point.

        Kwargs:
            eri : _ChemistsERIs
                Electronic repulsion integrals
            gf : list of GreensFunction
                Auxiliaries of the Green's function at each k-point
            se : list of SelfEnergy
                Auxiliaries of the self-energy at each k-point

        Returns:
            :class:`GreensFunction` at each k-point
        '''

        if eri is None: eri = self.ao2mo()
        if gf is None: gf = self.gf
        if gf is None: gf = self.init_gf()
        if se is None: se = self.build_se(eri, gf)

        fock = self.get_fock(eri, gf)

        se = []

        for s, f in zip(se, fock):
            se.append(s.get_greens_function(f))

        return se

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
        if gf is None: gf = self.init_gf()

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
                e, c = _agf2.cholesky_build(vv, vev)
                se_occ = aux.SelfEnergy(e, c, chempot=se_occ.chempot)
                
                vv = np.dot(se_vir.coupling, se_vir.coupling.T.conj())
                vev = np.dot(se_vir.coupling * se_vir.energy[None], se_vir.coupling.T.conj())
                e, c = _agf2.cholesky_build(vv, vev)
                se_vir = aux.SelfEnergy(e, c, chempot=se_vir.chempot)

                se[kx] = aux.combine(se_occ, se_vir)

        return se


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
            gf = self.init_gf()
            gf_froz = self.init_gf(frozen=True)
        else:
            gf_froz = gf

        if se is None:
            se = self.build_se(eri, gf_froz)

        self.converged, self.e_1b, self.e_2b, self.gf, self.se = \
                kernel(self, eri=eri, gf=gf, se=se, verbose=self.verbose, dump_chk=dump_chk)

        self._finalize()

        return self.converged, self.e_1b, self.e_2b, self.gf, self.se

    #TODO
    def dump_chk(self, chkfile=None, key='kagf2', gf=None, se=None, frozen=None, nmom=None, mo_energy=None, mo_coeff=None, mo_occ=None):
        return self

    #TODO
    def update_from_chk_(self, chkfile=None, key='agf2'):
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
        return self.get_nmo()

    @property
    def nocc(self):
        return self.get_nocc()

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
        mo_energy = padded_mo_energy(self, self.mo_energy)
        return [np.dot(mo, x.coupling) for mo,x in zip(mo_energy, self.gf)]

    @property
    def qmo_occ(self):
        coeff = [x.get_occupied().coupling for x in self.gf]
        occ = [2.0 * np.linalg.norm(c, axis=0) ** 2 for c in coeff]
        vir = [np.zeros_like(x.get_virtual().energy) for x in self.gf]
        qmo_occ = [np.concatenate([o, v]) for o,v in zip(occ, vir)]
        return qmo_occ



#TODO: parallelise get_hcore and get_veff
class _ChemistsERIs:
    ''' (pq|rs)

    Stored as (pq|J)(J|rs) if agf2.direct is True.
    '''

    def __init__(self, cell=None):
        self.mol = self.cell = cell
        self.mo_coeff = None
        self.nmo = None
        self.nocc = None
        self.kpts = None
        self.nonzero_padding = None

        self.fock = None
        self.h1e = None
        self.eri = None
        self.e_hf = None

    def _common_init_(self, agf2, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = agf2.mo_coeff

        self.mo_coeff = padded_mo_coeff(agf2, mo_coeff)
        self.mo_energy = padded_mo_energy(agf2, agf2.mo_energy)

        dm = agf2._scf.make_rdm1(agf2.mo_coeff, agf2.mo_occ)
        exxdiv = agf2._scf.exxdiv if agf2.keep_exxdiv else None
        with lib.temporary_env(agf2._scf, exxdiv=exxdiv):
            h1e_ao = agf2._scf.get_hcore()
            veff_ao = agf2._scf.get_veff(agf2.cell, dm)
            fock_ao = h1e_ao + veff_ao

        self.h1e = []
        self.fock = []
        for ki, ci in enumerate(self.mo_coeff):
            self.h1e.append(np.dot(np.dot(ci.conj().T, h1e_ao[ki]), ci))
            self.fock.append(np.dot(np.dot(ci.conj().T, fock_ao[ki]), ci))

        if not agf2.keep_exxdiv:
            madelung = tools.madelung(agf2.cell, agf2.kpts)
            self.mo_energy = [_adjust_occ(mo_e, agf2.nocc, -madelung) 
                              for k, mo_e in enumerate(self.mo_energy)]

        self.e_hf = agf2._scf.e_tot

        self.nmo = get_nmo(agf2, per_kpoint=False)
        self.nocc = get_nocc(agf2, per_kpoint=False)

        self.nonzero_padding = padding_k_idx(agf2, kind='joint')
        self.mol = self.cell = agf2.cell
        self.kpts = agf2.kpts


#TODO dtype - do we just inherit the mo_coeff.dtype or use np.result_type(eri, mo_coeff) ?
#TODO blksize, max_memory
#TODO symmetry broken (commented)
def _make_mo_eris_incore(agf2, mo_coeff=None):
    # (pq|rs) incore
    ''' Returns _ChemistsERIs
    '''

    cput0 = (time.clock(), time.time())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    eris = _ChemistsERIs()
    eris._common_init_(agf2, mo_coeff)
    with_df = agf2.with_df
    dtype = complex

    nmo = agf2.nmo
    npair = nmo * (nmo+1) // 2

    kpts = eris.kpts
    nkpts = len(kpts)
    khelper = agf2.khelper
    kconserv = khelper.kconserv

    eri = np.empty((nkpts, nkpts, nkpts, nmo, nmo, nmo, nmo), dtype=dtype)

    #for kp in range(nkpts):
    #    for kq in range(nkpts):
    #        for kr in range(nkpts):
    for kpqr in mpi_helper.nrange(nkpts**3):
        kpq, kr = divmod(kpqr, nkpts)
        kp, kq = divmod(kpq, nkpts)
        ks = kconserv[kp,kq,kr]

        coeffs = eris.mo_coeff[[kp,kq,kr,ks]]
        kijkl = kpts[[kp,kq,kr,ks]]

        eri_kpt = with_df.ao2mo(coeffs, kijkl, compact=False)
        eri_kpt = eri_kpt.reshape(nmo, nmo, nmo, nmo)

        if dtype in [np.float, np.float64]:
            eri_kpt = eri_kpt.real

        eri[kp,kq,kr] = eri_kpt / nkpts

    #for ikp, ikq, ikr in khelper.symm_map.keys():
    #    iks = kconserv[ikp,ikq,ikr]

    #    coeffs = eris.mo_coeff[[ikp,ikq,ikr,iks]]
    #    kijkl = kpts[[ikp,ikq,ikr,iks]]

    #    eri_kpt = with_df.ao2mo(coeffs, kijkl, compact=False)
    #    eri_kpt = eri_kpt.reshape(nmo, nmo, nmo, nmo)

    #    if dtype == np.float:
    #        eri_kpt = eri_kpt.real

    #    for kp, kq, kr in khelper.symm_map[(ikp, ikq, ikr)]:
    #        eri_kpt_symm = khelper.transform_symm(eri_kpt, kp, kq, kr)
    #        eri[kp,kq,kr] = eri_kpt_symm / nkpts

    mpi_helper.barrier()
    mpi_helper.allreduce_safe_inplace(eri)

    eris.eri = eri

    log.timer('MO integral transformation', *cput0)

    return eris

    ##NOTE Uncomment to covnert to incore:
    #eris = _make_mo_eris_direct(agf2, mo_coeff)
    #eris.eri = lib.einsum('ablp,abclq->abcpq', eris.qij.conj(), eris.qkl).reshape((agf2.nkpts,)*3+(agf2.nmo,)*4)
    #del eris.qij, eris.qkl

    #return eris

def _get_naux_from_cderi(with_df):
    ''' Get the largest possible naux from the _cderi object. There
        can be different numbers at each k-point.
    '''

    cell = with_df.cell
    load3c = df.df._load3c
    kpts = with_df.kpts
    nkpts = len(kpts)

    naux = 0

    for kp in range(nkpts):
        for kq in range(nkpts):
            with load3c(with_df._cderi, 'j3c', kpts[[kp,kq]], 'j3c-kptij') as j3c:
                naux = max(naux, j3c.shape[0])

                if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
                    with load3c(with_df._cderi, 'j3c-', kpts[[kp,kq]], 'j3c-kptij',
                                ignore_key_error=True) as j3c:
                        naux = max(naux, j3c.shape[0])

    return naux

def _make_ao_eris_direct_aftdf(agf2, eris):
    ''' Get the 3c AO tensors for AFTDF '''

    with_df = agf2.with_df
    cell = with_df.cell
    dtype = complex
    kpts = eris.kpts
    nkpts = len(kpts)
    ngrids = len(cell.gen_uniform_grids(with_df.mesh))
    nao = cell.nao
    kconserv = tools.get_kconserv(cell, kpts)
    
    bra = np.zeros((nkpts, nkpts, ngrids, cell.nao**2), dtype=dtype)
    ket = np.zeros((nkpts, nkpts, nkpts, ngrids, cell.nao**2), dtype=dtype)

    kij = np.array([(ki,kj) for ki in kpts for kj in kpts])
    kis, kjs = kij[:,0], kij[:,1]
    q = kjs - kis
    ukpts, uidx, uinv = kpts_helper.unique(q)

    #for uid, kpt in enumerate(ukpts):
    for uid in mpi_helper.nrange(len(ukpts)):
        q = ukpts[uid]
        adapted_ji = np.where(uinv == uid)[0]
        kjs = kij[:,1][adapted_ji]
        fac = with_df.weighted_coulG(q, False, with_df.mesh) / nkpts

        ##TODO: fix block size?? ft_loop has max_memory argument
        for aoaoks, p0, p1 in with_df.ft_loop(with_df.mesh, q, kjs):
            for ji, aoao in enumerate(aoaoks):
                ki, kj = divmod(adapted_ji[ji], nkpts)
                bra[ki,kj,p0:p1] = fac[p0:p1,None] * aoao.reshape(p1-p0, -1)

        ki, kj = divmod(adapted_ji[0], nkpts)
        kls = kpts[kconserv[ki,kj,:]]

        for aoaoks, p0, p1 in with_df.ft_loop(with_df.mesh, q, -kls):
            for kk, aoao in enumerate(aoaoks):
                for ji, ji_idx in enumerate(adapted_ji):
                    ki, kj = divmod(ji_idx, nkpts)
                    ket[ki,kj,kk,p0:p1] = aoao.conj().reshape(p1-p0, -1)

    mpi_helper.barrier()
    mpi_helper.allreduce_safe_inplace(bra)
    mpi_helper.allreduce_safe_inplace(ket)

    return bra.conj(), ket

def _make_ao_eris_direct_fftdf(agf2, eris):
    ''' Get the 3c AO tensors for FFTDF '''

    with_df = agf2.with_df
    cell = with_df.cell
    dtype = complex
    kpts = eris.kpts
    nkpts = len(kpts)
    kconserv = tools.get_kconserv(cell, kpts)
    coords = cell.gen_uniform_grids(with_df.mesh)
    aos = with_df._numint.eval_ao(cell, coords, kpts)
    ngrids = len(coords)

    bra = np.zeros((nkpts, nkpts, ngrids, cell.nao**2), dtype=dtype)
    ket = np.zeros((nkpts, nkpts, nkpts, ngrids, cell.nao**2), dtype=dtype)

    kij = np.array([(ki,kj) for ki in kpts for kj in kpts])
    kis, kjs = kij[:,0], kij[:,1]
    q = kjs - kis
    ukpts, uidx, uinv = kpts_helper.unique(q)

    #TODO: blksize loop over mesh?
    #for uid, kpt in enumerate(ukpts):
    for uid in mpi_helper.nrange(len(ukpts)):
        q = ukpts[uid]
        adapted_ji = np.where(uinv == uid)[0]
        ki, kj = divmod(adapted_ji[0], nkpts)

        fac = tools.get_coulG(cell, q, mesh=with_df.mesh)
        fac *= (cell.vol / ngrids) / nkpts
        phase = np.exp(-1j * np.dot(coords, q))

        for ji_idx in adapted_ji:
            ki, kj = divmod(ji_idx, nkpts)

            buf = lib.einsum('gi,gj->gij', aos[ki].conj(), aos[kj])
            buf = buf.reshape(ngrids, -1)
            bra[ki,kj] = buf

            for kk in range(nkpts):
                kl = kconserv[ki,kj,kk]

                buf = lib.einsum('gi,g,gj->ijg', aos[kk].conj(), phase.conj(), aos[kl])
                buf = tools.ifft(buf.reshape(-1, ngrids), with_df.mesh) * fac
                buf = tools.fft(buf.reshape(-1, ngrids), with_df.mesh) * phase
                ket[ki,kj,kk] = lib.transpose(buf)

                buf = None

    mpi_helper.barrier()
    mpi_helper.allreduce_safe_inplace(bra)
    mpi_helper.allreduce_safe_inplace(ket)

    return bra, ket

def _make_ao_eris_direct_gdf(agf2, eris):
    ''' Get the 3c AO tensors for GDF '''
    #TODO bra-ket symmetry? Must split 1/nkpts factor and sign (could be complex?)

    with_df = agf2.with_df
    cell = with_df.cell
    dtype = complex
    kpts = eris.kpts
    nkpts = len(kpts)
    kconserv = tools.get_kconserv(cell, kpts)
    ngrids = _get_naux_from_cderi(with_df)

    bra = np.zeros((nkpts, nkpts, ngrids, cell.nao**2), dtype=dtype)
    ket = np.zeros((nkpts, nkpts, nkpts, ngrids, cell.nao**2), dtype=dtype)

    kij = np.array([(ki,kj) for ki in kpts for kj in kpts])
    kis, kjs = kij[:,0], kij[:,1]
    q = kjs - kis
    ukpts, uidx, uinv = kpts_helper.unique(q)

    #for uid, kpt in enumerate(ukpts):
    for uid in mpi_helper.nrange(len(ukpts)):
        adapted_ji = np.where(uinv == uid)[0]

        for ji_idx in adapted_ji:
            ki, kj = divmod(ji_idx, nkpts)

            p1 = 0
            for qij_r, qij_i, sign in with_df.sr_loop(kpts[[ki,kj]], compact=False):
                p0, p1 = p1, p1 + qij_r.shape[0] 
                bra[ki,kj,p0:p1] = (qij_r - qij_i * 1j) / np.sqrt(nkpts)  #TODO: + or - ?

            for kk in range(nkpts):
                kl = kconserv[ki,kj,kk]

                q1 = 0
                for qkl_r, qkl_i, sign in with_df.sr_loop(kpts[[kk,kl]], compact=False):
                    q0, q1 = q1, q1 + qkl_r.shape[0]
                    ket[ki,kj,kk,q0:q1] = (qkl_r + qkl_i * 1j) * sign / np.sqrt(nkpts)

    mpi_helper.barrier()
    mpi_helper.allreduce_safe_inplace(bra)
    mpi_helper.allreduce_safe_inplace(ket)

    return bra, ket

#TODO: write out custom function combining aftdf and gdf to avoid copying
def _make_ao_eris_direct_mdf(agf2, eris):
    ''' Get the 3c AO tensors for MDF '''

    eri0 = _make_ao_eris_direct_aftdf(agf2, eris)
    eri1 = _make_ao_eris_direct_gdf(agf2, eris)

    bra = np.concatenate((eri0[0], eri1[0]), axis=-2)
    ket = np.concatenate((eri0[1], eri1[1]), axis=-2)

    return bra, ket

def _fao2mo(eri, cp, cq, dtype, out=None):
    ''' DF ao2mo '''
    #return np.einsum('lpq,pi,qj->lij', eri.reshape(-2, cp.shape[0], cq.shape[0]), cp.conj(), cq).reshape(-1, cp.shape[1]*cq.shape[1])

    npq, cpq, spq = ao2mo.incore._conc_mos(cp, cq, compact=False)[1:]
    sym = dict(aosym='s1', mosym='s1')
    naux = eri.shape[0]

    if out is None:
        out = np.zeros((naux*cp.shape[1]*cq.shape[1]), dtype=dtype)
    out = out.reshape(naux, cp.shape[1]*cq.shape[1])
    out = out.astype(dtype)

    if dtype in [np.float, np.float64]:
        out = ao2mo._ao2mo.nr_e2(eri, cpq, spq, out=out, **sym)
    else:
        cpq = np.asarray(cpq, dtype=complex)
        out = ao2mo._ao2mo.r_e2(eri, cpq, spq, [], None, out=out)

    return out.reshape(naux, npq)

def _make_mo_eris_direct(agf2, mo_coeff=None):
    # (pq|L)(L|rs) incore

    cput0 = (time.clock(), time.time())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    eris = _ChemistsERIs()
    eris._common_init_(agf2, mo_coeff)

    cell = agf2.cell
    with_df = agf2.with_df
    kpts = eris.kpts
    nkpts = len(kpts)
    kconserv = tools.get_kconserv(cell, kpts)
    nmo = eris.nmo

    if mo_coeff is None:
        mo_coeff = eris.mo_coeff

    if type(with_df) is df.FFTDF:
        bra, ket = _make_ao_eris_direct_fftdf(agf2, eris)
    elif type(with_df) is df.AFTDF:
        bra, ket = _make_ao_eris_direct_aftdf(agf2, eris)
    elif type(with_df) is df.GDF:
        bra, ket = _make_ao_eris_direct_gdf(agf2, eris)
    elif type(with_df) is df.MDF:
        bra, ket = _make_ao_eris_direct_mdf(agf2, eris)
    else:
        raise ValueError('Unknown DF type %s' % type(with_df))

    dtype = complex
    naux = bra.shape[2]

    kij = np.array([(ki,kj) for ki in kpts for kj in kpts])
    kis, kjs = kij[:,0], kij[:,1]
    q = kjs - kis
    ukpts, uidx, uinv = kpts_helper.unique(q)

    qij = np.zeros((nkpts, nkpts, naux, nmo**2), dtype=dtype)
    qkl = np.zeros((nkpts, nkpts, nkpts, naux, nmo**2), dtype=dtype)

    #for uid, kpt in enumerate(ukpts):
    for uid in mpi_helper.nrange(len(ukpts)):
        q = ukpts[uid]
        adapted_ji = np.where(uinv == uid)[0]
        kjs = kij[:,1][adapted_ji]

        for ji, ji_idx in enumerate(adapted_ji):
            ki, kj = divmod(adapted_ji[ji], nkpts)
            ci = mo_coeff[ki].conj()
            cj = mo_coeff[kj].conj()
            qij[ki,kj] = _fao2mo(bra[ki,kj], ci, cj, dtype, out=qij[ki,kj])

            for kk in range(nkpts):
                kl = kconserv[ki,kj,kk]
                ck = mo_coeff[kk]
                cl = mo_coeff[kl]
                qkl[ki,kj,kk] = _fao2mo(ket[ki,kj,kk], ck, cl, dtype, out=qkl[ki,kj,kk])

    mpi_helper.barrier()
    mpi_helper.allreduce_safe_inplace(qij)
    mpi_helper.allreduce_safe_inplace(qkl)

    eris.qij = qij
    eris.qkl = qkl
    eris.eri = (qij, qkl)

    log.timer('MO integral transformation', *cput0)

    return eris

def _make_qmo_eris_incore(agf2, eri, coeffs, kpts):
    ''' (xi|ja)
    '''
    #return np.einsum('pqrs,qi,rj,sa->pija', eri.eri[kpts[0],kpts[1],kpts[2]], coeffs[0], coeffs[1].conj(), coeffs[2])

    cput0 = (time.clock(), time.time())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    kx, ki, kj, ka = kpts
    ci, cj, ca = coeffs

    dtype = np.result_type(eri.eri.dtype, ci.dtype, cj.dtype, ca.dtype)
    xija = np.zeros((eri.nmo**2, cj.shape[1], ca.shape[1]), dtype=dtype)

    mo = eri.eri[kx,ki,kj].reshape(-1, eri.nmo**2)
    xija = _fao2mo(mo, cj, ca, dtype, out=xija)

    xija = xija.reshape(eri.nmo, eri.nmo, cj.shape[1], ca.shape[1])
    xija = lib.einsum('xyja,yi->xija', xija, ci)

    #log.timer('QMO integral transformation', *cput0)

    return xija

def _make_qmo_eris_direct(agf2, eri, coeffs, kpts):
    ''' (xi|L)(L|rs) incore
    '''
    #return (np.einsum('lpq,qi->lpi', eri.qij[kpts[0],kpts[1]].reshape(-1, eri.nmo, eri.nmo), coeffs[0].conj()).reshape(-1, eri.nmo*coeffs[0].shape[1]), 
    #        np.einsum('lpq,pi,qj->lij', eri.qkl[kpts[0],kpts[1],kpts[2]].reshape(-1, eri.nmo, eri.nmo), coeffs[1].conj(), coeffs[2]).reshape(-1, coeffs[1].shape[1]*coeffs[2].shape[1]))

    cput0 = (time.clock(), time.time())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    kx, ki, kj, ka = kpts
    ci, cj, ca = coeffs
    
    dtype = np.result_type(eri.qij.dtype, eri.qkl.dtype, ci.dtype, cj.dtype, ca.dtype)
    naux = eri.eri[0].shape[2]

    qwx = eri.eri[0][kx,ki].reshape(-1, eri.nmo)
    qwi = np.dot(qwx, ci.conj()).reshape(naux, -1)

    qyz = eri.eri[1][kx,ki,kj].reshape(-1, eri.nmo**2)
    qja = _fao2mo(qyz, cj, ca, dtype).reshape(naux, -1)

    #log.timer('QMO integral transformation', *cput0)

    return qwi, qja


def ft_loop(self, mesh=None, q=np.zeros(3), kpts=None, shls_slice=None,
            max_memory=4000, aosym='s1', intor='GTO_ft_ovlp', comp=1):
    from pyscf.pbc.df import ft_ao

    cell = self.cell
    if mesh is None:
        mesh = self.mesh
    if kpts is None:
        assert(df.aft.is_zero(q))
    if shls_slice is None:
        shls_slice = (0, cell.nbas, 0, cell.nbas)
    kpts = np.asarray(kpts)
    nkpts = len(kpts)

    ao_loc = cell.ao_loc_nr()
    b = cell.reciprocal_vectors()
    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
    gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
    ngrids = gxyz.shape[0]

    if aosym == 's2':
        assert(shls_slice[2] == 0)
        i0 = ao_loc[shls_slice[0]]
        i1 = ao_loc[shls_slice[1]]
        nij = i1*(i1+1)//2 - i0*(i0+1)//2
    else:
        ni = ao_loc[shls_slice[1]] - ao_loc[shls_slice[0]]
        nj = ao_loc[shls_slice[3]] - ao_loc[shls_slice[2]]
        nij = ni*nj

    size, rank = mpi_helper.size, mpi_helper.rank
    per_rank = ngrids // mpi_helper.size
    start = rank * per_rank
    stop = ngrids if rank == (size-1) else (rank+1) * per_rank


    blksize = max(16, int(max_memory*.9e6/(nij*nkpts*16*comp)))
    blksize = min(blksize, per_rank, 16384)
    passes = per_rank // blksize
    blksize = (stop-start) // passes

    for npass in range(passes):
        start0 = start + npass * blksize
        stop0 = stop if npass == (passes-1) else start + (npass+1) * blksize

        buf = np.empty(nkpts*nij*(stop0-start0)*comp, dtype=np.complex128)
        dat = ft_ao._ft_aopair_kpts(cell, Gv[start0:stop0], shls_slice, 
                                    aosym, b, gxyz[start0:stop0], Gvbase, 
                                    q, kpts, intor, comp, out=buf)

        dat = dat.reshape(nkpts*comp*(stop0-start0), -1)

        for nproc in range(size):
            mpi_helper.barrier()

            shape, dtype = mpi_helper.comm.bcast((dat.shape, dat.dtype), root=nproc)
            q0, q1 = mpi_helper.comm.bcast((start0, stop0), root=nproc)

            if nproc == rank:
                buf = dat
            else:
                buf = np.empty(shape, dtype=dtype)

            buf = mpi_helper.bcast(buf, root=nproc)

            blk = shape[0] // (nkpts*comp)
            buf = buf.reshape(nkpts, comp, blk, -1)

            if comp == 1:
                buf = np.squeeze(buf, axis=1)

            yield buf, q0, q1



if __name__ == '__main__':
    from pyscf.pbc import gto, scf, cc, mp

    def test_eri(rhf):
        gf2 = KRAGF2(rhf)
        eri_df = gf2.ao2mo()
        gf2.direct = False
        eri = gf2.ao2mo()

        if type(gf2.with_df) == df.FFTDF:
            bra, ket = _make_ao_eris_direct_fftdf(gf2, eri)
        elif type(gf2.with_df) == df.AFTDF:
            bra, ket = _make_ao_eris_direct_aftdf(gf2, eri)
        elif type(gf2.with_df) == df.GDF:
            bra, ket = _make_ao_eris_direct_gdf(gf2, eri)
        elif type(gf2.with_df) == df.MDF:
            bra, ket = _make_ao_eris_direct_mdf(gf2, eri)
        ao0 = np.einsum('ablp,abclq->abcpq', bra.conj(), ket).reshape((gf2.nkpts,)*3 + (gf2.nmo,)*4)
        ao1 = rhf.with_df.ao2mo_7d(np.asarray([[np.eye(gf2.nmo),]*gf2.nkpts]*4), kpts=rhf.kpts) / len(rhf.kpts)

        eri0 = np.einsum('ablp,abclq->abcpq', eri_df.eri[0].conj(), eri_df.eri[1]).reshape((gf2.nkpts,)*3 + (gf2.nmo,)*4)
        eri1 = eri.eri
        eri2 = rhf.with_df.ao2mo_7d(np.asarray(rhf.mo_coeff)+0j, kpts=rhf.kpts) / len(rhf.kpts)

        ci = np.random.random((gf2.nmo, gf2.nocc*2)) + 1.0j * np.random.random((gf2.nmo, gf2.nocc*2)) 
        cj = np.random.random((gf2.nmo, gf2.nocc*2)) + 1.0j * np.random.random((gf2.nmo, gf2.nocc*2))  
        ca = np.random.random((gf2.nmo, (gf2.nmo-gf2.nocc)*2)) + 1.0j * np.random.random((gf2.nmo, (gf2.nmo-gf2.nocc)*2))
        qmo0 = np.einsum('pqrs,qi,rj,sa->pija', eri2[0,1,0], ci, cj.conj(), ca)
        qmo1_bra, qmo1_ket = _make_qmo_eris_direct(gf2, eri_df, (ci,cj,ca), [0,1,0,1])
        qmo1 = np.dot(qmo1_bra.conj().T, qmo1_ket).reshape(qmo0.shape)
        qmo2 = _make_qmo_eris_incore(gf2, eri, (ci,cj,ca), [0,1,0,1])

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

        vj1, vk1 = gf2.get_jk(eri.eri,    [x.make_rdm1() for x in gf2.init_gf()])
        vj2, vk2 = gf2.get_jk(eri_df.eri, [x.make_rdm1() for x in gf2.init_gf()])

        print('J', np.allclose(vj0, vj1), np.allclose(vj0, vj2), np.linalg.norm(vj0-vj1), np.linalg.norm(vj0-vj2))
        print('K', np.allclose(vk0, vk1), np.allclose(vk0, vk2), np.linalg.norm(vk0-vk1), np.linalg.norm(vk0-vk2))


    #NOTE: block of code for ft_loop without prange on each process:
    #buf = np.empty(nkpts*nij*(stop-start)*comp, dtype=np.complex128)

    #dat = ft_ao._ft_aopair_kpts(cell, Gv[start:stop], shls_slice, aosym,
    #                            b, gxyz[start:stop], Gvbase, q, kpts,
    #                            intor, comp, out=buf)

    #dat = dat.reshape(nkpts*comp*(stop-start), -1)

    #q1 = 0
    #for nproc in range(size):
    #    mpi_helper.barrier()

    #    shape, dtype = mpi_helper.comm.bcast((dat.shape, dat.dtype))

    #    if nproc == rank:
    #        buf = dat
    #    else:
    #        buf = np.empty(shape, dtype=dtype)

    #    buf = mpi_helper.bcast(buf, root=nproc)

    #    blk = shape[0] // (nkpts*comp)
    #    buf = buf.reshape(nkpts, comp, blk, -1)
    #    q0, q1 = q1, q1 + blk

    #    if comp == 1:
    #        buf = np.squeeze(buf, axis=1)

    #    yield buf, q0, q1



    class KRHF(scf.KRHF):
        def __init__(self, *args, **kwargs):
            scf.KRHF.__init__(self, *args, **kwargs)
            self._hcore = None
            self._energy_nuc = None
            self._keys.update(['_hcore', '_energy_nuc'])

        def get_hcore(self, *args, **kwargs):
            if self._hcore is None:
                self._hcore = scf.KRHF.get_hcore(self, *args, **kwargs)
            return self._hcore

        def energy_nuc(self, *args, **kwargs):
            if self._energy_nuc is None:
                self._energy_nuc = scf.KRHF.energy_nuc(self, *args, **kwargs)
            return self._energy_nuc


    from types import MethodType

    from ase.lattice import bulk
    from pyscf.pbc.tools import pyscf_ase

    cell = gto.C(atom='He 1 0 1; He 0 0 1', 
                 basis='6-31g', 
                 a=np.eye(3)*3, 
                 mesh=[30,]*3,
                 #ke_cutoff=100,
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
    cell = gto.Cell()
    cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
    cell.a = ase_atom.cell[:]
    cell.max_memory = 20000
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    #cell.exp_to_discard = 0.1
    cell.verbose = 4 if mpi_helper.rank == 0 else 0
    cell.build()

    rhf = KRHF(cell)
    rhf.with_df = df.MDF(cell)
    old_ft_loop = rhf.with_df.ft_loop
    rhf.with_df.ft_loop = MethodType(ft_loop, rhf.with_df)
    rhf.conv_tol = 1e-10
    rhf.exxdiv = 'ewald'
    rhf.kpts = cell.make_kpts([1,1,1])

    rhf.run()

    rhf.with_df.ft_loop = old_ft_loop

    #test_eri(rhf)
    #test_fock(rhf)
    
    #gf2a = KRAGF2(rhf)
    #gf2a.direct = False
    #gf2a.damping = 0.5
    #gf2a.max_cycle = 50
    #gf2a.run()

    gf2b = KRAGF2(rhf)
    gf2b.direct = True
    gf2b.damping = 0.0
    gf2b.conv_tol = 1e-6
    gf2b.max_cycle = 30
    gf2b.keep_exxdiv = True
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

