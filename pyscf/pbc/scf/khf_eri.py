'''
Hartree-Fock for periodic systems with k-point sampling
and four-centre ERIs
'''

import time
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.scf import khf
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0
from pyscf.pbc.df.df_jk import _format_dms, _format_kpts_band, _format_jks
from pyscf.pbc.df import df_incore, GDF
from pyscf.agf2 import mpi_helper


def get_jk(mf, cell, dm_kpts, kpts, kpts_band=None, with_j=True, with_k=True,
           omega=None, exxdiv=None):
    ''' Get the J and K contributions to the Fock matrix via the 4c ERIs.
    '''

    if mf._eri is None:
        mf._build_eri()
    if kpts is None:
        kpts = mf.kpts
    if not (kpts_band is None or numpy.allclose(kpts, kpts_band)):
        # kpts_band won't have ERIs - will support most of the syntax
        # below for now but won't work yet
        raise NotImplementedError("kpts_band for 4c ERI KHF")

    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]
    dtype = numpy.result_type(mf._eri.dtype, *[x.dtype for x in dms])
    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)

    vj = vk = None
    if with_j:
        vj = numpy.zeros((nset, nband, nao, nao), dtype=dtype)
    if with_k:
        vk = numpy.zeros((nset, nband, nao, nao), dtype=dtype)

    for kik in mpi_helper.nrange(nband*nkpts):
        ki, kk = divmod(kik, nband)
        kj = ki
        kl = mf.kconserv[ki,kj,kk]

        for i in range(nset):
            dm = dms[i,kl].conj()
            if with_j:
                vj[i,ki] += lib.einsum('ijkl,lk->ij', mf._eri[ki,kj,kk], dm)
            if with_k:
                vk[i,ki] += lib.einsum('ilkj,lk->ij', mf._eri[ki,kl,kk], dm)

    if with_k and exxdiv == 'ewald':
        _ewald_exxdiv_for_G0(cell, kpts, dms, vk, kpts_band)

    vj = _format_jks(vj, dm_kpts, input_band, kpts)
    vk = _format_jks(vk, dm_kpts, input_band, kpts)

    return vj, vk


class KHF(khf.KRHF):
    def __init__(self, *args, **kwargs):
        khf.KRHF.__init__(self, *args, **kwargs)
        self._eri = None
        self._h1e = None
        self.kconserv = kpts_helper.get_kconserv(self.cell, self.kpts)
        for key in ['_eri', '_h1e', 'kconserv']:
            self._keys.add(key)

    def _build_eri(self):
        #TODO optimisations
        nao, nkpts = self.cell.nao, len(self.kpts)
        shape = (nkpts, nkpts, nkpts, nao, nao, nao, nao)
        self._eri = numpy.zeros(shape, dtype=numpy.complex128)

        if isinstance(self.with_df, (df_incore.IncoreGDF, GDF)):
            self._eri = df_incore._make_eri(
                    self.with_df, self.kpts,
                    kconserv=self.kconserv,
                    out=self._eri,
            )

        else:
            for ki, kpti in enumerate(self.kpts):
                for kj, kptj in enumerate(self.kpts):
                    for kk, kptk in enumerate(self.kpts):
                        kl = self.kconserv[ki,kj,kk]
                        kptl = self.kpts[kl]
                        kptijkl = (kpti, kptj, kptk, kptl)

                        v = self.with_df.get_eri(kptijkl, compact=False)
                        v /= nkpts
                        self._eri[ki,kj,kk] = v.reshape(nao,nao,nao,nao)

    def get_jk(self, cell=None, dm_kpts=None, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, **kwargs):
        if cell is None: cell = self.cell
        if kpts is None: kpts = self.kpts
        if dm_kpts is None: dm_kpts = self.make_rdm1()
        cpu0 = (time.clock(), time.time())
        if self.rsjk:
            raise NotImplementedError("rsjk for 4c ERI KHF")
        else:
            vj, vk = get_jk(self, cell, dm_kpts, kpts, kpts_band,
                            with_j, with_j, omega, self.exxdiv)
        logger.timer(self, 'vj and vk', *cpu0)
        return vj, vk

    def get_hcore(self, cell=None, kpts=None):
        if self._h1e is None:
            self._h1e = khf.get_hcore(self, cell, kpts)
        return self._h1e

    @property
    def kpts(self):
        if 'kpts' in self.__dict__:
            self.kpt = self.__dict__.pop('kpts')
        return self.with_df.kpts
    @kpts.setter
    def kpts(self, x):
        self.with_df.kpts = numpy.reshape(x, (-1,3))
        self.kconserv = kpts_helper.get_kconserv(self.cell, self.kpts)


KRHF = KHF
