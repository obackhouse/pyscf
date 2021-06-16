'''
KRAGF2 with renormalisation only at a given set of k-points
'''

import time
import numpy as np
from pyscf.lib import logger
from pyscf.agf2 import aux, mpi_helper, _agf2
from pyscf.pbc.agf2 import kragf2, kragf2_ao2mo, _kagf2


def build_se_part(agf2, eri, gf_occ, gf_vir, se_prev, os_factor=1.0, ss_factor=1.0):
    ''' See pyscf.pbc.agf2.kragf2.build_se_part
    '''

    assert type(gf_occ[0]) is aux.GreensFunction
    assert type(gf_vir[0]) is aux.GreensFunction

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    nkpts = agf2.nkpts
    nmo = agf2.nmo
    tol = agf2.weight_tol
    facs = dict(os_factor=os_factor, ss_factor=ss_factor)
    kptlist = agf2.kptlist
    if se_prev is None:
        kptlist = list(range(nkpts))

    khelper = agf2.khelper
    kconserv = khelper.kconserv

    if agf2.direct:
        fmo2qmo = kragf2_ao2mo._make_qmo_eris_direct
    else:
        fmo2qmo = kragf2_ao2mo._make_qmo_eris_incore

    vv = np.zeros((nkpts, nmo, nmo), dtype=np.complex128)
    vev = np.zeros((nkpts, nmo, nmo), dtype=np.complex128)

    for xia in mpi_helper.nrange(len(kptlist) * nkpts**2):
        xi, ka = divmod(xia, nkpts)
        x, ki = divmod(xi, nkpts)
        kx = kptlist[x]
        kj = kconserv[kx,ki,ka]

        ci, ei, ni = gf_occ[ki].coupling, gf_occ[ki].energy, gf_occ[ki].naux
        cj, ej, nj = gf_occ[kj].coupling, gf_occ[kj].energy, gf_occ[kj].naux
        ca, ea, na = gf_vir[ka].coupling, gf_vir[ka].energy, gf_vir[ka].naux

        if agf2.direct:
            qxi, qja = fmo2qmo(agf2, eri, (ci,cj,ca), (kx,ki,kj,ka))
            qxj, qia = fmo2qmo(agf2, eri, (cj,ci,ca), (kx,kj,ki,ka))
            vv_k, vev_k = _kagf2.build_mats_kragf2_direct(
                    qxi, qja, qxj, qia, ei, ej, ea, **facs,
            )
            del qxi, qja, qxj, qia
        else:
            pija = fmo2qmo(agf2, eri, (ci,cj,ca), (kx,ki,kj,ka))
            pjia = fmo2qmo(agf2, eri, (cj,ci,ca), (kx,kj,ki,ka))
            vv_k, vev_k = _kagf2.build_mats_kragf2_incore(
                    pija, pjia, ei, ej, ea, **facs,
            )
            del pija, pjia

        vv[x] += vv_k
        vev[x] += vev_k

    mpi_helper.barrier()
    for kx in range(nkpts):
        mpi_helper.allreduce_safe_inplace(vv[kx])
        mpi_helper.allreduce_safe_inplace(vev[kx])

    se = []
    for kx in range(nkpts):
        if kx in kptlist:
            if not np.allclose(vv[kx], vv[kx].T.conj()):
                error = np.max(np.absolute(vv[kx]-vv[kx].T.conj()))
                log.debug1('0th moment not hermitian at kpt %d, '
                           'error = %.3g', kx, error)
            vv[kx] = 0.5 * (vv[kx] + vv[kx].T.conj())

            if not np.allclose(vev[kx], vev[kx].T.conj()):
                error = np.max(np.absolute(vev[kx]-vev[kx].T.conj()))
                log.debug1('1st moment not hermitian at kpt %d, '
                           'error = %.3g', kx, error)
            vev[kx] = 0.5 * (vev[kx] + vev[kx].T.conj())

            e, c = _agf2.cholesky_build(vv[kx], vev[kx], do_twice=True)
            se_kx = aux.SelfEnergy(e, c, chempot=gf_occ[kx].chempot)
            se_kx.remove_uncoupled(tol=tol)
            se.append(se_kx)

        else:
            se.append(se_prev[kx])

    log.timer('se part', *cput0)

    return se



class KRAGF2_Partial(kragf2.KRAGF2):
    def __init__(self, *args, **kwargs):
        self.kptlist = kwargs.pop('kptlist')
        kragf2.KRAGF2.__init__(self, *args, **kwargs)
        self._keys.update('kptlist')

    build_se_part = build_se_part

    def build_se(self, eri=None, gf=None, os_factor=None, ss_factor=None, se_prev=None):
        ''' See pyscf.pbc.agf2.kragf2.KRAGF2.build_se
        '''

        if eri is None: eri = self.ao2mo()
        if gf is None: gf = self.gf
        if gf is None: gf = self.init_gf(eri)

        if os_factor is None: os_factor = self.os_factor
        if ss_factor is None: ss_factor = self.ss_factor

        facs = dict(os_factor=os_factor, ss_factor=ss_factor)
        gf_occ = [x.get_occupied() for x in gf]
        gf_vir = [x.get_virtual() for x in gf]

        if se_prev is None:
            se_occ_prev = se_vir_prev = None
        else:
            se_occ_prev = [x.get_occupied() for x in se_prev]
            se_vir_prev = [x.get_virtual() for x in se_prev]

        for kx in range(self.nkpts):
            if gf_occ[kx].naux == 0 or gf_vir[kx].naux == 0:
                logger.warn(self, 'Attempting to build a self-energy with '
                                  'no (i,j,a) or (a,b,i) configurations at '
                                  'k-point %d', kx)

        se_occ = self.build_se_part(eri, gf_occ, gf_vir, se_occ_prev, **facs)
        se_vir = self.build_se_part(eri, gf_vir, gf_occ, se_vir_prev, **facs)
        se = [aux.combine(o, v) for o,v in zip(se_occ, se_vir)]

        if se_prev is not None and self.damping != 0.0:
            for kx in range(self.nkpts):
                se[kx].coupling *= np.sqrt(1.0-self.damping)
                se_prev[kx].coupling *= np.sqrt(self.damping)
                se[kx] = aux.combine(se[kx], se_prev[kx])

                se_occ, se_vir = se[kx].get_occupied(), se[kx].get_virtual()

                vv = np.dot(se_occ.coupling, se_occ.coupling.T.conj())
                vev = np.dot(se_occ.coupling * se_occ.energy[None], 
                             se_occ.coupling.T.conj())
                e, c = _agf2.cholesky_build(vv, vev)
                se_occ = aux.SelfEnergy(e, c, chempot=se_occ.chempot)

                vv = np.dot(se_vir.coupling, se_vir.coupling.T.conj())
                vev = np.dot(se_vir.coupling * se_vir.energy[None], 
                             se_vir.coupling.T.conj())
                e, c = _agf2.cholesky_build(vv, vev)
                se_vir = aux.SelfEnergy(e, c, chempot=se_vir.chempot)

                se[kx] = aux.combine(se_occ, se_vir)

        return se



if __name__ == '__main__':
    from pyscf.pbc import gto, scf, df, agf2
    from gmtkn import sets
    system = sets['GAPS'].systems['Si']

    cell = gto.Cell()
    cell.atom = list(zip(system['atoms'], system['coords']))
    cell.a = system['a']
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.exp_to_discard = 0.1
    cell.verbose = 3
    cell.build()

    rhf = scf.KRHF(cell)
    rhf.kpts = cell.make_kpts([3,3,3])
    rhf.with_df = df.GDF(cell, rhf.kpts)
    rhf.with_df.build()
    rhf.exxdiv = None
    rhf.run()

    gf2_partial = KRAGF2_Partial(rhf, kptlist=[0])
    gf2_partial.damping = 0.5
    gf2_partial.diis_space = 10
    gf2_partial.conv_tol = 1e-5
    gf2_partial.run()

    gf2 = agf2.KRAGF2(rhf)
    gf2.damping = 0.5
    gf2.diis_space = 10
    gf2.conv_tol = 1e-5
    gf2.run()
