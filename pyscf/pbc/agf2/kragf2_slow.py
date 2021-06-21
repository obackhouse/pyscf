import time
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.agf2 import aux, mpi_helper
from pyscf.pbc.agf2 import kragf2, kragf2_ao2mo
import dyson


def build_se_part(agf2, eri, gf_occ, gf_vir, os_factor=1.0, ss_factor=1.0):
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(agf2.stdout, agf2.verbose)

    assert type(gf_occ[0]) is aux.GreensFunction
    assert type(gf_vir[0]) is aux.GreensFunction

    nkpts = agf2.nkpts
    nmo = agf2.nmo
    nmom = agf2.nmom
    tol = agf2.weight_tol

    fpos = os_factor + ss_factor
    fneg = -ss_factor

    khelper = agf2.khelper
    kconserv = khelper.kconserv

    if agf2.direct:
        fmo2qmo = kragf2_ao2mo._make_qmo_eris_direct
    else:
        fmo2qmo = kragf2_ao2mo._make_qmo_eris_incore

    t = np.zeros((nkpts, 2*nmom[1]+2, nmo, nmo), dtype=np.complex128)

    for kxia in mpi_helper.nrange(nkpts**3):
        kxi, ka = divmod(kxia, nkpts)
        kx, ki = divmod(kxi, nkpts)
        kj = kconserv[kx,ki,ka]

        ci, ei, ni = gf_occ[ki].coupling, gf_occ[ki].energy, gf_occ[ki].naux
        cj, ej, nj = gf_occ[kj].coupling, gf_occ[kj].energy, gf_occ[kj].naux
        ca, ea, na = gf_vir[ka].coupling, gf_vir[ka].energy, gf_vir[ka].naux

        eja = lib.direct_sum('j,a->ja', ej, -ea).ravel()

        if agf2.direct:
            qxi, qja = fmo2qmo(agf2, eri, (ci,cj,ca), (kx,ki,kj,ka))
            qxj, qia = fmo2qmo(agf2, eri, (cj,ci,ca), (kx,kj,ki,ka))
            for i in range(ni):
                qx = np.array(qxi.reshape(-1, nmo, ni)[:,:,i])
                xija = lib.dot(qx.T, qja)
                xjia = lib.dot(qxj.T, np.array(qia[:,i*na:(i+1)*na]))
                xjia = xjia.reshape(nmo, nj*na)
                xjia = fpos * xija + fneg * xjia
                eija = eja + ei[i]

                v1 = xija
                for n in range(2*nmom[1]+2):
                    t[kx,n] = lib.dot(v1, xjia.T.conj(), beta=1, c=t[kx,n])
                    v1 *= eija[None]

        else:
            xija = fmo2qmo(agf2, eri, (ci,cj,ca), (kx,ki,kj,ka))
            xjia = fmo2qmo(agf2, eri, (cj,ci,ca), (kx,kj,ki,ka))

            xija = np.array(xija.reshape(nmo, -1))
            xjia = np.array(xjia.reshape(nmo, -1))
            xjia = fpos * xija + fneg * xjia
            eija = lib.direct_sum('i,ja->ija', ei, eja)

            v1 = xija
            for n in range(2*nmom[1]+2):
                t[kx,n] = lib.dot(v1, xjia.T.conj(), beta=1, c=t[kx,n])
                v1 *= eija[None]

    mpi_helper.barrier()
    for kx in range(nkpts):
        mpi_helper.allreduce_safe_inplace(t[kx])

    se = []
    for kx in range(nkpts):
        for i in range(2*nmom[1]+2):
            if not np.allclose(t[kx,i], t[kx,i].T.conj()):
                error = np.max(np.absolute(t[kx,i]-t[kx,i].T.conj()))
                log.debug1('Moment %d not hermitian at kpt %d, '
                           'error = %.3g', i, kx, error)
            t[kx,i] = 0.5 * (t[kx,i] + t[kx,i].T.conj())

        m, b = dyson.block_lanczos_se.block_lanczos(t[kx], agf2.nmom[1])
        m = dyson.linalg.build_block_tridiagonal(m, b)
        e, v = np.linalg.eigh(m[nmo:,nmo:])
        v = np.dot(b[0].T.conj(), v[:nmo])

        se_kx = aux.SelfEnergy(e, v, chempot=gf_occ[kx].chempot)
        se_kx.remove_uncoupled(tol=tol)
        se.append(se_kx)

    log.timer('se part', *cput0)

    return se


class KRAGF2(kragf2.KRAGF2):
    def __init__(self, mf, nmom=(None,0), frozen=None, mo_energy=None, mo_coeff=None, mo_occ=None):
        
        kragf2.KRAGF2.__init__(self, mf, frozen=frozen, mo_energy=mo_energy,
                               mo_coeff=mo_coeff, mo_occ=mo_occ)

        self.nmom = nmom

        self._keys.update(['nmom'])

    build_se_part = build_se_part

    def build_se(self, eri=None, gf=None, os_factor=None, ss_factor=None, se_prev=None):
        if eri is None: eri = self.ao2mo()
        if gf is None: gf = self.gf
        if gf is None: gf = self.init_gf(eri)

        if os_factor is None: os_factor = self.os_factor
        if ss_factor is None: ss_factor = self.ss_factor

        facs = dict(os_factor=os_factor, ss_factor=ss_factor)
        gf_occ = [x.get_occupied() for x in gf]
        gf_vir = [x.get_virtual() for x in gf]

        if self.nmom[0] is None:
            fock = [None for kx in range(self.nkpts)]
        else:
            fock = self.get_fock()

        se_occ = self.build_se_part(eri, gf_occ, gf_vir, **facs)
        se_vir = self.build_se_part(eri, gf_vir, gf_occ, **facs)

        se = [aux.combine(o, v) for o,v in zip(se_occ, se_vir)]
        se = [se_k.compress(phys=fock, n=(self.nmom[0], None)) for se_k in se]

        if se_prev is not None and self.damping != 0.0:
            for kx in range(self.nkpts):
                se[kx].coupling *= np.sqrt(1.0-self.damping)
                se_prev[kx].coupling *= np.sqrt(self.damping)
                se[kx] = aux.combine(se[kx], se_prev[kx])

                #se_occ, se_vir = se[kx].get_occupied(), se[kx].get_virtual()

                #t_occ = se_occ.moment(range(2*self.nmom[1]+2))
                #t_vir = se_vir.moment(range(2*self.nmom[1]+2))

                #e, v = dyson.block_lanczos_se.kernel(t_occ, t_vir, self.nmom[1])

                #se[kx] = aux.SelfEnergy(e, v, chempot=se[kx].chempot)

                se[kx] = se[kx].compress(phys=fock, n=self.nmom)

        return se

    def dump_flags(self, verbose=None):
        kragf2.KRAGF2.dump_flags(self, verbose=verbose)
        logger.info(self, 'nmom = %s', repr(self.nmom))
        return self

    def run_diis(self, se, diis=None):
        if diis is None:
            return se

        nmom = self.nmom

        se_occ = [x.get_occupied() for x in se]
        se_vir = [x.get_virtual() for x in se]

        t_occ = [x.moment(range(2*nmom[1]+2)) for x in se_occ]
        t_vir = [x.moment(range(2*nmom[1]+2)) for x in se_vir]

        dat = np.array(np.concatenate([t_occ, t_vir], axis=0))
        dat = diis.update(dat)
        t_occ, t_vir = dat[:self.nkpts], dat[self.nkpts:]

        if nmom[0] is None:
            fock = [None for kx in range(self.nkpts)]
        else:
            fock = self.get_fock()

        se_out = []
        for kx in range(self.nkpts):
            e, v = dyson.kernel_se(
                    t_occ[kx], t_vir[kx], nmom[1], nmom[0], phys=fock[kx], chempot=se[kx].chempot,
            )

            se_kx = aux.SelfEnergy(e, v, chempot=se[kx].chempot)
            se_kx.remove_uncoupled(tol=self.weight_tol)
            se.append(se_kx)

        return se



if __name__ == '__main__':
    from pyscf.pbc import gto, scf, df, agf2

    cell = gto.Cell()
    cell.atom = 'He 0 0 0; He 1 0 1'
    cell.a = np.eye(3) * 3
    cell.basis = '6-31g'
    cell.verbose = 0
    cell.build()

    rhf = scf.KRHF(cell)
    rhf.kpts = cell.make_kpts([2,2,2])
    rhf.with_df = df.GDF(cell, rhf.kpts)
    rhf.exxdiv = None
    rhf.run()

    gf2 = agf2.KRAGF2(rhf)
    gf2.verbose = 3
    gf2.direct = True
    gf2.run()

    gf2 = KRAGF2(rhf)
    gf2.verbose = 3
    gf2.direct = True
    gf2.run()
