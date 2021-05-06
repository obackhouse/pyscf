'''
ADC(2) for periodic solids
'''

import time
import h5py
import numpy as np
from pyscf import __config__
from pyscf import lib, ao2mo
from pyscf.lib import logger
from pyscf.pbc.df import df
from pyscf.pbc.lib import kpts_helper
from pyscf.agf2 import mpi_helper


def kernel(adc, eri=None, verbose=None, nroots=5):

    log = logger.new_logger(adc, verbose)
    cput1 = cput0 = (time.clock(), time.time())
    name = adc.__class__.__name__

    if eri is None: eri = adc.ao2mo()
    if verbose is None: verbose = adc.verbose

    kpts = eri.kpts
    nkpts = len(kpts)
    real_system = kpts_helper.gamma_point(kpts)
    which = adc.which
    e, v, conv = [], [], []

    for ki, kpti in enumerate(kpts):
        m11 = adc.get_singles(ki, eri=eri)
        matvec, diag = adc.get_matvec(ki, m11=m11, eri=eri)
        matvecs = lambda xs: [matvec(x) for x in xs]

        guesses = adc.get_guesses(
                ki,
                diag,
                nroots=nroots,
                koopmans=adc.koopmans,
        )

        pick = adc.get_picker(
                koopmans=adc.koopmans,
                real_system=real_system,
                guess=guesses,
        )

        convk, ek, vk = lib.davidson_nosym1(
                matvecs,
                guesses,
                diag,
                nroots=nroots,
                pick=pick,
                tol=adc.conv_tol,
                tol_residual=adc.tol_residual,
                max_cycle=adc.max_cycle,
                max_space=adc.max_space,
                verbose=log,
        )

        if which == 'ip':
            ek *= -1

        e.append(ek)
        v.append(vk)
        conv.append(convk)

        mpi_helper.barrier()
        if mpi_helper.rank == 0:
            log.info('kpt %d (%.6f %.6f %.6f)', ki, *kpti)
            for n in range(nroots):
                log.info(
                        '  root %d   %s = %.15g  conv = %s',
                        n, which.upper(), ek[n], convk[n],
                )

        if adc.gamma_only:
            break

        if mpi_helper.rank == 0:
            cput1 = log.timer(
                    '%s-KRADC(2) kpt %d' %
                    (which.upper(), ki), *cput1,
            )

    if mpi_helper.rank == 0:
        log.timer('%s-KRADC(2)' % which.upper(), *cput0)

    return e, v


def get_singles(adc, ki, eri=None):
    ''' Get the 1h or 1p space for a k-point
    '''

    if eri is None:
        eri = adc.ao2mo()

    kpts = adc.kpts
    nkpts = len(kpts)
    nocc, nvir = adc.nocc, adc.nvir

    m11 = np.zeros((nocc[ki], nocc[ki]), dtype=eri.dtype)

    for kjk in mpi_helper.nrange(nkpts**2):
        kj, kk = divmod(kjk, nkpts)
        kl = adc.kconserv[ki,kj,kk]

        iajb = lib.dot(
                eri.Lov[ki,kj].T,
                eri.Lov[kk,kl],
        )
        ibja = lib.dot(
                eri.Lov[ki,kl].T,
                eri.Lov[kk,kj],
        )

        iajb = iajb.reshape(nocc[ki], nvir[kj], nocc[kk], nvir[kl])
        ibja = ibja.reshape(nocc[ki], nvir[kl], nocc[kk], nvir[kj])

        t2 = iajb / lib.direct_sum('i-a+j-b->iajb',
                eri.eo[ki],
                eri.ev[kj],
                eri.eo[kk],
                eri.ev[kl],
        )

        v = 2.0 * iajb - ibja.swapaxes(1,3)

        m11 = lib.dot(
                t2.reshape(nocc[ki], -1),
                v.reshape(nocc[ki], -1).T.conj(),
                alpha=0.5,
                beta=1.0,
                c=m11,
        )

    mpi_helper.barrier()
    mpi_helper.allreduce_safe_inplace(m11)
    mpi_helper.barrier()

    m11 += m11.T.conj()
    m11 += np.diag(eri.eo[ki])

    return m11


def get_matvec(adc, ki, m11=None, eri=None):
    ''' 
    Get the matrix-vector operation and the diagonal of the secular
    matrix for a k-point.
    '''

    if eri is None:
        eri = adc.ao2mo()
    if m11 is None:
        m11 = adc.get_singles(eri=eri)

    kpts = adc.kpts
    nkpts = len(kpts)
    nocc, nvir = adc.nocc, adc.nvir
    ni = nocc[ki]

    ejka = np.empty((nkpts, nkpts), dtype=object)
    for kjk in mpi_helper.nrange(nkpts**2):
        kj, kk = divmod(kjk, nkpts)
        kl = adc.kconserv[ki,kj,kk]
        ejka[kj,kk] = lib.direct_sum('j+k-a->jka',
                eri.eo[kj],
                eri.eo[kk],
                eri.ev[kl],
        )

    for kjk in range(nkpts**2):
        kj, kk = divmod(kjk, nkpts)
        root = kjk % mpi_helper.size
        mpi_helper.barrier()
        ejka[kj,kk] = mpi_helper.bcast(ejka[kj,kk], root=root)

    def matvec(y):
        y = np.asarray(y, order='C', dtype=eri.dtype)
        r = np.zeros_like(y)

        yi, yjka = adc.unpack_vector(y, ki)
        ri, rjka = adc.unpack_vector(r, ki)

        for kjk in mpi_helper.nrange(nkpts**2):
            kj, kk = divmod(kjk, nkpts)
            kl = adc.kconserv[ki,kj,kk]
            nj, nk, na = nocc[kj], nocc[kk], nvir[kl]

            ijka = lib.dot(
                    eri.Loo[ki,kj].T,
                    eri.Lov[kk,kl],
            )
            ikja = lib.dot(
                    eri.Loo[ki,kk].T,
                    eri.Lov[kj,kl],
            )

            ijka = ijka.reshape(ni, nj, nk, na)
            ikja = ijka.reshape(ni, nk, nj, na)

            v = 2.0 * ijka - ikja.swapaxes(1,2)
            v = np.conj(v)

            ri += np.dot(
                    ijka.reshape(ni, -1),
                    yjka[kj,kk].ravel(),
            )

            rjka[kj,kk] += np.dot(
                    yi.ravel(),
                    v.reshape(ni, -1),
            ).reshape(nj, nk, na)

            rjka[kj,kk] += ejka[kj,kk] * yjka[kj,kk]


        mpi_helper.barrier()
        mpi_helper.allreduce_safe_inplace(ri)

        for kjk in range(nkpts**2):
            kj, kk = divmod(kjk, nkpts)
            root = kjk % mpi_helper.size
            mpi_helper.barrier()
            rjka[kj,kk] = mpi_helper.bcast(rjka[kj,kk], root=root)

        ri += np.dot(m11, yi)

        return adc.pack_vector(ri, rjka)

    diag = adc.pack_vector(np.diag(m11), ejka)

    return matvec, diag


def get_guesses(adc, ki, diag, nroots=5, koopmans=False):
    ''' Get guesses for Davidson solver for a k-point.
    '''

    nocc, nvir = adc.nocc, adc.nvir

    guesses = np.zeros((nroots, diag.size), dtype=diag.dtype)

    if koopmans:
        arg = np.argsort(np.absolute(diag[:nocc[ki]]))
        nroots = min(nroots, nocc[ki])
    else:
        arg = np.argsort(np.absolute(diag))

    for root, guess in enumerate(arg[:nroots]):
        guesses[root, guess] = 1.0

    return list(guesses)


def get_picker(adc, koopmans=False, real_system=False, guess=None):
    ''' Get the eigenvalue picker.
    '''

    if not koopmans:
        def pick(w, v, nroots, envs):
            w, v, idx = lib.linalg_helper.pick_real_eigs(w, v, nroots, envs)
            mask = np.argsort(np.absolute(w))
            return w[mask], v[:,mask], idx
    else:
        assert guess is not None
        def pick(w, v, nroots, envs):
            x0 = lib.linalg_helper._gen_x0(envs['v'], envs['xs'])
            s = np.dot(np.asarray(guess).conj(), np.asarray(x0).T)
            s = lib.einsum('pi,pi->i', np.conj(s), s)
            idx = np.argsort(-s)[:nroots]
            return lib.linalg_helper._eigs_cmplx2real(w, v, idx, real_system)

    return pick


class KRADC2(lib.StreamObject):
    def __init__(self, mf, which='ip', mo_energy=None, mo_coeff=None, mo_occ=None):
        if mo_energy is None: mo_energy = mf.mo_energy
        if mo_coeff is None:  mo_coeff  = mf.mo_coeff
        if mo_occ is None:    mo_occ    = mf.mo_occ

        if not isinstance(mf.with_df, df.DF):
            raise NotImplementedError(repr(mf.with_df.__class__))

        self.cell = mf.cell
        self._scf = mf
        self.kpts = mf.kpts
        self.kconserv = kpts_helper.get_kconserv(self.cell, self.kpts)
        self.verbose = self.cell.verbose
        self.stdout = self.cell.stdout
        self.max_memory = mf.max_memory
        self.with_df = mf.with_df
        self.which = which

        self.conv_tol = getattr(__config__, 'pbc_adc_conv_tol', 1e-12)
        self.max_space = getattr(__config__, 'pbc_adc_max_space', 12)
        self.max_cycle = getattr(__config__, 'pbc_adc_max_cycle', 50)
        self.tol_residual = getattr(__config__, 'pbc_adc_tol_res', 1e-6)
        self.koopmans = getattr(__config__, 'pbc_adc_koopmans', False)
        self.gamma_only = getattr(__config__, 'pbc_adc_gamma_only', False)

        self.e = None
        self.v = None
        self.mo_energy = mo_energy
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self._nmo = [x.size for x in mo_occ]
        self._nocc = [np.sum(x > 0) for x in mo_occ]
        self._nvir = [np.sum(x == 0) for x in mo_occ]
        self._keys = set(self.__dict__.keys())

    get_singles = get_singles
    get_matvec = get_matvec
    get_guesses = get_guesses
    get_picker = get_picker

    def ao2mo(self):
        ''' Get the electronic repulsion integrals
        '''

        return _make_mo_eris(self, which=self.which)

    def unpack_vector(self, v, ki):
        kpts = self.kpts
        nkpts = len(kpts)
        nocc, nvir = self.nocc, self.nvir
        
        vi = v[:nocc[ki]]
        vija = np.empty((nkpts, nkpts), dtype=object)

        p1 = nocc[ki]
        for kj in range(nkpts):
            for kk in range(nkpts):
                kl = self.kconserv[ki,kj,kk]
                ni, nj, na = nocc[kj], nocc[kk], nvir[kl]
                p0, p1 = p1, p1 + ni*nj*na
                vija[kj,kk] = v[p0:p1].reshape(ni, nj, na)

        return vi, vija

    def pack_vector(self, vi, vija):
        kpts = self.kpts
        nkpts = len(kpts)

        v = [vi]

        for kj in range(nkpts):
            for kk in range(nkpts):
                v.append(vija[kj,kk].ravel())

        return np.concatenate(v)

    def kernel(self, *args, **kwargs):
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()

        self.e, self.v = kernel(self, *args, **kwargs)

        self._finalize()

        return self.e, self.v

    def dump_flags(self, verbose=None):
        if mpi_helper.rank != 0:
            return self

        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('conv_tol = %g', self.conv_tol)
        log.info('tol_residual = %g', self.tol_residual)
        log.info('max_cycle = %d', self.max_cycle)
        log.info('max_space = %d', self.max_space)
        log.info('which = %s', self.which)
        log.info('koopmans = %s', self.koopmans)
        log.info('gamma_only = %s', self.gamma_only)
        log.info('max_memory %d MB (Current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        return self

    def _finalize(self):
        if mpi_helper.rank != 0:
            return self

        for n in range(len(self.kpts)):
            logger.info(
                    self,
                    'kpt %d   %s = %.15g',
                    n, self.which.upper(), self.e[n][0],
            )

            if self.gamma_only:
                break

        return self


    @property
    def nmo(self):
        return self._nmo

    @property
    def nocc(self):
        return self._nocc

    @property
    def nvir(self):
        return self._nvir


def _make_mo_eris(adc, which='ip'):
    ''' Make the 3c MO ERIs
    '''

    log = logger.Logger(adc.stdout, adc.verbose)
    cput0 = (time.clock(), time.time())

    if adc._scf.with_df._cderi is None:
        adc._scf.with_df.build()

    if adc._scf.cell.dimension == 2:
        raise NotImplementedError

    kpts = adc.kpts
    nkpts = len(kpts)
    naux = adc._scf.with_df.auxcell.nao_nr()
    nao = adc._scf.cell.nao
    mo_energy = adc.mo_energy
    mo_coeff = adc.mo_coeff
    mo_occ = adc.mo_occ

    if kpts_helper.gamma_point(kpts):
        dtype = np.float64
    else:
        dtype = np.complex128
    dtype = np.result_type(dtype, *[x.dtype for x in mo_coeff])

    oa = [x > 0 for x in mo_occ]
    va = [x == 0 for x in mo_occ]
    if which == 'ea':
        oa, va = va, oa

    ei = [e[o] for e,o in zip(mo_energy, oa)]
    ea = [e[o] for e,o in zip(mo_energy, va)]
    ci = [c[:,o] for c,o in zip(mo_coeff, oa)]
    ca = [c[:,o] for c,o in zip(mo_coeff, va)]

    eris = _ChemistsERIs(cell)
    eris._common_init_(adc)

    Lov = np.empty((nkpts, nkpts), dtype=object)
    Loo = np.empty((nkpts, nkpts), dtype=object)

    with h5py.File(adc._scf.with_df._cderi, 'r') as f:
        kptij_lst = f['j3c-kptij'][:]

    for kij in mpi_helper.nrange(nkpts**2):
        ki, kj = divmod(kij, nkpts)
        kpti, kptj = kpts[ki], kpts[kj]
        kpti_kptj = np.array((kpti, kptj))
        with h5py.File(adc._scf.with_df._cderi, 'r') as f:
            Lpq = np.asarray(df._getitem(f, 'j3c', kpti_kptj, kptij_lst))
        Lpq /= np.sqrt(nkpts)

        cov = np.asarray(
                np.hstack((ci[ki], ca[kj])), 
                dtype=dtype, 
                order='F',
        )
        coo = np.asarray(
                np.hstack((ci[ki], ci[kj])),
                dtype=dtype,
                order='F',
        )

        noi, noj = ei[ki].size, ei[kj].size
        nvi, nvj = ea[ki].size, ea[kj].size

        if dtype == np.float64:
            Lov[ki, kj] = ao2mo._ao2mo.nr_e2(
                    Lpq,
                    cov,
                    (0, noi, noi, noi+nvj),
                    aosym='s2',
            ).reshape(-1, noi*nvj)

            Loo[ki, kj] = ao2mo._ao2mo.nr_e2(
                    Lpq,
                    coo,
                    (0, noi, noi, noi+noj),
                    aosym='s2',
            ).reshape(-1, noi*noj)

        else:
            if Lpq[0].size != nao**2:
                Lpq = lib.unpack_tril(Lpq).astype(np.complex128)

            Lov[ki, kj] = ao2mo._ao2mo.r_e2(
                    Lpq,
                    cov,
                    (0, noi, noi, noi+nvj),
                    [], None,
            ).reshape(-1, noi*nvj)

            Loo[ki, kj] = ao2mo._ao2mo.r_e2(
                    Lpq,
                    coo,
                    (0, noi, noi, noi+noj),
                    [], None,
            ).reshape(-1, noi*noj)

    for kjk in range(nkpts**2):
        kj, kk = divmod(kjk, nkpts)
        root = kjk % mpi_helper.size
        mpi_helper.barrier()
        Loo[kj,kk] = mpi_helper.bcast(Loo[kj,kk], root=root)
        Lov[kj,kk] = mpi_helper.bcast(Lov[kj,kk], root=root)

    if mpi_helper.rank == 0:
        log.timer_debug1('transformation of integrals', *cput0)

    eris.eo = np.empty((nkpts,), dtype=object)
    eris.ev = np.empty((nkpts,), dtype=object)
    eris.eo[:] = ei
    eris.ev[:] = ea
    eris.Loo = Loo
    eris.Lov = Lov

    return eris


class _ChemistsERIs:
    def __init__(self, cell=None):
        self.cell = cell
        self.kpts = None

        self.Loo = None
        self.Lov = None
        self.eo = None
        self.ev = None

    def _common_init_(self, adc):
        self.cell = adc.cell
        self.kpts = adc.kpts

    def _assert_built(self):
        if getattr(self, 'Lov', None) is None:
            raise AttributeError

    @property
    def dtype(self):
        self._assert_built()

        if self.Lov.dtype == object:
            dtype = np.result_type(
                    *[x.dtype for x in self.Lov.ravel()],
                    *[x.dtype for x in self.Loo.ravel()],
            )
        else:
            dtype = np.result_type(
                    self.Lov.dtype,
                    self.Loo.dtype,
            )

        return dtype


if __name__ == '__main__':
    from pyscf.pbc import gto, scf

    cell = gto.Cell()
    if 0:
        cell.atom = 'C 0 0 0; C 1 0 1'
        cell.basis = 'gth-szv'
        cell.pseudo = 'gth-pade'
        cell.a = (np.ones((3,3)) - np.eye(3)) * 3
    else:
        cell.atom = 'He 0 0 0; He 1 1 1'
        cell.basis = '6-31g'
        cell.a = np.eye(3) * 3
    cell.verbose = 0
    cell.build()

    kmesh = [2,2,1]
    kpts = cell.make_kpts(kmesh)

    rhf = scf.KRHF(cell)
    rhf.with_df = df.DF(cell, kpts)
    rhf.with_df.build()
    rhf.run()

    adc2 = KRADC2(rhf)
    adc2.verbose = 0
    ip = adc2.kernel()[0][0][0]
    adc2.which = 'ea'
    ea = adc2.kernel()[0][0][0]
    mpi_helper.barrier()
    if mpi_helper.rank == 0:
        print(ip, ea, ip+ea)


    from pyscf import adc
    from pyscf.pbc.tools.pbc import super_cell

    scell = super_cell(cell, kmesh)
    rhf = scf.RHF(scell)
    rhf = rhf.density_fit()
    rhf.run()

    adc2 = adc.ADC(rhf)
    ip = adc2.kernel()[0][0]
    adc2.method_type = 'ea'
    ea = adc2.kernel()[0][0]
    mpi_helper.barrier()
    if mpi_helper.rank == 0:
        print(ip, ea, ip+ea)


