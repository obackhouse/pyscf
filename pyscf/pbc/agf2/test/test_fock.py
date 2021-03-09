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

import unittest
import numpy as np
from pyscf.pbc import gto, scf, df, tools, agf2
from pyscf.pbc.agf2 import kragf2_ao2mo
from pyscf import scf as mol_scf
from pyscf import lib, ao2mo


def get_cell_he():
    cell = gto.C(atom='He 1 0 1; He 0 0 1',
                 basis='6-31g',
                 a=np.eye(3)*3,
                 mesh=[15,15,15],
                 verbose=0)
    return cell


def get_krhf(cell, with_df, kpts):
    krhf = scf.KRHF(cell)
    krhf.max_cycle = 1  # non-canonical
    krhf.with_df = with_df(cell)
    krhf.kpts = cell.make_kpts(kpts)
    krhf.exxdiv = None
    if type(krhf.with_df) in [df.DF, df.MDF]:
        df.DF.build(krhf.with_df)
    krhf._kmesh = kpts
    krhf._keys.add('_kmesh')
    return krhf


class KnownValues(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.cell = get_cell_he()
        self.rhf_fft = get_krhf(self.cell, df.FFTDF, [2,1,2])
        self.rhf_aft = get_krhf(self.cell, df.AFTDF, [1,1,2])
        self.rhf_gdf = get_krhf(self.cell, df.GDF,   [1,2,2])
        self.rhf_mdf = get_krhf(self.cell, df.MDF,   [1,1,2])

    @classmethod
    def tearDownClass(self):
        del self.cell, self.rhf_fft, self.rhf_aft, self.rhf_gdf, self.rhf_mdf

    def _test_fock(self, rhf):
        rhf.run()
        rhf.mo_coeff = [np.random.random(x.shape) + np.random.random(x.shape) * 1.0j for x in rhf.mo_coeff]
        gf2 = agf2.KRAGF2(rhf)
        prec = 10

        fock_ref = np.array([np.diag(x) for x in rhf.mo_energy])

        h1e = rhf.get_hcore()
        j_ref, k_ref = rhf.get_jk(h1e=h1e, dm_kpts=rhf.make_rdm1())
        fock_ref = rhf.get_fock(h1e=h1e, dm=rhf.make_rdm1())
        j_ref = lib.einsum('kpq,kpi,kqj->kij', j_ref, [x.conj() for x in rhf.mo_coeff], rhf.mo_coeff)
        k_ref = lib.einsum('kpq,kpi,kqj->kij', k_ref, [x.conj() for x in rhf.mo_coeff], rhf.mo_coeff)
        fock_ref = lib.einsum('kpq,kpi,kqj->kij', fock_ref, [x.conj() for x in rhf.mo_coeff], rhf.mo_coeff)

        eri_direct = gf2.ao2mo()
        rdm1 = [x.make_rdm1() for x in gf2.init_gf(eri_direct)]
        j_direct, k_direct = gf2.get_jk(eri_direct.eri, rdm1)
        fock_direct = gf2.get_fock(eri_direct, rdm1=rdm1)
        self.assertAlmostEqual(np.max(np.absolute(j_ref-j_direct)), 0, prec)
        self.assertAlmostEqual(np.max(np.absolute(k_ref-k_direct)), 0, prec)
        self.assertAlmostEqual(np.max(np.absolute(fock_ref-fock_direct)), 0, prec)

        gf2.direct = False
        eri_incore = gf2.ao2mo()
        rdm1 = [x.make_rdm1() for x in gf2.init_gf(eri_incore)]
        j_incore, k_incore = gf2.get_jk(eri_incore.eri, rdm1)
        fock_incore = gf2.get_fock(eri_incore, rdm1=rdm1)
        self.assertAlmostEqual(np.max(np.absolute(j_ref-j_incore)), 0, prec)
        self.assertAlmostEqual(np.max(np.absolute(k_ref-k_incore)), 0, prec)
        self.assertAlmostEqual(np.max(np.absolute(fock_ref-fock_incore)), 0, prec)

    def test_fock_fft(self):
        self._test_fock(self.rhf_fft)

    def test_fock_aft(self):
        self._test_fock(self.rhf_aft)

    def test_fock_gdf(self):
        self._test_fock(self.rhf_gdf)

    def test_fock_mdf(self):
        self._test_fock(self.rhf_mdf)

    def test_fock_exxdiv(self):
        rhf = self.rhf_fft
        rhf.exxdiv = 'ewald'
        self._test_fock(rhf)
        rhf.exxdiv = None


if __name__ == '__main__':
    unittest.main()
