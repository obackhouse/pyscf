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
from pyscf.pbc.lib.kpts_helper import get_kconserv
from pyscf.pbc.agf2 import kragf2_ao2mo
from pyscf import scf as mol_scf
from pyscf import lib, ao2mo


def get_cell_he():
    cell = gto.C(atom='He 1 0 1; He 0 0 1',
                 basis='6-31g',
                 a=np.eye(3)*3,
                 mesh=[10,10,10],
                 verbose=0)
    return cell


def get_krhf(cell, with_df, kpts):
    krhf = scf.KRHF(cell)
    krhf.with_df = with_df(cell)
    krhf.kpts = cell.make_kpts(kpts)
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

    #def _test_ao(self, rhf, make):
    #    gf2 = agf2.KRAGF2(rhf)
    #    eris = lambda x: None
    #    eris.kpts = gf2.kpts
    #    prec = 10

    #    size_7d = (gf2.nkpts,)*3 + (self.cell.nao,)*4
    #    eri_ref = gf2.with_df.ao2mo_7d([np.array([np.eye(self.cell.nao),]*gf2.nkpts)]*4)
    #    eri_ref = eri_ref.reshape(size_7d) / gf2.nkpts

    #    bra, ket = make(gf2, eris)
    #    eri_direct = lib.einsum('abQp,abcQq->abcpq', bra, ket).reshape(size_7d)
    #    self.assertAlmostEqual(np.max(np.absolute(eri_ref-eri_direct)), 0, prec)

    #def test_ao_fft(self):
    #    self._test_ao(self.rhf_fft, kragf2_ao2mo._make_ao_eris_direct_fftdf)

    #def test_ao_aft(self):
    #    self._test_ao(self.rhf_aft, kragf2_ao2mo._make_ao_eris_direct_aftdf)

    #def test_ao_gdf(self):
    #    self._test_ao(self.rhf_gdf, kragf2_ao2mo._make_ao_eris_direct_gdf)

    #def test_ao_mdf(self):
    #    self._test_ao(self.rhf_mdf, kragf2_ao2mo._make_ao_eris_direct_mdf)

    def _test_mo(self, rhf):
        rhf.run(max_cycles=1)
        gf2 = agf2.KRAGF2(rhf)
        prec = 8

        size_7d = (gf2.nkpts,)*3 + (gf2.nmo,)*4
        eri_ref = gf2.with_df.ao2mo_7d([np.array(rhf.mo_coeff),]*4)
        eri_ref = eri_ref.reshape(size_7d) / gf2.nkpts

        eri_incore = kragf2_ao2mo._make_mo_eris_incore(gf2).eri
        self.assertAlmostEqual(np.max(np.absolute(eri_ref-eri_incore)), 0, prec)

        if type(rhf.with_df) is df.GDF:
            qij = kragf2_ao2mo._make_mo_eris_direct(gf2).eri
            nmo = rhf.mo_occ[0].size
            nkpts = len(rhf.kpts)
            eri_direct = np.zeros((nkpts, nkpts, nkpts, nmo, nmo, nmo, nmo), dtype=np.complex128)
            kconserv = get_kconserv(rhf.cell, rhf.kpts)
            for i in range(nkpts):
                for j in range(nkpts):
                    for k in range(nkpts):
                        l = kconserv[i,j,k]
                        eri_direct[i,j,k] = np.dot(qij[i,j].T, qij[k,l]).reshape(nmo, nmo, nmo, nmo)
            self.assertAlmostEqual(np.max(np.absolute(eri_ref-eri_direct)), 0, prec)

    def test_mo_fft(self):
        self._test_mo(self.rhf_fft)

    def test_mo_aft(self):
        self._test_mo(self.rhf_aft)

    def test_mo_gdf(self):
        self._test_mo(self.rhf_gdf)

    def test_mo_mdf(self):
        self._test_mo(self.rhf_mdf)


if __name__ == '__main__':
    unittest.main()
