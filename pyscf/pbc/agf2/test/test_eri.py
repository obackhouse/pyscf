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

    def test_ao_fft(self):
        rhf = self.rhf_fft
        gf2 = agf2.KRAGF2(rhf)
        eris = lambda x: None
        eris.kpts = gf2.kpts

        size_7d = (gf2.nkpts,)*3 + (self.cell.nao,)*4
        eri_ref = gf2.with_df.ao2mo_7d([np.array([np.eye(self.cell.nao),]*gf2.nkpts)]*4)
        eri_ref = eri_ref.reshape(size_7d) / gf2.nkpts

        bra, ket = kragf2_ao2mo._make_ao_eris_direct_fftdf(gf2, eris)
        eri_direct = lib.einsum('abQp,abcQq->abcpq', bra.conj(), ket).reshape(size_7d)
        self.assertAlmostEqual(np.max(np.absolute(eri_ref-eri_direct)), 0, 8)

    def test_ao_aft(self):
        rhf = self.rhf_aft
        gf2 = agf2.KRAGF2(rhf)
        eris = lambda x: None
        eris.kpts = gf2.kpts

        size_7d = (gf2.nkpts,)*3 + (self.cell.nao,)*4
        eri_ref = gf2.with_df.ao2mo_7d([np.array([np.eye(self.cell.nao),]*gf2.nkpts)]*4)
        eri_ref = eri_ref.reshape(size_7d) / gf2.nkpts

        bra, ket = kragf2_ao2mo._make_ao_eris_direct_aftdf(gf2, eris)
        eri_direct = lib.einsum('abQp,abcQq->abcpq', bra.conj(), ket).reshape(size_7d)
        self.assertAlmostEqual(np.max(np.absolute(eri_ref-eri_direct)), 0, 8)

    def test_ao_gdf(self):
        rhf = self.rhf_gdf
        gf2 = agf2.KRAGF2(rhf)
        eris = lambda x: None
        eris.kpts = gf2.kpts

        size_7d = (gf2.nkpts,)*3 + (self.cell.nao,)*4
        eri_ref = gf2.with_df.ao2mo_7d([np.array([np.eye(self.cell.nao),]*gf2.nkpts)]*4)
        eri_ref = eri_ref.reshape(size_7d) / gf2.nkpts

        bra, ket = kragf2_ao2mo._make_ao_eris_direct_gdf(gf2, eris)
        eri_direct = lib.einsum('abQp,abcQq->abcpq', bra.conj(), ket).reshape(size_7d)
        self.assertAlmostEqual(np.max(np.absolute(eri_ref-eri_direct)), 0, 8)

    def test_ao_mdf(self):
        rhf = self.rhf_mdf
        gf2 = agf2.KRAGF2(rhf)
        eris = lambda x: None
        eris.kpts = gf2.kpts

        size_7d = (gf2.nkpts,)*3 + (self.cell.nao,)*4
        eri_ref = gf2.with_df.ao2mo_7d([np.array([np.eye(self.cell.nao),]*gf2.nkpts)]*4)
        eri_ref = eri_ref.reshape(size_7d) / gf2.nkpts

        bra, ket = kragf2_ao2mo._make_ao_eris_direct_mdf(gf2, eris)
        eri_direct = lib.einsum('abQp,abcQq->abcpq', bra.conj(), ket).reshape(size_7d)
        self.assertAlmostEqual(np.max(np.absolute(eri_ref-eri_direct)), 0, 8)

    def test_mo_fft(self):
        rhf = self.rhf_fft
        rhf.run(max_cycles=1)
        gf2 = agf2.KRAGF2(rhf)

        size_7d = (gf2.nkpts,)*3 + (gf2.nmo,)*4
        eri_ref = gf2.with_df.ao2mo_7d([np.array(rhf.mo_coeff),]*4)
        eri_ref = eri_ref.reshape(size_7d) / gf2.nkpts

        eri_incore = kragf2_ao2mo._make_mo_eris_incore(gf2).eri
        self.assertAlmostEqual(np.max(np.absolute(eri_ref-eri_incore)), 0, 8)

        bra, ket = kragf2_ao2mo._make_mo_eris_direct(gf2).eri
        eri_direct = lib.einsum('abQp,abcQq->abcpq', bra.conj(), ket).reshape(size_7d)
        self.assertAlmostEqual(np.max(np.absolute(eri_ref-eri_direct)), 0, 8)

    def test_mo_aft(self):
        rhf = self.rhf_aft
        rhf.run(max_cycles=1)
        gf2 = agf2.KRAGF2(rhf)

        size_7d = (gf2.nkpts,)*3 + (gf2.nmo,)*4
        eri_ref = gf2.with_df.ao2mo_7d([np.array(rhf.mo_coeff),]*4)
        eri_ref = eri_ref.reshape(size_7d) / gf2.nkpts

        eri_incore = kragf2_ao2mo._make_mo_eris_incore(gf2).eri
        self.assertAlmostEqual(np.max(np.absolute(eri_ref-eri_incore)), 0, 8)

        bra, ket = kragf2_ao2mo._make_mo_eris_direct(gf2).eri
        eri_direct = lib.einsum('abQp,abcQq->abcpq', bra.conj(), ket).reshape(size_7d)
        self.assertAlmostEqual(np.max(np.absolute(eri_ref-eri_direct)), 0, 8)

    def test_mo_gdf(self):
        rhf = self.rhf_gdf
        rhf.run(max_cycles=1)
        gf2 = agf2.KRAGF2(rhf)

        size_7d = (gf2.nkpts,)*3 + (gf2.nmo,)*4
        eri_ref = gf2.with_df.ao2mo_7d([np.array(rhf.mo_coeff),]*4)
        eri_ref = eri_ref.reshape(size_7d) / gf2.nkpts

        eri_incore = kragf2_ao2mo._make_mo_eris_incore(gf2).eri
        self.assertAlmostEqual(np.max(np.absolute(eri_ref-eri_incore)), 0, 8)

        bra, ket = kragf2_ao2mo._make_mo_eris_direct(gf2).eri
        eri_direct = lib.einsum('abQp,abcQq->abcpq', bra.conj(), ket).reshape(size_7d)
        self.assertAlmostEqual(np.max(np.absolute(eri_ref-eri_direct)), 0, 8)

    def test_mo_mdf(self):
        rhf = self.rhf_mdf
        rhf.run(max_cycles=1)
        gf2 = agf2.KRAGF2(rhf)

        size_7d = (gf2.nkpts,)*3 + (gf2.nmo,)*4
        eri_ref = gf2.with_df.ao2mo_7d([np.array(rhf.mo_coeff),]*4)
        eri_ref = eri_ref.reshape(size_7d) / gf2.nkpts

        eri_incore = kragf2_ao2mo._make_mo_eris_incore(gf2).eri
        self.assertAlmostEqual(np.max(np.absolute(eri_ref-eri_incore)), 0, 8)

        bra, ket = kragf2_ao2mo._make_mo_eris_direct(gf2).eri
        eri_direct = lib.einsum('abQp,abcQq->abcpq', bra.conj(), ket).reshape(size_7d)
        self.assertAlmostEqual(np.max(np.absolute(eri_ref-eri_direct)), 0, 8)


if __name__ == '__main__':
    unittest.main()
