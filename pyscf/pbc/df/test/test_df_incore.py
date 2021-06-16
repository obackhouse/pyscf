# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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

import unittest
import numpy
from pyscf import lib
import pyscf.pbc
from pyscf import ao2mo
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
from pyscf.pbc.df import df_incore
#from mpi4pyscf.pbc.df import df
pyscf.pbc.DEBUG = False

L = 5.
n = 11
cell = pgto.Cell()
cell.a = numpy.diag([L,L,L])
cell.mesh = numpy.array([n,n,n])

cell.atom = '''He    3.    2.       3.
               He    1.    1.       1.'''
cell.basis = 'ccpvdz'
cell.verbose = 0
cell.max_memory = 1000
cell.build(0,0)

mf0 = pscf.RHF(cell)
mf0.exxdiv = 'vcut_sph'


numpy.random.seed(1)
kpts = numpy.random.random((5,3))
kpts[0] = 0
kpts[3] = kpts[0]-kpts[1]+kpts[2]
kpts[4] *= 1e-5

kmdf = df_incore.IncoreDF(cell)
kmdf.linear_dep_threshold = 1e-7
kmdf.auxbasis = 'weigend'
kmdf.kpts = kpts

kmdf_ref = df_incore.GDF(cell)
kmdf_ref.linear_dep_threshold = 1e-7
kmdf_ref.auxbasis = 'weigend'
kmdf_ref.kpts = kpts


def tearDownModule():
    global cell, mf0, kmdf, kmdf_ref
    del cell, mf0, kmdf, kmdf_ref


class KnownValues(unittest.TestCase):
    def test_get_eri_gamma(self):
        odf = df_incore.IncoreDF(cell)
        odf.linear_dep_threshold = 1e-7
        odf.auxbasis = 'weigend'
        eri0000 = odf.get_eri()
        eri1111 = kmdf.get_eri((kpts[0],kpts[0],kpts[0],kpts[0]))
        eri4444 = kmdf.get_eri((kpts[4],kpts[4],kpts[4],kpts[4]))

        odf_ref = df_incore.GDF(cell)
        odf_ref.linear_dep_threshold = 1e-7
        odf_ref.auxbasis = 'weigend'
        eri0000_ref = odf_ref.get_eri()
        eri1111_ref = kmdf_ref.get_eri((kpts[0],kpts[0],kpts[0],kpts[0]))
        eri4444_ref = kmdf_ref.get_eri((kpts[4],kpts[4],kpts[4],kpts[4]))

        self.assertTrue(numpy.allclose(eri0000, eri0000_ref, atol=1e-7))
        self.assertTrue(numpy.allclose(eri1111, eri1111_ref, atol=1e-7))
        self.assertTrue(numpy.allclose(eri4444, eri4444_ref, atol=1e-7))

    def test_get_eri_1111(self):
        eri1111 = kmdf.get_eri((kpts[1],kpts[1],kpts[1],kpts[1]))
        eri1111_ref = kmdf_ref.get_eri((kpts[1],kpts[1],kpts[1],kpts[1]))
        check2 = kmdf.get_eri((kpts[1]+5e-8,kpts[1]+5e-8,kpts[1],kpts[1]))
        check2_ref = kmdf_ref.get_eri((kpts[1]+5e-8,kpts[1]+5e-8,kpts[1],kpts[1]))
        self.assertTrue(numpy.allclose(eri1111, check2, atol=1e-7))
        self.assertTrue(numpy.allclose(eri1111_ref, check2_ref, atol=1e-7))
        self.assertTrue(numpy.allclose(eri1111, eri1111_ref, atol=1e-7))
        self.assertTrue(numpy.allclose(check2, check2_ref, atol=1e-7))

    def test_get_eri_0011(self):
        eri0011 = kmdf.get_eri((kpts[0],kpts[0],kpts[1],kpts[1]))
        eri0011_ref = kmdf_ref.get_eri((kpts[0],kpts[0],kpts[1],kpts[1]))
        self.assertTrue(numpy.allclose(eri0011, eri0011_ref, atol=1e-7))

    def test_get_eri_0110(self):
        eri0110 = kmdf.get_eri((kpts[0],kpts[1],kpts[1],kpts[0]))
        eri0110_ref = kmdf_ref.get_eri((kpts[0],kpts[1],kpts[1],kpts[0]))
        check2 = kmdf.get_eri((kpts[0]+5e-8,kpts[1]+5e-8,kpts[1],kpts[0]))
        check2_ref = kmdf_ref.get_eri((kpts[0]+5e-8,kpts[1]+5e-8,kpts[1],kpts[0]))
        self.assertTrue(numpy.allclose(eri0110, check2, atol=1e-7))
        self.assertTrue(numpy.allclose(eri0110_ref, check2_ref, atol=1e-7))
        self.assertTrue(numpy.allclose(eri0110, eri0110_ref, atol=1e-7))
        self.assertTrue(numpy.allclose(check2, check2_ref, atol=1e-7))

    def test_get_eri_0123(self):
        eri0123 = kmdf.get_eri(kpts[:4])
        eri0123_ref = kmdf.get_eri(kpts[:4])
        self.assertTrue(numpy.allclose(eri0123, eri0123_ref, atol=1e-7))



if __name__ == '__main__':
    print("Full Tests for df_incore")
    unittest.main()

