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
Gamma point Auxiliary second-order Green's function perturbation theory
'''

#TODO: currently only supports incore 4c ERIs
#TODO: how to handle exxdiv?

import numpy as np
from pyscf import lib
from pyscf.agf2 import ragf2, uagf2
from pyscf.pbc import tools
from pyscf.pbc.mp.mp2 import _gen_ao2mofn
from pyscf.pbc.cc.ccsd import _adjust_occ


class RAGF2(ragf2.RAGF2):
    def kernel(self, eri=None, gf=None, se=None, dump_chk=True):
        from pyscf.pbc.df.df_ao2mo import warn_pbc2d_eri
        warn_pbc2d_eri(self._scf)
        return ragf2.RAGF2.kernel(self, eri=eri, gf=gf, se=se, dump_chk=dump_chk)

    def ao2mo(self, mo_coeff=None):
        with lib.temporary_env(self._scf, exxdiv=None):
            eri = ragf2._make_mo_eris_incore(self, mo_coeff)

        madelung = tools.madelung(self._scf.cell, self._scf.kpt)
        self.mo_energy = _adjust_occ(self.mo_energy, eri.nocc, -madelung)

        return eri


#class UAGF2(uagf2.UAGF2):
#    def kernel(self, eri=None, gf=None, se=None, dump_chk=True):
#        from pyscf.pbc.df.df_ao2mo import warn_pbc2d_eri
#        warn_pbc2d_eri(self._scf)
#        return uagf2.UAGF2.kernel(self, eri=eri, gf=gf, se=se, dump_chk=dump_chk)
#
#    def ao2mo(self, mo_coeff=None):
#        ao2mofn = _gen_ao2mofn(self._scf)
#
#        with lib.temporary_env(self._scf, exxdiv=None):
#            eri = uagf2._make_mo_eris_incore(self, mo_coeff)
#
#        madelung = tools.madelung(self._scf.cell, self._scf.kpt)
#        eri.mo_energy = (_adjust_occ(eri.mo_energy[0], eri.nocc[0], -madelung),
#                         _adjust_occ(eri.mo_energy[0], eri.nocc[0], -madelung))
#
#        return eri
