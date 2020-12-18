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
#         Alejandro Santana-Bonilla <alejandro.santana_bonilla@kcl.ac.uk>
#         George H. Booth <george.booth@kcl.ac.uk>
#

import numpy as np
import ctypes
from pyscf import lib
from pyscf.agf2 import mpi_helper


def build_mats_kragf2_incore(qija, qjia, ei, ej, ea, os_factor=1.0, ss_factor=1.0):
    return _build_mats_kragf2_incore(qija, qjia, ei, ej, ea, os_factor=os_factor, ss_factor=ss_factor)


def build_mats_kragf2_direct(qxi, qja, qxj, qia, ei, ej, ea, os_factor=1.0, ss_factor=1.0):
    return _build_mats_kragf2_direct(qxi, qja, qxj, qia, ei, ej, ea, os_factor=os_factor, ss_factor=ss_factor)


def _build_mats_kragf2_incore(qija, qjia, ei, ej, ea, os_factor=1.0, ss_factor=1.0):
    # Python version

    nmo, nocci, noccj, nvir = qija.shape
    assert qjia.shape == (nmo, noccj, nocci, nvir)

    vv = np.zeros((nmo, nmo), dtype=qija.dtype)
    vev = np.zeros((nmo, nmo), dtype=qjia.dtype)

    fpos = os_factor + ss_factor
    fneg = -ss_factor

    eja = lib.direct_sum('j,a->ja', ej, -ea).ravel()

    for i in range(nocci):
        xija = np.array(qija[:,i].reshape(nmo, -1))
        xjia = np.array(qjia[:,:,i].reshape(nmo, -1))
        xjia = fpos * xija + fneg * xjia

        vv = lib.dot(xija, xjia.T.conj(), beta=1, c=vv)

        eija = eja + ei[i]
        exija = xija * eija[None]

        vev = lib.dot(exija, xjia.T.conj(), beta=1, c=vev)

    return vv, vev


def _build_mats_kragf2_direct(qxi, qja, qxj, qia, ei, ej, ea, os_factor=1.0, ss_factor=1.0):
    # Python version

    naux = qxi.shape[0]
    nocci = ei.size
    noccj = ej.size
    nvir = ea.size
    nmo = qxi.size // (naux*nocci)
    assert qxi.size == (naux * nmo * nocci)
    assert qxj.size == (naux * nmo * noccj)
    assert qja.size == (naux * noccj * nvir)
    assert qia.size == (naux * nocci * nvir)

    vv = np.zeros((nmo, nmo), dtype=qxi.dtype)
    vev = np.zeros((nmo, nmo), dtype=qxi.dtype)

    fpos = os_factor + ss_factor
    fneg = -ss_factor

    eja = lib.direct_sum('j,a->ja', ej, -ea).ravel()

    #TODO: mesh can be large - block the dot products
    for i in range(nocci):
        qx = np.array(qxi.reshape(naux, nmo, nocci)[:,:,i])
        xija = lib.dot(qx.T.conj(), qja)
        xjia = lib.dot(qxj.T.conj(), np.array(qia[:,i*nvir:(i+1)*nvir]))
        xjia = xjia.reshape(nmo, noccj*nvir)
        xjia = fpos * xija + fneg * xjia

        vv = lib.dot(xija, xjia.T.conj(), beta=1, c=vv)

        eija = eja + ei[i]
        exija = xija * eija[None]

        vev = lib.dot(exija, xjia.T.conj(), beta=1, c=vev)

    return vv, vev
