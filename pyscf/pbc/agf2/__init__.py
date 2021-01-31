from pyscf.pbc import scf
from pyscf.pbc.agf2 import kragf2, agf2
from pyscf.pbc.agf2 import kragf2_ao2mo
from pyscf.pbc.agf2 import _kagf2

def RAGF2(mf, frozen=None, mo_energy=None, mo_coeff=None, mo_occ=None):
    mf = scf.addons.convert_to_rhf(mf)
    return agf2.RAGF2(mf, frozen, mo_energy, mo_coeff, mo_occ)

def UAGF2(mf, frozen=None, mo_energy=None, mo_coeff=None, mo_occ=None):
    mf = scf.addons.convert_to_uhf(mf)
    return agf2.UAGF2(mf, frozen, mo_energy, mo_coeff, mo_occ)

def AGF2(mf, frozen=None, mo_energy=None, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.uhf.UHF):
        return UAGF2(mf, nmom, frozen, mo_energy, mo_coeff, mo_occ)

    elif isinstance(mf, scf.rohf.ROHF):
        lib.logger.warn(mf, 'RAGF2 method does not support ROHF reference. '
                            'Converting to UHF and using UAGF2.')
        return UAGF2(mf, nmom, frozen, mo_energy, mo_coeff, mo_occ)

    elif isinstance(mf, scf.hf.RHF):
        return RAGF2(mf, nmom, frozen, mo_energy, mo_coeff, mo_occ)

    else:
        raise RuntimeError('AGF2 code only supports RHF, ROHF and UHF references')

def KRAGF2(mf, frozen=None, mo_energy=None, mo_coeff=None, mo_occ=None):
    mf = scf.addons.convert_to_rhf(mf)
    if not isinstance(mf, scf.khf.KRHF):
        mf = scf.addons.convert_to_rhf(mf)
    return kragf2.KRAGF2(mf, frozen, mo_energy, mo_coeff, mo_occ)

def KAGF2(mf, frozen=None, mo_energy=None, mo_coeff=None, mo_occ=None):
    if isinstance(mf, scf.kuhf.KUHF):
        raise NotImplementedError

    elif isinstance(mf, scf.krohf.KROHF):
        raise NotImplementedError

    elif isinstance(mf, scf.khf.KRHF):
        return KRAGF2(mf, nmom, frozen, mo_energy, mo_coeff, mo_occ)

    else:
        raise RuntimeError('AGF2 code only supports RHF, ROHF and UHF references')

