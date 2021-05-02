from pyscf import scf, lib
from pyscf.lib import param, logger
import numdifftools as nd
import numpy as np
import time
import sys


def kernel(method, atmlst=None, ndargs={}):
    ''' Get the gradient from the PySCF object.

    Args:
        method : any PySCF method
            Must have an `as_scanner` method.
        atmlst : list of int, optional
            Mask of atoms to compute derivatives for
        ndargs : dict, optional
            Keyword arguments for numdifftools
    '''

    cput0 = cput1 = (time.clock(), time.time())
    log = logger.Logger(method.stdout, method.verbose)

    log.debug('Computing numerical gradients for %s', method.__class__.__name__)

    mol = method.mol
    atoms = [mol.atom_symbol(k) for k in range(mol.natm)]
    x0 = np.array(mol.atom_coords()) * param.BOHR
    if atmlst is None:
        atmlst = range(mol.natm)

    with lib.temporary_env(method, verbose=0):
        scanner = method.as_scanner()

    def _get_obj(atm, x):
        def _obj(dx):
            coords = x0.copy()
            coords[atm,x] += dx * param.BOHR
            return scanner(list(zip(atoms, coords)))
        return _obj

    dx = np.zeros((len(atmlst), 3))

    for k, ia in enumerate(atmlst):
        for x in range(3):
            f = _get_obj(k, x)
            deriv = nd.Derivative(f, **ndargs)
            dx[k, x] = deriv([0.0])

            cput1 = log.timer_debug1('gradients for atom %d (%s)'
                                     % (k, 'xyz'[x]), *cput1)

    if log.verbose >= logger.DEBUG:
        from pyscf.grad.rhf import _write
        _write(log, mol, dx, atmlst)

    return dx


if __name__ == '__main__':
    from pyscf import gto, scf, mp, cc

    mol = gto.M(atom='N 0 0 0; N 0 0 1', basis='6-31g', verbose=0)
    rhf = scf.RHF(mol).run()
    mp2 = mp.MP2(rhf).run()
    ccsd = cc.CCSD(rhf).run()

    ndargs = {
        'order': 3,
    }

    a = rhf.nuc_grad_method().kernel()
    b = kernel(rhf, ndargs=ndargs)
    print('%6s %6s %8.3g' % ('RHF', np.allclose(a, b), np.linalg.norm(a-b)))

    a = mp2.nuc_grad_method().kernel()
    b = kernel(mp2, ndargs=ndargs)
    print('%6s %6s %8.3g' % ('MP2', np.allclose(a, b), np.linalg.norm(a-b)))

    a = ccsd.nuc_grad_method().kernel()
    b = kernel(ccsd, ndargs=ndargs)
    print('%6s %6s %8.3g' % ('CCSD', np.allclose(a, b), np.linalg.norm(a-b)))



    
