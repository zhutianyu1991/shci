from functools import reduce
import numpy
from pyscf import gto, fci, scf, ao2mo
from pyscf import tools
from pyscf import symm

mol = gto.M(
            atom = [['Be', 0, 0, 0]],
            basis = 'sto-6g',
            verbose = 5,
)
myhf = scf.RHF(mol)
myhf.kernel()

cisolver = fci.FCI(mol, myhf.mo_coeff)
print('E(FCI) = %.12f' % cisolver.kernel()[0])

#
# Example 1: Convert an SCF object to FCIDUMP
#
tools.fcidump.from_scf(myhf, 'FCIDUMP')
