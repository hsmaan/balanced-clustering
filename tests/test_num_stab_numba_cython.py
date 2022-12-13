import pyximport
import numpy

pyximport.install(setup_args={"include_dirs": numpy.get_include()}, reload_support=True)
from ._emi_cython import expected_mutual_information