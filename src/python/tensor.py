from . import lightspeed as pls
import numpy
import re

# => Tensor <= #

@property
def _tensor_np(
    self,
    ):

    return numpy.asanyarray(self)
    # Nice idea, but leads to circular reference
    # if not hasattr(self,'_np_hook'):
    #     self._np_hook = numpy.asanyarray(self)
    # return self._np_hook

@staticmethod
def _tensor_zeros(
    shape,
    name='I',
    ):

    return Tensor(shape,name)

@staticmethod
def _tensor_zeros_like(
    other,
    name='I',
    ):
    
    return Tensor(other.shape,name)

@staticmethod
def _tensor_ones(
    shape,
    name='I',
    ):

    T = Tensor(shape,name)
    T[...] = 1.
    return T

@staticmethod
def _tensor_ones_like(
    other,
    name='I',
    ):
    
    T = Tensor(other.shape,name)
    T[...] = 1.
    return T

@staticmethod
def _tensor_array(
    arr,
    name='I',
    ):

    return Tensor.tensor_copy(numpy.array(arr),name='I')

@staticmethod
def _tensor_tensor_copy(
    other,
    name='I',
    ):

    T = Tensor(other.shape,name)
    T.np[...] = other
    return T

@staticmethod
def _tensor_numpy_copy(
    other,    
    ):

    T = numpy.zeros_like(other)
    T[...] = other
    return T

@staticmethod
def _tensor_chain(
    As,
    trans,
    C=None,
    alpha=1.0,
    beta=0.0,
    ):

    A2s = pls.TensorVec()
    for A in As:
        A2s.append(A)

    if C is None:
        return Tensor.py_chain(A2s,trans)
    else:
        return Tensor.py_chain(A2s,trans,C,alpha,beta)

@staticmethod
def _tensor_permute(
    permstr,
    A,
    C=None,
    alpha=1.0,
    beta=0.0,
    ):

    Astr, Cstr = re.split(r'->', permstr)
    if C is None:
        return Tensor.py_permute([x for x in Astr],[x for x in Cstr],A)
    else:
        return Tensor.py_permute([x for x in Astr],[x for x in Cstr],A,C,alpha,beta)

@staticmethod
def _tensor_einsum(
    einstr,
    A,
    B,
    C=None,
    alpha=1.0,
    beta=0.0,
    ):

    Astr, Bstr, Cstr = re.split(r',|->', einstr)
    if C is None:
        return Tensor.py_einsum([x for x in Astr],[x for x in Bstr],[x for x in Cstr],A,B)
    else:
        return Tensor.py_einsum([x for x in Astr],[x for x in Bstr],[x for x in Cstr],A,B,C,alpha,beta)

@staticmethod
def _tensor_eigh(
    T,      # Target Matrix
    S=None, # Metric Matrix
    ):

    T.square_error()
    U = Tensor.zeros_like(T, name='U') 
    t = Tensor((T.shape[0],), 't')
    if S is None:
        Tensor.syev(T,U,t)
    else:
        Tensor.generalized_syev(T,S,U,t)
    return t, U

@staticmethod
def _tensor_svd(
    T,
    full_matrices=False,
    ):

    T.ndim_error(2)
    m = T.shape[0] 
    n = T.shape[1]
    k = min(m,n)
    U = Tensor.zeros((m,k) if not full_matrices else (m,m))
    V = Tensor.zeros((k,n) if not full_matrices else (n,n))
    s = Tensor.zeros((k,)) 
    Tensor.gesvd(T,U,s,V,full_matrices)
    return U,s,V

Tensor = pls.Tensor
Tensor.np = _tensor_np
Tensor.zeros = _tensor_zeros
Tensor.zeros_like = _tensor_zeros_like
Tensor.ones = _tensor_ones
Tensor.ones_like = _tensor_ones_like
Tensor.array = _tensor_array
Tensor.tensor_copy = _tensor_tensor_copy
Tensor.numpy_copy = _tensor_numpy_copy
Tensor.chain = _tensor_chain
Tensor.permute = _tensor_permute
Tensor.einsum = _tensor_einsum
Tensor.eigh = _tensor_eigh
Tensor.svd = _tensor_svd

# => Patch to look like ndarray <= #

def _tensor_getitem(self, ind,):
    return self.np[ind]

def _tensor_setitem(self, ind, y,):
    self.np[ind] = y

Tensor.__getitem__ = _tensor_getitem
Tensor.__setitem__ = _tensor_setitem

