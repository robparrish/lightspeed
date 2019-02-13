from . import lightspeed as pls

Storage = pls.Storage
DIIS = pls.DIIS
Davidson = pls.Davidson

# => Solver <= #

@staticmethod
def _to_storage_helper(
    state,
    use_disk=False,
    ):

    state2 = None
    if isinstance(state,pls.Storage):
        state2 = state
    elif isinstance(state,pls.Tensor):
        state2 = pls.Storage.from_tensor(state,use_disk)
    elif isinstance(state, list):
        state2 = pls.Storage.from_tensor_vec(state,use_disk)
    else:
        raise RuntimeError('Solver: Cannot determine which type to wrap to Storage')
    
    return state2
    
@staticmethod
def _from_storage_helper(
    ext2,   
    state):

    ext = None
    if isinstance(state,pls.Storage):
        ext = ext2
    elif isinstance(state,pls.Tensor):
        ext = pls.Tensor.zeros_like(state)
        Storage.to_tensor(ext2,ext)
    elif isinstance(state, list):
        ext = [pls.Tensor.zeros_like(x) for x in state]
        Storage.to_tensor_vec(ext2,ext)
    else:
        raise RuntimeError('Solver: Cannot determine which type to wrap to Storage')
    
    return ext

Storage.to_storage = _to_storage_helper
Storage.from_storage = _from_storage_helper

# => DIIS <= #

def _diis_iterate(
    self,
    state,
    error,
    use_disk=False,
    ):

    state2 = Storage.to_storage(state,use_disk)
    error2 = Storage.to_storage(error,use_disk)
    ext2 = self.py_iterate(state2,error2)
    return Storage.from_storage(ext2,state)

DIIS.iterate = _diis_iterate

# => Davidson <= #

def _davidson_add_vectors(
    self,
    b,
    Ab,
    Sb=None, 
    use_disk=False,
    ):

    b2 = [Storage.to_storage(x,use_disk) for x in b]
    Ab2 = [Storage.to_storage(x,use_disk) for x in Ab]
    Sb2 = [Storage.to_storage(x,use_disk) for x in Sb] if Sb else b2
    self.py_add_vectors(b2,Ab2,Sb2)
    gs = list([Storage.from_storage(x,b[0]) for x in self.gs])
    hs = list(self.hs)
    return gs, hs

def _davidson_add_preconditioned(
    self,
    d,
    use_disk=False,
    ):

    d2 = [Storage.to_storage(x,use_disk) for x in d]
    self.py_add_preconditioned(d2)
    cs = list([Storage.from_storage(x,d[0]) for x in self.cs])
    return cs

Davidson.add_vectors = _davidson_add_vectors
Davidson.add_preconditioned = _davidson_add_preconditioned
