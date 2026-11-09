"""SCIENTIFIC ONLY: exact NumPy/Numba MT19937 state-pointer access intrinsics.

This is NOT a stable public Numba API. Do not ship or merge into production.
All three LLVM primitives below delegate to Numba's installed, original
randomimpl generator, and reuse its real thread-local pointer ONLY within
the same Numba kernel call. Neither the generator nor integer sampling is
reimplemented. Numba 0.65 and 0.68 may lower these differently.
"""
from llvmlite import ir
from numba import types
from numba.core.typing import signature
from numba.extending import intrinsic
from numba.cpython import randomimpl

_VOID_PTR=ir.IntType(8).as_pointer()

@intrinsic
def np_state_ptr(typingcontext):
    """Exactly Numba's current NumPy random state, no state copy/new PRNG."""
    def codegen(context,builder,sig,args):
        actual=randomimpl.get_state_ptr(context,builder,"np")
        return builder.bitcast(actual,_VOID_PTR)
    return signature(types.voidptr),codegen


@intrinsic
def random_from_state(typingcontext,ptr):
    """Exact Numba np.random.random() float64 draw from SAME MT19937."""
    if ptr!=types.voidptr:
        raise TypeError("Requires same-call Numba voidptr state handle")
    def codegen(context,builder,sig,args):
        actual=builder.bitcast(args[0],randomimpl.rnd_state_ptr_t)
        return randomimpl.get_next_double(context,builder,actual)
    return signature(types.float64,ptr),codegen


@intrinsic
def uniform_from_state(typingcontext,lo,hi,ptr):
    """Same LLVM operation order as original Numba np.random.uniform."""
    if (lo!=types.float64 or hi!=types.float64 or
            ptr!=types.voidptr):
        raise TypeError("Requires exact float64/float64/same-call pointer")
    def codegen(context,builder,sig,args):
        lower,upper,untyped_ptr=args
        width=builder.fsub(upper,lower)
        actual=builder.bitcast(untyped_ptr,randomimpl.rnd_state_ptr_t)
        sampled=randomimpl.get_next_double(context,builder,actual)
        return builder.fadd(lower,builder.fmul(width,sampled))
    return signature(types.float64,lo,hi,ptr),codegen
