use crate::measurements::tof::Tof;

crate::impl_adaptive_measurement!(AdaptiveTof, Tof<V>, 1, tof, tof_mut);
