use crate::measurements::uwb::UwbAnchor;

crate::impl_adaptive_measurement!(AdaptiveUwbAnchor, UwbAnchor<V>, 1, anchor, anchor_mut);
