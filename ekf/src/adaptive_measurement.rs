use core::marker::PhantomData;

use crate::{Ekf, MAX_COVARIANCE, MIN_COVARIANCE, Measurement, ScalarUpdate};
use heapless::HistoryBuf;
use nalgebra::RealField;

pub trait AdaptiveStrategy<V: RealField + Copy> {
    fn update(&mut self, residual: V, hph: V) -> Option<V>;
    /// Returns the most recently estimated noise value, or `None` if not yet available.
    fn current_variance(&self) -> Option<V>;
}

pub struct Window<V: RealField + Copy, const N: usize> {
    buf: HistoryBuf<V, N>,
    last_variance: Option<V>,
}

impl<V: RealField + Copy, const N: usize> Window<V, N> {
    pub fn new() -> Self {
        Self {
            buf: HistoryBuf::new(),
            last_variance: None,
        }
    }
}

impl<V: RealField + Copy, const N: usize> AdaptiveStrategy<V> for Window<V, N> {
    fn update(&mut self, residual: V, hph: V) -> Option<V> {
        let variance = if self.buf.is_full() {
            let n: V = nalgebra::convert(N as f32);
            let c_e: V = self
                .buf
                .iter()
                .copied()
                .map(|v| v * v)
                .fold(nalgebra::convert(0.0_f64), |acc: V, v| acc + v)
                / n;
            let min_var: V = nalgebra::convert(MIN_COVARIANCE);
            let max_var: V = nalgebra::convert(MAX_COVARIANCE);
            let r_hat = (c_e - hph).max(min_var).min(max_var);
            Some(r_hat)
        } else {
            None
        };
        self.buf.write(residual);
        self.last_variance = variance;
        variance
    }

    fn current_variance(&self) -> Option<V> {
        self.last_variance
    }
}

pub struct ExponentialDecay<V: RealField + Copy> {
    pub alpha: V,
    prev_var: V,
}

impl<V: RealField + Copy> ExponentialDecay<V> {
    pub fn new(alpha: V, initial_var: V) -> Self {
        Self {
            alpha,
            prev_var: initial_var,
        }
    }
}

impl<V: RealField + Copy> AdaptiveStrategy<V> for ExponentialDecay<V> {
    fn update(&mut self, residual: V, hph: V) -> Option<V> {
        let min_var: V = nalgebra::convert(MIN_COVARIANCE);
        let max_var: V = nalgebra::convert(MAX_COVARIANCE);
        let r_sample = (residual * residual - hph).max(min_var);
        let new_var =
            ((V::one() - self.alpha) * self.prev_var + self.alpha * r_sample).min(max_var);
        self.prev_var = new_var;
        Some(new_var)
    }

    fn current_variance(&self) -> Option<V> {
        Some(self.prev_var)
    }
}

pub struct AdaptiveMeasurement<M, const D: usize, S, V: RealField + Copy = f64>
where
    M: Measurement<V>,
    S: AdaptiveStrategy<V>,
{
    states: [S; D],
    measurement: M,
    _phantom: PhantomData<V>,
}

impl<T: Measurement<V>, const D: usize, S: AdaptiveStrategy<V>, V: RealField + Copy>
    AdaptiveMeasurement<T, D, S, V>
{
    pub fn inner(&self) -> &T {
        &self.measurement
    }

    pub fn inner_mut(&mut self) -> &mut T {
        &mut self.measurement
    }

    pub fn current_variances(&self) -> [Option<V>; D] {
        core::array::from_fn(|i| self.states[i].current_variance())
    }
}

impl<T: Measurement<V>, const D: usize, const N: usize, V: RealField + Copy>
    AdaptiveMeasurement<T, D, Window<V, N>, V>
{
    pub fn new(measurement: T) -> Self {
        Self {
            states: core::array::from_fn(|_| Window::new()),
            measurement,
            _phantom: PhantomData,
        }
    }
}

impl<T: Measurement<V>, const D: usize, V: RealField + Copy>
    AdaptiveMeasurement<T, D, ExponentialDecay<V>, V>
{
    pub fn new_ema(measurement: T, alpha: V, initial_var: V) -> Self {
        Self {
            states: core::array::from_fn(|_| ExponentialDecay::new(alpha, initial_var)),
            measurement,
            _phantom: PhantomData,
        }
    }
}

impl<T: Measurement<V>, const D: usize, S: AdaptiveStrategy<V>, V: RealField + Copy> Measurement<V>
    for AdaptiveMeasurement<T, D, S, V>
{
    const DIM: usize = D;

    fn scalar(&mut self, i: usize, ekf: &Ekf<V>) -> Option<ScalarUpdate<V>> {
        self.measurement.scalar(i, ekf).map(|mut scalar_update| {
            let h = &scalar_update.jacobian;
            let hph = (h * ekf.covariance * h.transpose())[(0, 0)];

            if let Some(variance) = self.states[i].update(scalar_update.residual, hph) {
                scalar_update.variance = variance;
            }

            scalar_update
        })
    }

    #[cfg(feature = "stats")]
    fn last_residuals(&self, out: &mut [Option<V>]) {
        self.measurement.last_residuals(out);
    }

    #[cfg(feature = "stats")]
    fn last_innovation_covariance(&self, out: &mut [Option<V>]) {
        self.measurement.last_innovation_covariance(out);
    }
}

/// Generate an `AdaptiveMeasurement` newtype wrapper for a `Measurement` type.
///
/// `V` is the float type parameter introduced by the macro; use it in the inner type and
/// argument types. Constructor proxies generate both a `Window` variant and an `_ema`
/// `ExponentialDecay` variant. The `concrete` section handles constructors/methods that
/// exist only on a specific float type; pass the concrete inner type so the macro can call
/// its associated functions directly.
#[macro_export]
macro_rules! impl_adaptive_measurement {
    // Full form: generic + per-concrete-type constructor and method proxies.
    // Each `concrete` entry specifies: (float_type, concrete_inner_type) { constructors, methods }.
    (
        $wrapper:ident, $inner:ty, $dim:expr, $accessor:ident, $accessor_mut:ident,
        constructors [
            $( fn $ctor:ident ( $($ctor_arg:ident : $ctor_ty:ty),* $(,)? ) ),*
            $(,)?
        ],
        methods [
            $( fn $method:ident ( $($m_arg:ident : $m_ty:ty),* $(,)? ) ),*
            $(,)?
        ],
        concrete [
            $( ( $cv:ty , $ci:ty ) {
                constructors [
                    $( fn $c_ctor:ident ( $($c_ctor_arg:ident : $c_ctor_ty:ty),* $(,)? ) ),*
                    $(,)?
                ],
                methods [
                    $( fn $c_method:ident ( $($c_m_arg:ident : $c_m_ty:ty),* $(,)? ) ),*
                    $(,)?
                ]
            } ),*
            $(,)?
        ]
    ) => {
        $crate::impl_adaptive_measurement!(
            $wrapper, $inner, $dim, $accessor, $accessor_mut,
            constructors [ $( fn $ctor ( $($ctor_arg : $ctor_ty),* ) ),* ],
            methods [ $( fn $method ( $($m_arg : $m_ty),* ) ),* ]
        );

        $(
            $(
            impl<const N: usize>
                $wrapper<$crate::adaptive_measurement::Window<$cv, N>, $cv>
            {
                pub fn $c_ctor($($c_ctor_arg: $c_ctor_ty),*) -> Self {
                    Self($crate::adaptive_measurement::AdaptiveMeasurement::new(
                        <$ci>::$c_ctor($($c_ctor_arg),*),
                    ))
                }
            }

            ::paste::paste! {
                impl $wrapper<$crate::adaptive_measurement::ExponentialDecay<$cv>, $cv> {
                    pub fn [<$c_ctor _ema>](
                        $($c_ctor_arg: $c_ctor_ty,)*
                        alpha: $cv,
                        initial_var: $cv,
                    ) -> Self {
                        Self($crate::adaptive_measurement::AdaptiveMeasurement::new_ema(
                            <$ci>::$c_ctor($($c_ctor_arg),*),
                            alpha,
                            initial_var,
                        ))
                    }
                }
            }
            )*

            impl<S: $crate::adaptive_measurement::AdaptiveStrategy<$cv>> $wrapper<S, $cv> {
                $(
                pub fn $c_method(&mut self $(, $c_m_arg: $c_m_ty)*) {
                    self.0.inner_mut().$c_method($($c_m_arg),*)
                }
                )*
            }
        )*
    };

    // Extended form: base + generic constructor proxies + &mut self method proxies.
    (
        $wrapper:ident, $inner:ty, $dim:expr, $accessor:ident, $accessor_mut:ident,
        constructors [
            $( fn $ctor:ident ( $($ctor_arg:ident : $ctor_ty:ty),* $(,)? ) ),*
            $(,)?
        ],
        methods [
            $( fn $method:ident ( $($m_arg:ident : $m_ty:ty),* $(,)? ) ),*
            $(,)?
        ]
    ) => {
        $crate::impl_adaptive_measurement!($wrapper, $inner, $dim, $accessor, $accessor_mut);

        $(
        impl<V: nalgebra::RealField + Copy, const N: usize>
            $wrapper<$crate::adaptive_measurement::Window<V, N>, V>
        {
            pub fn $ctor($($ctor_arg: $ctor_ty),*) -> Self {
                Self($crate::adaptive_measurement::AdaptiveMeasurement::new(
                    <$inner>::$ctor($($ctor_arg),*),
                ))
            }
        }

        ::paste::paste! {
            impl<V: nalgebra::RealField + Copy>
                $wrapper<$crate::adaptive_measurement::ExponentialDecay<V>, V>
            {
                pub fn [<$ctor _ema>]($($ctor_arg: $ctor_ty),*, alpha: V, initial_var: V) -> Self {
                    Self($crate::adaptive_measurement::AdaptiveMeasurement::new_ema(
                        <$inner>::$ctor($($ctor_arg),*),
                        alpha,
                        initial_var,
                    ))
                }
            }
        }
        )*

        impl<S: $crate::adaptive_measurement::AdaptiveStrategy<V>, V: nalgebra::RealField + Copy>
            $wrapper<S, V>
        {
            $(
            pub fn $method(&mut self $(, $m_arg: $m_ty)*) {
                self.0.inner_mut().$method($($m_arg),*)
            }
            )*
        }
    };

    // Base form: struct, new/new_ema constructors, named accessors, Measurement impl.
    ($wrapper:ident, $inner:ty, $dim:expr, $accessor:ident, $accessor_mut:ident) => {
        pub struct $wrapper<
            S: $crate::adaptive_measurement::AdaptiveStrategy<V>,
            V: nalgebra::RealField + Copy = f64,
        >($crate::adaptive_measurement::AdaptiveMeasurement<$inner, $dim, S, V>);

        impl<V: nalgebra::RealField + Copy, const N: usize>
            $wrapper<$crate::adaptive_measurement::Window<V, N>, V>
        {
            pub fn new(inner: $inner) -> Self {
                Self($crate::adaptive_measurement::AdaptiveMeasurement::<
                    $inner,
                    {$dim},
                    $crate::adaptive_measurement::Window<V, N>,
                    V,
                >::new(inner))
            }
        }

        impl<V: nalgebra::RealField + Copy>
            $wrapper<$crate::adaptive_measurement::ExponentialDecay<V>, V>
        {
            pub fn new_ema(inner: $inner, alpha: V, initial_var: V) -> Self {
                Self($crate::adaptive_measurement::AdaptiveMeasurement::<
                    $inner,
                    {$dim},
                    $crate::adaptive_measurement::ExponentialDecay<V>,
                    V,
                >::new_ema(inner, alpha, initial_var))
            }
        }

        impl<S: $crate::adaptive_measurement::AdaptiveStrategy<V>, V: nalgebra::RealField + Copy>
            $wrapper<S, V>
        {
            pub fn $accessor(&self) -> &$inner {
                let m: &$crate::adaptive_measurement::AdaptiveMeasurement<$inner, {$dim}, S, V> = &self.0;
                m.inner()
            }

            pub fn $accessor_mut(&mut self) -> &mut $inner {
                let m: &mut $crate::adaptive_measurement::AdaptiveMeasurement<$inner, {$dim}, S, V> = &mut self.0;
                m.inner_mut()
            }

            pub fn current_variances(&self) -> [Option<V>; {$dim}] {
                let m: &$crate::adaptive_measurement::AdaptiveMeasurement<$inner, {$dim}, S, V> = &self.0;
                m.current_variances()
            }
        }

        impl<S: $crate::adaptive_measurement::AdaptiveStrategy<V>, V: nalgebra::RealField + Copy>
            $crate::Measurement<V> for $wrapper<S, V>
        {
            const DIM: usize = $dim;

            fn scalar(&mut self, i: usize, ekf: &$crate::Ekf<V>) -> Option<$crate::ScalarUpdate<V>> {
                let m: &mut $crate::adaptive_measurement::AdaptiveMeasurement<$inner, {$dim}, S, V> = &mut self.0;
                m.scalar(i, ekf)
            }

            #[cfg(feature = "stats")]
            fn last_residuals(&self, out: &mut [Option<V>]) {
                let m: &$crate::adaptive_measurement::AdaptiveMeasurement<$inner, {$dim}, S, V> = &self.0;
                m.last_residuals(out)
            }

            #[cfg(feature = "stats")]
            fn last_innovation_covariance(&self, out: &mut [Option<V>]) {
                let m: &$crate::adaptive_measurement::AdaptiveMeasurement<$inner, {$dim}, S, V> = &self.0;
                m.last_innovation_covariance(out)
            }
        }
    };
}
