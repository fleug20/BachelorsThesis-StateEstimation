use crate::measurements::gnss::{GeodeticOrigin, Gnss};

crate::impl_adaptive_measurement!(
    AdaptiveGnss, Gnss<V>, 3, gnss, gnss_mut,
    constructors [
        fn from_ned(ned: nalgebra::Vector3<V>, stddev: nalgebra::Vector3<V>),
    ],
    methods [
        fn set_ned(ned: nalgebra::Vector3<V>),
    ],
    concrete [
        (f64, Gnss<f64>) {
            constructors [
                fn from_geodetic(
                    lat_deg: f64,
                    lon_deg: f64,
                    alt_m: f64,
                    origin: &GeodeticOrigin<f64>,
                    stddev: nalgebra::Vector3<f64>,
                ),
            ],
            methods [
                fn set_geodetic(
                    lat_deg: f64,
                    lon_deg: f64,
                    alt_m: f64,
                    origin: &GeodeticOrigin<f64>,
                ),
            ]
        },
        (f32, Gnss<f32>) {
            constructors [
                fn from_geodetic(
                    lat_deg: f64,
                    lon_deg: f64,
                    alt_m: f64,
                    origin: &GeodeticOrigin<f64>,
                    stddev: nalgebra::Vector3<f32>,
                ),
            ],
            methods [
                fn set_geodetic(
                    lat_deg: f64,
                    lon_deg: f64,
                    alt_m: f64,
                    origin: &GeodeticOrigin<f64>,
                ),
            ]
        },
    ]
);
