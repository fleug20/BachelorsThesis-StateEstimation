# ekf

## Rust

```sh
cargo test
```

### `no_std`

The crate exposes a `no_std` feature for use in environments without the standard
library (e.g. embedded targets). Disable the default `std` feature and enable
`no_std` instead:

```toml
# Cargo.toml of the consuming crate
[dependencies]
ekf = { version = "...", default-features = false, features = ["no_std"] }
```

To verify a `no_std` build locally:

```sh
cargo build --no-default-features --features no_std --release
```

## Python

Requires Python 3.9+ and a Rust toolchain.

```sh
# At the root of the ba-implementation repository
python3 -m venv .venv

# Apply virtual environment in the current shell
source .venv/bin/activate

cd ekf
pip install -r requirements.txt
maturin develop --features python
```

`maturin develop --features python` compiles the crate with the `python` feature and installs
the resulting extension module into the active virtual environment as `ekf`.

Run the binding test while the shell has the virtual environment applied:

```sh
python examples/binding.py
```

Rebuild after Rust changes with `maturin develop --features python` (add
`--release` for an optimized build).

### API sketch

```python
import numpy as np
from ekf import Ekf, ConstantVelocity, Imu, ImuNoise, AccelConvention, Gnss, GeodeticOrigin

ekf = Ekf()
ekf.predict(ConstantVelocity(accel_stddev=0.5), dt=0.01)

imu = Imu(
    accel=np.array([0.0, 0.0, -9.81]),
    gyro=np.zeros(3),
    noise=ImuNoise(accel_stddev=0.1, gyro_stddev=0.01),
    accel_convention=AccelConvention.SpecificForce,
)
ekf.predict(imu, dt=0.01)

origin = GeodeticOrigin(lat_deg=48.2082, lon_deg=16.3738, alt_m=170.0)
gnss = Gnss.from_geodetic(48.2082, 16.3738, 170.0, origin, stddev=np.array([1.0, 1.0, 2.0]))
ekf.correct(gnss)

ekf.state        # (9,) float64
ekf.covariance   # (9, 9) float64
ekf.attitude     # (4,) float64, [w, x, y, z]
ekf.position     # (3,) float64, [N, E, D]
ekf.velocity     # (3,) float64, [VN, VE, VD]
```

State indices (`IDX_N`, `IDX_E`, `IDX_D`, `IDX_VN`, `IDX_VE`, `IDX_VD`,
`IDX_AX`, `IDX_AY`, `IDX_AZ`) and `STATE_DIM` are also exported by the
module.
