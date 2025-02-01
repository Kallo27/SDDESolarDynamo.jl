# API Reference

## `f(B, B_max = 10, B_min = 1)`
Box-shaped function for the magnetic field range.

- **Arguments**:
  - `B`: Magnetic field.
  - `B_max`: Maximum field limit (default: `10`).
  - `B_min`: Minimum field limit (default: `1`).
- **Returns**: The value of the function for the given magnetic field.

---

## `drift(du, u, h, p, t)`
Drift function for the SDDE.

- **Arguments**:
  - `du`: Derivative of `u`.
  - `u`: State variable.
  - `h`: History function.
  - `p`: Parameter tuple.
  - `t`: Time.
- **Returns**: The drift component of the SDDE.

---

## `noise!(du, u, h, p, t)`
Noise function for the SDDE.

- **Arguments**:
  - Same as `drift`.
- **Returns**: The noise component of the SDDE.

---

## `bfield(θ, Tsim, dt)`
Solves the SDDE problem.

- **Arguments**:
  - `θ`: Parameters `[τ, T, Nd, sigma, Bmax]`.
  - `Tsim`: Time span `(t_start, t_end)`.
  - `dt`: Time step.
- **Returns**: A solution object for the SDDE problem.

---

## `f_dist(θ; type = 1, indices = 1:6:120, fourier_data, Tsim, dt)`
Distance function for the sABC algorithm.

- **Arguments**:
  - `θ`: Parameters for the SDDE model.
  - `type`: Type of distance metric (default: `1`).
  - `indices`: Fourier indices for comparison.
  - `fourier_data`: Reference Fourier data.
  - `Tsim`: Time span for simulation.
  - `dt`: Time step.
- **Returns**: Distance metrics for comparison.

---

## `reduced_fourier_spectrum(u, indices = 1:6:120)`
Computes the reduced Fourier spectrum.

- **Arguments**:
  - `u`: Input time series data.
  - `indices`: Indices of Fourier components to retain.
- **Returns**: Reduced Fourier spectrum.