# Theory of the THz-TDS processing chain

This note explains what `cjcthztds` does to a measured time trace, step by
step, and why each step exists, for students who know Fourier transforms but
have not processed THz time-domain spectroscopy (THz-TDS) data before. It
also fixes the sign convention of the spectra and of the field ratio and
shows how to compare a measured field ratio with a model from
[cjctmm](https://github.com/cjcscheper/cjctmm).

Background: Jepsen, Cooke & Koch, *Terahertz spectroscopy and imaging –
Modern techniques and applications*, Laser Photonics Rev. 5, 124 (2011);
Neu & Schmuttenmaer, *Tutorial: An introduction to terahertz time domain
spectroscopy*, J. Appl. Phys. 124, 231101 (2018). For the discrete Fourier
transform, windows and zero padding: Harris, *On the use of windows for
harmonic analysis with the discrete Fourier transform*, Proc. IEEE 66, 51
(1978).

## 1. What is measured

A THz-TDS setup samples the electric field $`E(t)`$ of a single-cycle THz
pulse: a delay stage moves the probe pulse in steps of $`\Delta t`$
(typically 20–50 fs), and at each step the detector reads the field at that
moment. The result is a **time trace** $`E(t_n)`$, $`t_n = t_0 + n\,\Delta t`$,
$`n = 0 \dots N-1`$, spanning $`T = N\Delta t`$ (typically 20–200 ps).

Two traces are measured: a **reference** $`E_\mathrm{ref}(t)`$ without the
sample (transmission) or on a mirror (reflection), and a **sample** trace
$`E_\mathrm{sam}(t)`$. Everything the setup does to the pulse (emitter,
optics, detector response) is in both; the sample's own effect is in the
ratio of their spectra (section 7).

## 2. One time axis for all traces

The ratio of two spectra needs both on the same frequencies, so both traces
need the same time step and length. Traces from one instrument with one
setting already share their axis, and `analyze` then uses it unchanged.
Otherwise (`common_time_axis`) the traces are interpolated linearly onto a
uniform axis with the reference's step that covers all of them, and a trace
is set to 0 where it was not measured. Linear interpolation is a low-pass
filter; with 20–50 fs steps and a pulse spectrum below ~5 THz its effect in
the usable band is small, but prefer measuring all traces on the same grid.

## 3. Centring on the reference peak

`center_on_reference_peak` shifts the time axis so the reference pulse peaks
at $`t = 0`$; the traces themselves are not moved. The peak is found between
samples (`find_peak`): a cubic spline is drawn through the largest $`|E|`$
sample and `spline_half_width` samples either side, and its largest value is
the peak. A spline passes through every sample and follows any smooth pulse
shape between them, so unlike a fitted parabola it has no bias from the pulse
not being a parabola.
The maximum is found exactly, among the zeros of the spline's derivative and
the samples. The spline interpolates rather than fits: noise on the samples
goes straight into the peak, and there is no fit covariance from which an
uncertainty could be derived, so `find_peak` gives none.

Why centre at all, when it changes **neither the magnitude nor the phase of
the field ratio** (both traces move by the same time, and a common shift
cancels in the ratio, see section 6)?

- The phase of a *single* spectrum contains $`e^{-i\omega t_\mathrm{peak}}`$:
  with the pulse at $`t = 0`$ that ramp is gone and the remaining phase
  describes the pulse shape.
- Window regions are given relative to the reference pulse, so the same
  regions fit every measurement, wherever the stage happened to start.
- The peak time of each sample on the centred axis is directly its delay
  relative to the reference: $`(n-1)d/c`$ for a slab of index $`n`$ and
  thickness $`d`$ in transmission.

## 4. Windowing: keeping the pulse, cutting the echoes

A sample slab produces echoes: the part of the pulse reflected twice inside
the slab arrives $`2nd/c`$ later. In the spectrum the direct pulse and the
echo interfere and give Fabry–Pérot fringes, spaced $`c/(2nd)`$ in
frequency (`examples/example_windowing.py`). If only the direct pass is
wanted, a **window** that is 1 around the direct pulse and 0 at the echo
removes it before the transform. Windows also cut the noise between pulses,
which improves the signal-to-noise ratio at high frequencies.

`tukey_window` is a Tukey (tapered-cosine) window over a region
$`[t_\mathrm{start}, t_\mathrm{stop}]`$ of width $`W`$: with
$`x = (t - t_\mathrm{start})/W`$ and taper fraction $`\alpha`$,

```math
w(x) = \begin{cases}
\tfrac12\left[1 - \cos(2\pi x/\alpha)\right] & 0 \le x < \alpha/2 \\
1 & \alpha/2 \le x \le 1 - \alpha/2 \\
\tfrac12\left[1 - \cos(2\pi (1-x)/\alpha)\right] & 1 - \alpha/2 < x \le 1 \\
0 & \text{otherwise.}
\end{cases}
```

$`\alpha = 0`$ is a rectangle, $`\alpha = 1`$ a Hann window. The trade-offs:

- **Window length.** Multiplying by $`w(t)`$ convolves the spectrum with
  $`\tilde w(f)`$, whose width is about $`1/W`$. Spectral features narrower than
  $`1/W`$ (sharp absorption lines, the fringes themselves) are smoothed out.
  A window must therefore be short enough to exclude the echo and long enough
  to keep the pulse's own ringing (e.g. after a resonant sample).
- **Edges.** A sharp edge where the signal is not zero puts a step into the
  trace, and a step rings across the whole spectrum (the sidelobes of
  $`\tilde w`$ fall only as $`1/f`$ for a rectangle, as $`1/f^3`$ for a cosine
  taper). A larger $`\alpha`$ gives smoother edges but weights the trace down
  over more of the region.
- **Same window for both.** With `center_on_peak=True` (default) every trace
  gets a window of the region's width, centred on its own peak inside the
  region, so a region drawn around the reference also fits a delayed sample,
  and both pulses are weighted the same way. The window's effect then largely
  cancels in the ratio.

Several regions are combined by taking the largest window value at each time.

## 5. Zero padding is interpolation, not information

The discrete Fourier transform of $`N`$ samples gives the spectrum at
$`f_k = k/(N\Delta t)`$, spaced $`\delta f = 1/T`$. Appending zeros
(`zero_pad`) makes $`N`$ larger and $`\delta f`$ smaller, but the zeros were not
measured. The padded transform evaluates **the same** continuous spectrum,
$`\sum_n E(t_n)e^{-i2\pi f t_n}\Delta t`$, at more frequencies: the extra
terms are zero. When the length grows by a whole factor, every new
frequency grid contains the old one, and the padded spectrum passes exactly
through the old values (`test_zero_padding_leaves_the_spectrum_unchanged`).

So zero padding makes curves smoother and peaks easier to read off, but it
does not separate features closer than $`1/T`$ (or $`1/W`$ after a window):
that resolution is set by how long the field was measured.

## 6. The Fourier transform

`fourier_transform` returns

```math
E(f_k) = \sum_{n=0}^{N-1} E(t_n)\, e^{-i 2\pi f_k t_n}\, \Delta t,
\qquad f_k = \frac{k}{N\Delta t},\quad k = 0 \dots \lfloor N/2 \rfloor,
```

which is numpy's `rfft` multiplied by $`\Delta t\, e^{-i2\pi f_k t_0}`$. It
approximates the continuous transform $`\int E(t)\,e^{-i2\pi f t}\,dt`$:

- $`\Delta t`$ makes the amplitude independent of the sampling and padding
  (units: signal × s). A Gaussian $`e^{-t^2/2\sigma^2}`$ gives
  $`\sigma\sqrt{2\pi}\,e^{-2\pi^2\sigma^2 f^2}`$ (`test_fourier_transform_of_a_gaussian`).
- $`e^{-i2\pi f t_0}`$ refers the phase to $`t = 0`$ of the time axis instead
  of to the first sample, so a pulse centred at $`t = 0`$ has a flat phase.

Neither factor changes a field ratio of two traces on the same axis. Only
$`f \ge 0`$ is returned: a real trace has $`E(-f) = E(f)^*`$. The highest
frequency is the Nyquist frequency $`1/(2\Delta t)`$ (12.5 THz for 40 fs
steps). Above the pulse's bandwidth the spectrum is a flat noise floor
(`examples/example_amplitude_spectra.py`); that, not the Nyquist frequency,
limits the usable band.

## 7. The complex field ratio

`compute_complex_field_ratio` divides the sample spectrum by the reference
spectrum:

```math
t(f) = \frac{E_\mathrm{sam}(f)}{E_\mathrm{ref}(f)}.
```

The name is geometry-agnostic on purpose: in transmission it is the sample's
complex transmission coefficient relative to the same path without the
sample, in reflection its reflection coefficient relative to the mirror. The
corresponding power ratio is $`|t|^2`$ (`Analysis.power_ratio`). Where the
reference is exactly zero the ratio is NaN. Outside the band the reference
is tiny but not zero, and the ratio is noise over noise; `freq_min_hz`,
`freq_max_hz` and `frequency_range_mask` limit plots and fits to the usable band (they
never change the computed arrays).

**Phase unwrapping.** `np.angle` gives the phase modulo $`2\pi`$. The phase of
a delayed sample keeps decreasing and wraps every $`2\pi`$; `unwrapped_phase`
removes the jumps, but only inside the band, because in the noise the steps
between frequencies exceed $`\pi`$ and unwrapping goes wrong. Unwrapping fixes
the phase up to a whole multiple of $`2\pi`$; that multiple is chosen so a
straight line fitted to the band phase passes within $`\pi`$ of zero at
$`f = 0`$, as the phase of a real response must (it is odd in $`f`$). Starting
the band too high, where the line extrapolates badly, can still put the
phase off by $`2\pi`$: start it as low as the signal allows.

## 8. Time convention and the sign of the phase

**The spectra use numpy's forward kernel $`e^{-i\omega t}`$:**
$`E(\omega) = \int E(t)\,e^{-i\omega t}\,dt`$, so the field is synthesised as
$`E(t) = \frac{1}{2\pi}\int E(\omega)\,e^{+i\omega t}\,d\omega`$: every frequency
component oscillates as $`e^{+i\omega t}`$. This is the engineering
convention.

**A delayed pulse has a decreasing phase.** If the sample only delays the
pulse by $`\Delta t`$ and scales it by $`a`$, $`E_\mathrm{sam}(t) = a\,E_\mathrm{ref}(t - \Delta t)`$,
then

```math
E_\mathrm{sam}(\omega) = a\, e^{-i\omega\Delta t}\, E_\mathrm{ref}(\omega),
\qquad t(\omega) = a\, e^{-i\omega\Delta t},
\qquad \arg t = -\omega\,\Delta t,
\qquad \frac{d \arg t}{d\omega} = -\Delta t .
```

A sample that delays the pulse (a slab with $`n > 1`$, $`\Delta t = (n-1)d/c`$)
gives a field-ratio phase that *falls* linearly with frequency. This is the
anchor of the convention: `test_field_ratio_of_a_delayed_copy_is_a_phase_ramp`
and `test_field_ratio_of_a_sub_sample_delay` in `tests/test_spectra.py` check
exactly this, and must never be changed to match code.

**Comparing with cjctmm.** cjctmm uses the physics convention: time dependence
$`e^{-i\omega t}`$ and $`n = n' + i\kappa`$ with $`\kappa > 0`$ absorbing, so a
wave through a layer gains $`e^{+in\omega d/c}`$, an *increasing* phase. Every
complex amplitude in one convention is the complex conjugate of the other.
So, for a slab in transmission, with the reference measured through the same
thickness of air:

```math
t_\mathrm{measured}(\omega) = \left[\frac{t_\mathrm{cjctmm}(\omega)}{e^{i\omega d/c}}\right]^*
```

which in code is

```python
result = solve_tmm(omega, layer_thickness, layer_n, conjugate_model=True)   # n' + i kappa as usual
model = result.t * np.exp(1j * omega * d / SPEED_OF_LIGHT)                   # divide out the air path
# model is directly comparable with analysis.field_ratio[1] at omega = 2 pi analysis.freq_hz
```

`conjugate_model=True` returns $`t^*`$; the air path $`e^{i\omega d/c}`$ in the
physics convention becomes $`e^{-i\omega d/c}`$ after conjugation, so dividing
by it multiplies by $`e^{+i\omega d/c}`$. Check with a lossless slab: the
model's phase is $`-n\omega d/c + \omega d/c = -(n-1)\omega d/c`$, the measured
$`-\omega\Delta t`$. Equivalently, conjugate the measurement instead
(`np.conj(analysis.field_ratio)`) and compare with the plain cjctmm result
divided by $`e^{i\omega d/c}`$. Magnitudes and power ratios are the same in
both conventions.
