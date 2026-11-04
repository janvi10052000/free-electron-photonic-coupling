# Theory

Derivations and references are added here at the start of each phase. Phase 0 (setup) has no equations. Reference numbers stay the same across phases, so they are not consecutive within a phase.

## Phase 1: PINEM sidebands

A fast electron crosses the near field of a structure or a guided optical mode. In a field of one frequency the Hamiltonian is periodic in time, so the electron's energy can change only by whole multiples of the photon energy $\hbar\omega$. This is the basis of photon-induced near-field electron microscopy (PINEM) [1, 2, 8]. We want the probability $P_n$ of a net exchange of $n$ photons, and the spectrum a spectrometer would record.

Phase 1's tests recompute the numbers in sections 1.5 to 1.8, except the transverse estimate in section 1.8, which is by hand.

### 1.1 Conventions

These choices fix every factor of 2 and every sign below.

- $e>0$, so the electron has charge $-e$. It moves along $+z$ at its relativistic speed $v$, with $\gamma=(1-v^2/c^2)^{-1/2}$, and $m$ is its rest mass.
- **Field convention.** The real field is written with a complex amplitude $E_z(z)$ as $\mathcal{E}_z(z,t)=\mathrm{Re}[E_z(z)e^{-i\omega t}]$. So $\lvert E_z(z)\rvert$ is the peak real field at $z$. Section 1.4 shows why this matters.
- $n>0$ means the electron gains energy.
- $W$ is the electron's total energy and $E_k=(\gamma-1)mc^2$ is its kinetic energy. $E$ is the energy axis of a spectrum, measured from the centre of the zero-loss peak (the line the beam makes in the spectrometer when no energy is exchanged) and positive for gain.
- $\tau=t-z/v$ is the electron's arrival time at $z=0$. It stays constant along the electron's path and fixes which part of the optical cycle the electron rides on.

### 1.2 Derivation of $g$ and $P_n$

**Step 1: linearize the Hamiltonian.** Take one electron on a straight line in a prescribed field, in the gauge with no scalar potential, so $\mathcal{E}_z=-\partial_tA_z$:

$$H=\sqrt{m^2c^4+c^2\left(\hat p_z+eA_z\right)^2}$$

Expand about the central momentum $p_0=\gamma mv=\hbar k_e$, with central energy $W_0=\gamma mc^2$. The first derivative of energy with respect to momentum is $v$, and the second is $1/(\gamma^3m)$. Keeping only the linear term gives

$$H\approx W_0+v\left(\hat p_z-p_0\right)+evA_z(z,t)$$

This is the key approximation. The largest dropped term, $(\hat p_z+eA_z-p_0)^2/(2\gamma^3m)$, contains recoil, meaning the change of velocity with momentum, and the term $e^2A_z^2/(2\gamma^3m)$, the energy of any fast quiver in the field, which an electron riding a phase-matched wave does not have. Section 1.8 estimates both.

Only $A_z$ appears because the first-order term of the full three-dimensional expansion is $e\mathbf{v}\cdot\mathbf{A}$, with $\mathbf{v}$ along $z$. Physically, only $\mathcal{E}_z$ does work on a straight path along $z$.

**Step 2: equation for the envelope.** Factor out the central plane wave, $\psi=e^{i(k_ez-W_0t/\hbar)}\phi(z,t)$, and put it in $i\hbar\partial_t\psi=H\psi$. Acting on $\phi$, $i\hbar\partial_t$ becomes $W_0+i\hbar\partial_t$ and $\hat p_z$ becomes $p_0-i\hbar\partial_z$, so the $W_0$ and $p_0$ terms cancel, leaving

$$\left(\partial_t+v\partial_z\right)\phi=-\frac{iev}{\hbar}A_z(z,t)\phi$$

**Step 3: follow the electron.** The left side is the rate of change seen while riding along with the electron. Change variables to $\tau=t-z/v$ and $\zeta=z$. Since $\partial_t=\partial_\tau$ and $\partial_z=\partial_\zeta-\frac{1}{v}\partial_\tau$, the left side becomes $v\partial_\zeta\phi$ at fixed $\tau$, and the $v$ cancels. So each arrival time evolves on its own along the straight line $z=v(t-\tau)$. Integrating through the field gives

$$\phi_{\rm out}(\tau)=\phi_{\rm in}(\tau)e^{i\Phi(\tau)},\qquad\Phi(\tau)=-\frac{e}{\hbar}\int A_z\left(z,\tau+\frac{z}{v}\right)dz$$

Within the linearized $H$ this is a pure phase for any field strength, so the arrival-time density at the exit is unchanged and only the energy and momentum content changes. That is a property of the linearization: with recoil the density at the exit is already slightly modulated (section 1.8), and a later free drift turns the phase modulation into bunching [1].

**Step 4: single frequency.** The matching vector potential is $A_z=\mathrm{Re}[E_z(z)e^{-i\omega t}/(i\omega)]$, because $-\partial_t$ of it returns $\mathcal{E}_z$. On the electron's path $t=\tau+z/v$, so $e^{-i\omega t}=e^{-i\omega\tau}e^{-i\omega z/v}$. That is where the exponential in $g$ comes from. Using $\mathrm{Re}(X)=\frac{1}{2}(X+X^\ast)$, which is where the $\frac{1}{2}$ comes from, the $\Phi$ of Step 3 gives

$$i\Phi(\tau)=-\frac{e}{2\hbar\omega}\left[Qe^{-i\omega\tau}-Q^\ast e^{i\omega\tau}\right],\qquad Q=\int E_z(z)e^{-i\omega z/v}dz$$

The coefficient of $Q$ defines the coupling constant $g$:

$$\boxed{g=\frac{e}{2\hbar\omega}\int E_z(z)e^{-i\omega z/v}dz}$$

$$i\Phi(\tau)=g^\ast e^{i\omega\tau}-ge^{-i\omega\tau}=2i\lvert g\rvert\sin\left(\omega\tau-\varphi_g\right),\qquad\varphi_g=\arg g$$

So $g$ is the spatial Fourier component of $E_z$ at wavenumber $\omega/v$. Gaining $\hbar\omega$ takes momentum $\hbar\omega/v>\hbar\omega/c$, so a plane wave in vacuum over an infinite path gives $g=0$. The field must be a near field or a slow guided mode [8].

**Step 5: expand in sidebands.** The Jacobi–Anger expansion, with Bessel functions $J_\ell$, is $e^{ix\sin\theta}=\sum_\ell J_\ell(x)e^{i\ell\theta}$. Take $x=2\lvert g\rvert$ and $\theta=\omega\tau-\varphi_g$, set $\ell=-n$, and use $J_{-n}=(-1)^nJ_n$:

$$e^{i\Phi(\tau)}=\sum_nc_ne^{-in\omega\tau},\qquad c_n=(-1)^nJ_n(2\lvert g\rvert)e^{in\varphi_g}=J_n(2\lvert g\rvert)e^{in\arg(-g)}$$

Putting back the plane wave of Step 2 with $\tau=t-z/v$, term $n$ of $\psi$ varies as $e^{i(k_e+n\omega/v)z-i(W_0+n\hbar\omega)t/\hbar}$. So it has energy $W_0+n\hbar\omega$, and $n>0$ is gain. It has momentum $p_0+n\hbar\omega/v$ within the linearized $H$; section 1.8 gives the next term. The amplitude has the same form as eq 2.3 of [8]. The phase $\varphi_g$ drops out of the probabilities:

$$\boxed{P_n=\lvert c_n\rvert^2=J_n^2(2\lvert g\rvert)}$$

### 1.3 A classical check on the prefactor

The work done on a classical electron with arrival time $\tau$ is

$$\Delta W(\tau)=-e\int\mathcal{E}_z\left(z,\tau+\frac{z}{v}\right)dz=-2\lvert g\rvert\hbar\omega\cos\left(\omega\tau-\varphi_g\right)=-\hbar\partial_\tau\Phi(\tau)$$

A phase $\Phi(t-z/v)$ shifts the local energy by $-\hbar\partial_t\Phi=-\hbar\partial_\tau\Phi$, which is exactly the classical work above. The largest classical energy change is therefore $2\lvert g\rvert\hbar\omega$. That is also where the sideband comb ends, as [1] notes, since $J_n(x)$ falls off rapidly once $\lvert n\rvert>x=2\lvert g\rvert$.

For a phase-matched travelling wave $E_z(z)=E_0e^{i\omega z/v}$ with real positive $E_0$, over a length $L$,

$$2\lvert g\rvert\hbar\omega=eE_0L$$

The right side is force times distance and contains no $\hbar$. So classical mechanics alone fixes the prefactor, which makes this a good test of the code's prefactor and sign.

### 1.4 The factor of 2 between conventions

A "complex field amplitude" does not by itself fix the prefactor. $e/(2\hbar\omega)$ is correct only for the peak-amplitude convention of section 1.1. The three cited papers write the same physics differently:

| Source | Real field | Coupling quantity | Sidebands | Relation to our $g$ |
|---|---|---|---|---|
| Feist et al. [1] | $A=F(z)\sin(\omega t)/\omega$, so $F$ is a peak amplitude | $g=\frac{e}{2\hbar\omega}\int F(z)e^{-i\Delta kz}dz$, with $\Delta k\approx\omega/v$ | $J_n^2(2\lvert g\rvert)$ | $E_z=-F$, so the same $\lvert g\rvert$ |
| Park et al. [2] | $\mathrm{Re}[\tilde E_z(z,0)e^{-i\omega t}]$, from their Eq. A.2 with $\vec E=-\partial_t\vec A$ | $\Omega=\frac{e}{\hbar\omega}\lvert\tilde F_z\rvert$, with $\tilde F_z=\int\tilde E_z(z,0)e^{-i\omega z/v}dz$ | $J_n^2(\Omega)$ | $\Omega=2\lvert g\rvert$ |
| García de Abajo et al. [8] | $E^{[8]}_ze^{-i\omega t}+\text{c.c.}$, from their $A=\frac{2c}{\omega}\mathrm{Im}\lbrace E^{[8]}e^{-i\omega t}\rbrace$ | $\beta=\frac{e}{\hbar\omega}\int E^{[8]}_ze^{-i\omega z/v}dz$ (eq 2.4) | $J_n^2(2\lvert\beta\rvert)$ | $E_z=2E^{[8]}_z$, so $\beta=g$ |

The factor $c$ in the $A$ of [8] goes with Gaussian-style units, where the field is $-\partial_tA/c$. $eE$ is a force in either system, so $\beta$ has the same form in SI.

Traps to avoid:

- A "+ c.c." amplitude used with our prefactor makes $\lvert g\rvert$ too small by exactly 2.
- Taking Park's $\Omega$ as $\lvert g\rvert$ makes it too large by exactly 2. In [2] the letter $g$ means the electron envelope, not the coupling.
- A field computed with the $e^{+i\omega t}$ convention must be complex-conjugated before it goes into our formula.
- Later phases must keep this convention: a cycle-averaged power written with peak amplitudes carries a factor $\frac{1}{2}$.

### 1.5 Checks on $P_n$

Put $x=2\lvert g\rvert$. The $J_n(x)$ are the Fourier coefficients of $f(\theta)=e^{ix\sin\theta}$, so Parseval's theorem gives most of these.

| Check | Why it holds |
|---|---|
| $\sum_nP_n=1$ | $\lvert f\rvert=1$ and Parseval. The phase factor conserves probability, so this model has no way to lose an electron. A real one can still hit the structure (Phase 4) or scatter outside the spectrometer's acceptance. |
| $P_{-n}=P_n$ | $J_{-n}=(-1)^nJ_n$, and squaring removes the sign. |
| $\sum_nnP_n=0$ | The summand is odd in $n$. The classical energy change also averages to zero over a cycle. |
| $\sum_nn^2P_n=2\lvert g\rvert^2$ | The Fourier coefficients of $\partial_\theta f=ix\cos\theta f$ are $inJ_n$, and $\lvert\partial_\theta f\rvert^2=x^2\cos^2\theta$ averages to $x^2/2$. Parseval then gives the sum. It equals the classical $\langle\Delta W^2\rangle/(\hbar\omega)^2$ over a cycle. |
| $P_0$ first vanishes at $\lvert g\rvert=j_{0,1}/2=1.2024$ | $j_{0,1}=2.4048$ is the first zero of $J_0$. It is a double zero of $P_0$, with $P_0\approx4J_1^2(j_{0,1})(\lvert g\rvert-j_{0,1}/2)^2$. |
| $P_{\pm1}=\lvert g\rvert^2-\lvert g\rvert^4+\dots$ | From the series $J_1(2y)=y-y^3/2+\dots$ In general $P_{\pm n}\approx\lvert g\rvert^{2n}/(n!)^2$. |

The rounded value 1.2024 gives a tiny but non-zero $P_0$, so the code must use the exact root of $J_0$.

### 1.6 The spectrum

The zero-loss peak (ZLP), normalized over energy, and the spectrum are

$$Z(E)=\frac{1}{\sigma_E\sqrt{2\pi}}\exp\left(-\frac{E^2}{2\sigma_E^2}\right),\qquad\mathrm{FWHM}=2\sqrt{2\ln2}\sigma_E\approx2.355\sigma_E$$

$$S(E)=\sum_nP_nZ\left(E-n\hbar\omega\right)$$

At $\lambda=1550$ nm, $\hbar\omega=hc/\lambda=0.7999$ eV, about 0.80 eV. The code must use the exact value. The FWHM range we consider, 0.3 to 1.0 eV, is $\sigma_E=0.127$ to $0.425$ eV.

**Moments.** Each peak has $\int E^2Z(E-n\hbar\omega)dE=\sigma_E^2+(n\hbar\omega)^2$, so

$$\int S(E)dE=1,\qquad\langle E\rangle=0,\qquad\mathrm{Var}(E)=\sigma_E^2+(\hbar\omega)^2\sum_nn^2P_n=\sigma_E^2+2\lvert g\rvert^2(\hbar\omega)^2$$

These hold for any symmetric $Z$ with finite variance, whether or not the peaks are resolved. They test the model and the code. They are not an estimator for measured spectra, whose Voigt-type wings have no finite variance.

**The Gaussian is our modelling choice.** None of the three papers models a measured spectrum this way: [1] fits pseudo-Voigt profiles, with a 0.7 eV FWHM incident peak; [2] uses Voigt fits; and [8] gives no line-shape formula.

**When adding probabilities is valid.** $S(E)$ adds probabilities, not amplitudes. The exit amplitude at energy $E$ is $\sum_nc_n\tilde\phi_{\rm in}(E-n\hbar\omega)$, where $\tilde\phi_{\rm in}$ is the incident amplitude in energy. Squaring it and averaging over the incident mixture gives

$$S_{\rm exact}(E)=\sum_{n,n'}c_nc_{n'}^\ast\rho\left(E-n\hbar\omega,E-n'\hbar\omega\right)$$

where $\rho(E,E')$ is the incident density matrix in energy and $Z(E)=\rho(E,E)$. Our $S(E)$ keeps only $n=n'$, so it needs the incident electron to have no coherence between energies that differ by a multiple of $\hbar\omega$.

- For a beam with no fixed optical phase this is exact, whether the width comes from the detector or from a stationary incoherent energy spread, however short the emitter's coherence time, because each energy gets the same comb.
- For a pulse, the energy-integrated terms between sidebands $q$ apart equal $\lvert\int D(\tau)e^{iq\omega\tau}d\tau\rvert$, with $D$ the arrival-time density. For a Gaussian of total spread $\sigma_\tau$ this is $e^{-q^2\omega^2\sigma_\tau^2/2}$. A density even over one optical period (5.17 fs at 1550 nm) makes it exactly zero, and $S(E)$ itself then holds only if that spread is a mixture of arrival times, not one coherent pulse.
- It fails when the beam keeps a fixed phase to the field, for example an electron already modulated at $\omega$ with a fixed phase to this field, or a pulse locked to the optical phase with structure at or below the optical period. A comb with random phase to the field only widens $Z$.

A transform-limited ZLP in our range would last only $\sigma_\tau=\hbar/(2\sigma_E)$, which is 2.6 to 0.77 fs. So a long coherent pulse is not what justifies $S(E)$ here. Incoherence, or an arrival-time spread not locked to the optical phase, is.

**Resolving the sidebands.** Take an isolated pair of equal peaks with half-spacing $a=\hbar\omega/2$. The second derivative of their sum at the midpoint is proportional to $a^2/\sigma_E^2-1$, so the dip between them disappears at $\sigma_E=\hbar\omega/2$:

$$\mathrm{FWHM}_c=\sqrt{2\ln2}\hbar\omega\approx1.177\hbar\omega=0.942\text{ eV at }1550\text{ nm}$$

This is a rule of thumb for one pair, not a limit on the measurement. A real $S(E)$ has unequal peaks and usually more than two. At weak coupling its peaks merge at smaller widths, and at strong coupling a weak central peak leaves structure well above this width. So although the top of our range, 1.0 eV, is above $\mathrm{FWHM}_c$, we expect $\lvert g\rvert$ to stay recoverable from a fit. Phase 6 tests this.

For the same ZLP width, sidebands at 1550 nm are half as far apart as at the 800 nm used in [1]. As a ratio $\mathrm{FWHM}/\hbar\omega$, [1] is about 0.45 and our range is 0.38 to 1.25. Henke et al. [4] work at this photon energy with a 0.5 eV gun energy spread, and fit individual sidebands with a comb of Voigt peaks and $P_N=J_N^2(2\lvert g\rvert)$.

### 1.7 Incoherent average

Electrons at different beam positions or arrival times sample different values of $\lvert g\rvert$. With $u=\lvert g\rvert$ and a distribution $w(u)$, we add probabilities:

$$\bar P_n=\int_0^\infty w(u)J_n^2(2u)du$$

Both [1] (Methods) and [2] (Eqs 22 and 28) average probabilities in this way. Roadmap [8] gives no such formula. The reasons differ for the two variables:

- **Beam position.** Adding probabilities is exact once the spectrometer collects all transverse momenta, because position states are orthogonal. It does not depend on the beam's transverse coherence.
- **Arrival time.** The spectrometer measures energy, which is conjugate to arrival time, so this is not about resolution. The total weight of sideband $n$, $\int\lvert\phi_{\rm in}(\tau)\rvert^2J_n^2(2\lvert g(\tau)\rvert)d\tau$, follows from Parseval's theorem when the incident electron meets the no-coherence condition of section 1.6 and the structure $g$ puts inside each sideband is much smaller than $\hbar\omega$, so that neighbouring sidebands do not mix. For an envelope of width $T$ and peak $g_0$, that structure is about $\max(1,2g_0)\hbar/T$. Drawing every sideband with the same $Z$ needs two more things: a ZLP much wider than that structure, and a ZLP energy uncorrelated with arrival time.

Here $\langle\cdot\rangle$ is the average over $w(u)$:

| Quantity | Single $\lvert g\rvert$ | After averaging |
|---|---|---|
| Normalization | $1$ | $1$ |
| Symmetry | $P_{-n}=P_n$ | $\bar P_{-n}=\bar P_n$ |
| First moment | $0$ | $0$ |
| Second moment | $2\lvert g\rvert^2$ | $2\langle\lvert g\rvert^2\rangle$ |
| Spectrum variance | $\sigma_E^2+2\lvert g\rvert^2(\hbar\omega)^2$ | $\sigma_E^2+2\langle\lvert g\rvert^2\rangle(\hbar\omega)^2$ |
| Zero of $P_0$ | at $\lvert g\rvert=1.2024$ | fills in |

For a narrow spread of standard deviation $\delta$ about the zero, $\bar P_0\approx4J_1^2(j_{0,1})\delta^2\approx1.078\delta^2$.

Three exact results give tests for the averaging code:

- **Pulse envelope.** For $\lvert g\rvert(\tau)=g_0\exp[-\tau^2/(2T^2)]$ and Gaussian arrival times of standard deviation $\sigma_\tau$ centred on the peak, $\langle\lvert g\rvert^2\rangle=g_0^2/\sqrt{1+2\sigma_\tau^2/T^2}$. In code, integrate over $\tau$ rather than $u$, because $w(u)$ diverges at the pulse peak.
- **Finite beam.** For a field decaying as $\lvert g\rvert=g_0e^{-\kappa x}$ with decay rate $\kappa$, and Gaussian beam positions $x$ of mean 0 and standard deviation $s$, ignoring clipping, $\langle\lvert g\rvert^2\rangle=g_0^2\exp(2\kappa^2s^2)$.
- **Rayleigh benchmark.** For $w(u)=(2u/\mu)e^{-u^2/\mu}$, where $\mu=\langle u^2\rangle$, eq 10.22.67 of [D], with its $t$ as our $u$, $a=b=2$ and its $p^2=1/\mu$, gives $\bar P_n=e^{-2\mu}I_{\lvert n\rvert}(2\mu)$, where $I_\nu$ is the modified Bessel function of the first kind. This is a mathematical check, not a model of a real beam.

Mapping a real beam, with clipping, to $w(u)$ comes in Phase 4.

### 1.8 What is assumed, and when it fails

**Assumed:**

- **Classical field.** The field is prescribed, with no photon statistics and no back-action. This is the limit $\lvert\alpha\rvert\to\infty$, $\lvert g_{\rm qu}\rvert\to0$ at fixed $\lvert g\rvert=\lvert g_{\rm qu}\alpha\rvert$ of the quantum interaction with a coherent state $\alpha$. At finite $\lvert g_{\rm qu}\rvert$ the electron also loses $\lvert g_{\rm qu}\rvert^2$ photons on average, so $\sum_nnP_n\neq0$. The quantum case comes in Phase 3.
- **One straight ray** at a fixed transverse position, so $g$ depends on the position chosen. For a field with decay rate $\kappa$, a beam tilt or divergence $\vartheta$ needs $\vartheta L\kappa\ll1$. The field also pushes the electron sideways (below).
- **Vacuum path.** The electron starts and ends outside the field, with no inelastic scattering in material.
- **Independent electrons.** Electrons do not interact with each other.
- **One $g$ for the whole ZLP.** Each energy component is modulated independently, so no coherent beam is needed. Every energy sees the same $g$ when $k_TL\mathrm{FWHM}/\hbar\omega\ll1$, with $k_T$ defined below.
- **Single frequency or slow envelope.** An envelope of width $T$ must vary slowly compared with the optical period, $\omega T\gg\max(1,2g_0)$, and with the time it slips past the electron over the interaction, $L\lvert1/v-1/v_g\rvert$, with $v_g$ its group velocity. In a waveguide this is $L\lvert n_g-n_e\rvert/c$, with $n_g$ the group index and $n_e=c/v$. If it is slow compared with the period but not with the slip time, a guided envelope $f(t-z/v_g)$ stays inside the integral, $g(\tau)=\frac{e}{2\hbar\omega}\int E_z(z)f(\tau+z/v-z/v_g)e^{-i\omega z/v}dz$, and $\lvert g\rvert(\tau)$ is not simply $g_0f(\tau)$.
- **Full collection.** The spectrometer collects all transverse momenta.
- **Spin and the negative-energy branch** are ignored.
- **No recoil**, from the linearization in Step 1.

**Not assumed:** perturbation theory in $g$, since within the linearized $H$ the result holds for any $\lvert g\rvert$, although recoil still limits the field strength (below); a non-relativistic electron, since $\gamma$ enters $g$ only through $v$; or a rotating-wave approximation, since both $e^{\pm i\omega t}$ terms are kept. The formula does not assume phase matching either, but recoil does more harm away from it.

**Recoil.** Let $\Delta p=\hbar(k_n-k_e)$ be the momentum change of sideband $n$. The Taylor series of Step 1 gives $n\hbar\omega=v\Delta p+\Delta p^2/(2\gamma^3m)$. Solving to second order gives

$$k_n=k_e+\frac{n\omega}{v}-k_Tn^2+\dots,\qquad k_T=\frac{\hbar\omega^2}{2\gamma^3mv^3}=\frac{2\pi}{z_T}$$

where $z_T$ is the Talbot distance given in section 2.1.3 of [8]. After a free drift $z_T$ every sideband's recoil phase $k_Tn^2z_T=2\pi n^2$ is a multiple of $2\pi$, so the sidebands are back in step, the electron analogue of optical Talbot self-imaging. Drift changes only the phases of $c_n$, so it does not change $P_n$. At 1550 nm, $z_T\approx0.25$ m at 60 keV and $3.7$ m at 300 keV.

With recoil, the sideband amplitudes obey coupled equations. For a uniform wave $E_z=E_0e^{i(\omega/v+\Delta)z}$ over a length $L$, with $h=eE_0/(2\hbar\omega)$ and after removing a factor $e^{in\Delta z}$,

$$\partial_zc_n=-i\left(k_Tn^2+n\Delta\right)c_n-hc_{n-1}+h^\ast c_{n+1}$$

The first term is the free evolution of each sideband, and the other two are the photon exchange of Step 5. At $k_T=0$ this gives the Bessel result. The function `sideband_amplitudes_with_recoil` in `src/fepc/pinem.py` solves it exactly. It shows the following, all in one dimension:

- **At weak coupling** the first sideband itself is mismatched by $k_T$, so $P_{\pm1}\approx\lvert g\rvert^2\mathrm{sinc}^2(k_TL/2)$, with $\mathrm{sinc}(x)=\sin(x)/x$. So we always need $k_TL\ll1$. For $L\le1$ mm, $k_TL\le0.026$ from 60 to 300 keV.
- **At strong coupling** the number to watch is an optical phase slip $\xi$. On a uniform phase-matched wave, an electron at the extreme phase gains energy $eE_0z$ up to position $z$, so its speed changes by $\Delta v=eE_0z/(\gamma^3mv)$. Its arrival at the end of the wave shifts by $\int_0^L(\Delta v/v^2)dz$, an optical phase slip of

  $$\xi=\frac{\omega L}{v}\frac{eE_0L}{2\gamma^3mv^2}=2k_T\lvert g\rvert L$$

  This sets the width of the comb: $\sum_nn^2P_n$ falls by a fraction of about $0.11\xi^2$. Individual $P_n$ are more sensitive. At large $\lvert g\rvert$ the error $\frac{1}{2}\sum_n\lvert P_n-J_n^2\rvert$ reaches 1% at $\xi\approx1.3/\sqrt{2\lvert g\rvert}$. For 60 keV and $L=1$ mm, $\xi$ reaches 1 at $\lvert g\rvert\approx20$, but this error already reaches 1% at $\lvert g\rvert\approx6$. A spread of $\lvert g\rvert$ blurs the fringes in $P_n$ and relaxes this.
- **Density.** A phase slip of up to $\xi$ means the arrival-time density at the exit is modulated by about $\xi$.
- **Off phase matching**, recoil already acts at first order and breaks $P_{-n}=P_n$. At strong coupling the number to watch is then $k_T\lvert g\rvert^2L=\lvert g\rvert\xi/2$, unless $g$ is reduced by cancellation along the path. At weak coupling the relative asymmetry of $P_{\pm1}$ is of order $k_TL$.

Three further points lie outside this model:

- **Transverse recoil is as large.** At phase matching a TM evanescent field has $\lvert\mathcal{E}_x\rvert=\gamma\lvert\mathcal{E}_z\rvert$, and the magnetic force cancels all but $1/\gamma^2$ of its push, leaving a sideways force $eE_0/\gamma$. With decay rate $\kappa=\omega/(\gamma v)$, the sideways shift reaches $\kappa\Delta x\approx\xi$. So $\xi\ll1$ also keeps the ray straight, and the one-dimensional results above give only the order of magnitude.
- **Ponderomotive term.** A phase-matched electron sees a static field in its rest frame, so it does not quiver: the $A_z^2$ term cancels against the $-eA_z$ part of the canonical momentum, leaving only the recoil above. A real ponderomotive phase needs a field that oscillates along the path.
- **Energy change.** The condition $2\lvert g\rvert\hbar\omega\ll E_k$ is also necessary. It is much weaker than these for $L$ well above a micrometre, and for a shorter near field it is the first to fail.

### References for Phase 1

- [1] A. Feist, K. E. Echternkamp, J. Schauss, S. V. Yalunin, S. Schäfer and C. Ropers, Quantum coherent optical phase modulation in an ultrafast transmission electron microscope, Nature **521**, 200–203 (2015). https://doi.org/10.1038/nature14463
- [2] S. T. Park, M. Lin and A. H. Zewail, Photon-induced near-field electron microscopy (PINEM): theoretical and experimental, New J. Phys. **12**, 123028 (2010). https://doi.org/10.1088/1367-2630/12/12/123028
- [4] J.-W. Henke et al., Integrated photonics enables continuous-beam electron phase modulation, Nature **600**, 653–658 (2021). https://doi.org/10.1038/s41586-021-04197-5
- [8] F. J. García de Abajo et al., Roadmap for Quantum Nanophotonics with Free Electrons, ACS Photonics **12**, 4760–4817 (2025). https://doi.org/10.1021/acsphotonics.5c00585
- [D] NIST Digital Library of Mathematical Functions, section 10.22, Version 1.2.8 (release date 2026-09-15). https://dlmf.nist.gov/10.22
- [T] A. Feist, Next-Generation Ultrafast Transmission Electron Microscopy – Development and Applications, PhD thesis, Georg-August-University Göttingen (2018). https://doi.org/10.53846/goediss-7028

**Source notes.** Everything attributed to [1] was read in the reproduction of that article in chapter 5 of [T]: Eq. 5.1 (pseudo-Voigt fit), the $A$ written just before Eq. 5.3, Eq. 5.3 ($g$), Eq. 5.15 ($P_N$), section 5.2 (cutoff), sections 5.4.1 and 5.4.2 and Fig. 5.1 (800 nm, 0.7 eV) and section 5.4.6 (averaging). The full text in Nature is paywalled and has not yet been compared. Statements from [4] were read in its arXiv version, arXiv:2105.03729.
