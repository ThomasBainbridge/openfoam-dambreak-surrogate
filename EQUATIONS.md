# Governing Equations

The OpenFOAM `interFoam` solver uses a Volume of Fluid (VOF) method to track the water–air interface. The following equations govern the two-phase flow.

---

## Volume fraction transport

$$\frac{\partial \alpha}{\partial t} + \nabla \cdot (\alpha \mathbf{U}) + \nabla \cdot \left[\alpha(1-\alpha)\mathbf{U}_r\right] = 0 \tag{1}$$

$\alpha$ is the water volume fraction ($\alpha = 1$ in water, $\alpha = 0$ in air). The third term is the interface-compression term unique to `interFoam`, active only at the interface where $\alpha(1-\alpha) \neq 0$, which keeps the interface sharp without explicit interface reconstruction.

---

## Mixture properties

$$\rho = \alpha\rho_w + (1-\alpha)\rho_a \tag{2}$$

$$\mu = \alpha\mu_w + (1-\alpha)\mu_a \tag{3}$$

Density and dynamic viscosity are computed by linear interpolation between the water and air values.

---

## Continuity

$$\nabla \cdot \mathbf{U} = 0 \tag{4}$$

Incompressibility is assumed for both phases.

---

## Momentum

$$\frac{\partial (\rho \mathbf{U})}{\partial t} + \nabla \cdot (\rho \mathbf{U} \otimes \mathbf{U}) = -\nabla p + \nabla \cdot \left[\mu \left(\nabla \mathbf{U} + \nabla \mathbf{U}^T\right)\right] + \rho \mathbf{g} + \mathbf{f}_\sigma \tag{5}$$

Standard incompressible Navier–Stokes for the mixture, with gravity and surface tension included as body forces.

---

## Surface tension force

$$\mathbf{f}_\sigma = \sigma \kappa \nabla \alpha \tag{6}$$

Continuum Surface Force (CSF) model (Brackbill et al., 1992). $\sigma$ is the surface tension coefficient and $\kappa$ is the interface curvature, computed from the volume fraction field.

---

## OpenFOAM modified pressure

$$p_{rgh} = p - \rho \mathbf{g} \cdot \mathbf{x} \tag{7}$$

`interFoam` solves for the modified pressure $p_{rgh}$, which subtracts the hydrostatic contribution. This avoids resolving large hydrostatic pressure gradients numerically and improves solver conditioning.

---

## Symbols

| Symbol | Description |
|---|---|
| $\alpha$ | Water volume fraction |
| $\mathbf{U}$ | Mixture velocity field |
| $\mathbf{U}_r$ | Interface-compression velocity |
| $\rho$ | Mixture density |
| $\mu$ | Mixture dynamic viscosity |
| $\rho_w,\ \rho_a$ | Water and air densities |
| $\mu_w,\ \mu_a$ | Water and air dynamic viscosities |
| $p$ | Pressure |
| $p_{rgh}$ | OpenFOAM modified pressure |
| $\mathbf{g}$ | Gravitational acceleration vector |
| $\mathbf{x}$ | Position vector |
| $\sigma$ | Surface tension coefficient |
| $\kappa$ | Interface curvature |
| $\mathbf{f}_\sigma$ | Surface tension body force |
