# diffPLOG2TROE

This repository provides a set of routines for managing Pressure Logarithmic (PLOG) reactions in
chemical kinetic models using the CHEMKIN format. Many standard solvers, including OpenFOAM and
older versions of CHEMKIN, do not natively support the PLOG formalism. To address this limitation,
the code offers several methods for replacing and handling PLOG reactions, ensuring compatibility
with these solvers and the CHEMKIN framework. The implementation is written in Python, with most
computations performed using JAX, selected for its speed and full differentiability.

Alternatives:
- [PLOG_replace](https://universityofgalway.ie/combustionchemistrycentre/softwaredownloads/),
developed at Galway university (never tested personally).

## Theoretical background (:))
### Fall Off reactions
A falloff reaction is one that has a rate that is first-order in $[M]$ at low pressure, like a
three-body reaction, but becomes zero-order in $[M]$ as $[M]$ increases. Dissociation/association
reactions of polyatomic molecules often exhibit this behavior. The simplest expression for the rate
coefficient for a falloff reaction is the Lindemann form

### Chemically activated bimolecular reactions (CABR)

Introduced by J. Miller [2010](),
as a generalized polynomial fitting for temperature and pressure dependent kinetic constants, by
defining the following expression for the kinetic constant:

$$
k \left(T, P_{i}\right) = \sum_{k=1}^{M} A_{i, k} T^{b_{i, k}} exp\left(-E_{act}^{i,
k}/(RT)\right), i=1, ..., Np, M \geq 1
$$

at a set of pressures, $P = P_{1}, P_{2}, ..., P_{Np}$. $M$ and $Np$ are user specified numbers. The
extrapolation is bounded by the two pressure limits, $P_{1}$ and $P_{Np}$. To calculate $k \left(T,
P_{i}\right)$ for any pressure, interpolate $log\:k$ as a linear function of $log\:P$. If $P$ is
between $P_{i}$ and $P_{i+1}$ for any temperature a rate constant can be find from:

$$
log\:k \left(T, P\right) = log\:k\left(T, P_{i}\right) + \left(log\:P - log\:P_{i}\right)
\frac{log\:k\left(T, P_{i+1}\right) - log\:k\left(T, P_{i}\right)}{log\:P_{i+1} - log\:P_{i}}
$$

The PLOG formalism was introduced as a further extension to already existing formalisms to fit the
broadened pressure fall-off effect. Such as the broaden factor introduced by Troe []():

$$
k \left(T, Pr\right) = k_{\infty}\frac{Pr}{1+Pr}
$$

Where $Pr$ is the reduced pressure and is defined as follow: $Pr = \frac{k_{0}[M]}{k_{\infty}}$.
