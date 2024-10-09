# diffPLOG2TROE

This repository provides a set of routines for managing Pressure Logarithmic (PLOG) reactions in
chemical kinetic models using the CHEMKIN format. Many standard solvers, including OpenFOAM and
older versions of CHEMKIN, do not natively support the PLOG formalism. To address this limitation,
the code offers several methods for replacing and handling PLOG reactions, ensuring compatibility
with these solvers and the CHEMKIN format. The implementation is written in Python, with most of the
computations performed using JAX.

Alternatives:
- [PLOG_replace](https://universityofgalway.ie/combustionchemistrycentre/softwaredownloads/),
developed at Galway university (never tested personally). Limited to the substitution of a plog with
a single modified Arrhenius expression.

## Install
1. Clone this repository using git.
    ```bash
    > git clone https://github.com/tdinelli/diffPLOG2TROE
    ```
2. Navigate into the directory containing the code and install the package by running the following commands:
    ```bash
    > python setup.py sdist bdist_wheel
    > pip install .
    ```

## Theoretical background
In certain circumstances, the rate of a chemical reaction may be dependent on both pressure and
temperature. CHEMKIN allows for two distinct types of such reactions: unimolecular/recombination
fall-off reactions and chemically activated bimolecular reactions. In general, the rate for
unimolecular/recombination fall-off reactions increases with increasing pressure, while the rate for
chemically activated bimolecular reactions decreases with increasing pressure. In both cases,
CHEMKIN provides a range of expressions that smoothly blend between high- and low-pressure limiting
rate expressions.

### Fall Off reactions
A falloff reaction is defined as a reaction that exhibits a first-order dependence on the
concentration of the reactant, $[M]$, at low pressure. However, as the concentration of $[M]$
increases, the reaction order with respect to $[M]$ approaches zero. This behavior is exemplified by
a three-body reaction. Dissociation and association reactions of polyatomic molecules frequently
manifest this characteristic. The most straightforward expression for the rate coefficient of a
falloff reaction is the Lindemann form. Let's consider as example the following reaction.
$$CH_3 + CH_3 (+M) \leftrightarrow C_2H_6 (+M)$$
There are multiple approaches to representing the rate expressions in this fall-off region. The most
straightforward approach is that proposed by Lindemann. Additionally, two other methodologies have
been developed that offer a more precise characterization of the fall-off region than the Lindemann
formulation. The CHEMKIN package accommodates all three of these forms as optional inputs. We
commence with an examination of the Lindemann approach. In order to obtain a pressure-dependent rate
expression, it is necessary to employ the Lindemann form for the rate coefficient, which blends the
Arrhenius rate parameters required for both the high- and low-pressure limiting cases. In the
Arrhenius form, the parameters are provided for the high-pressure limit ($k_{\infty}$) and the
low-pressure limit ($k_0$) as follows:

$$
k_{\infty} = A_{\infty}T^{\beta_{\infty}}e^{-\frac{Ea_{\infty}}{RT}}
$$
$$
k_{0} = A_{0}T^{\beta_{0}}e^{-\frac{Ea_{0}}{RT}}
$$

The rate constant at any given pressure is then calculated according to the following formula:

$$
k = k_{\infty} \left(\frac{P_{r}}{1+Pr}\right)F
$$

Where the reduced pressure $Pr$ is given by:

$$
Pr = \frac{k_{0}[M]}{k_{\infty}}
$$

Where $[M]$ is the concentration of the mixture, possibly including enhanced third-body
efficiencies.

### Chemically activated bimolecular reactions (CABR)
To illustrate a chemically activated bimolecular reaction, one might consider the reaction:

$$
CH_3+CH_3(+M) \leftrightarrow C_2H_5+H(+M)
$$

This reaction, which is endothermic, occurs via the same chemically activated $C_{2}H_{6}^{*}$
adduct as the recombination reaction
$$CH_3 + CH_3 (+M) \leftrightarrow C_2H_6 (+M)$$
. As the pressure rises, deactivating collisions of $C_{2}H_{6}^{*}$ with other molecules result in
an increase in the rate coefficient for $C_{2}H_{6}$ formation. Concurrently, these deactivating
collisions preclude the dissociation of $C_{2}H_{6}^{*}$ into $C_{2}H_{5} + H$, thereby causing this
rate coefficient to decrease with increasing pressure. Similarly, chemically-activated reactions are
described by a blending between a low-pressure and a high-pressure rate expression, as is the case
with falloff reactions. The distinction lies in the manner of expression, whereby the forward rate
constant is written as being proportional to the low-pressure rate constant.

$$
k = k_{0} \left(\frac{1}{1+Pr}\right)F
$$

### Available formalisms to compute $F$
|               | Formula for F                                                                                                                                                                                                                                                                                                                          | Notes                                                                                                                                                             |
| :------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Lindemann** | $F = 1$                                                                                                                                                                                                                                                                                                                                |                                                                                                                                                                   |
| **Troe**      | $log_{10}F = \frac{log_{10}Fcent}{1 + f_{1}^2}$<br/><br/>$Fcent = (1-A) exp\left(-\frac{T}{T_3}\right) + Aexp\left(-\frac{T}{T_1}\right) + exp\left(\frac{T_2}{T}\right)$<br/><br/>$f_1 = \frac{log_{10}Pr + c}{n - 0.14\left(log_{10}Pr + c\right)}$<br/><br/>$c = -0.4 - 0.67log_{10}Fcent$<br/><br/>$n = 0.75 - 1.27 log_{10}Fcent$ | Parameters to be specified in the CHEMKIN formalism are:<br/> $A$, $T_3$, $T_1$ and $T_2$ which  is optional.                                                     |
| **SRI**       | $X = \frac{1}{1 + log_{10}^{2}Pr}$<br/><br/>$F = d\left[a\times exp\left(-b/T\right) + exp\left(-T/c\right)\right]^{X} T^{e}$                                                                                                                                                                                                          | Parameters to be specified in the CHEMKIN formalism are:<br/> $a$, $b$, $c$, $d$ and $e$ keep in mind that the original fomulation correspond to $d=1$ and $e=0$. |


### Pressure logarithmic formulation (PLOG)
Introduced by J. Miller, as a generalized polynomial fitting for temperature and pressure
dependent kinetic constants, by defining the following expression for the kinetic constant:

$$
k \left(T, P_{i}\right) = \sum_{k=1}^{M} A_{i, k} T^{b_{i, k}} exp\left(-E_{act}^{i,
k}/(RT)\right), i=1, ..., Np, M \geq 1
$$

at a set of pressures, $P = P_{1}, P_{2}, ..., P_{Np}$. $M$ and $Np$ are user specified numbers. The
extrapolation is bounded by the two pressure limits, $P_{1}$ and $P_{Np}$. To calculate $k \left(T,
P_{i}\right)$ for any pressure, interpolate $logk$ as a linear function of $logP$. If $P$ is
between $P_{i}$ and $P_{i+1}$ for any temperature a rate constant can be find from:

$$
logk \left(T, P\right) = logk\left(T, P_{i}\right) + \left(logP - logP_{i}\right) \frac{logk\left(T,
P_{i+1}\right) - logk\left(T, P_{i}\right)}{logP_{i+1} - logP_{i}}
$$

## TODO

Finish and publish code documentation.
