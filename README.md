# DecGMCA

DecGMCA (Deconvolution Generalized Morphological Component Analysis) is an algorithm aiming at solving joint Deconvolution and Blind Source Separation (DBSS) problems.

## Contents
1. [Introduction](#intro)
1. [Procedure](#procedure)
1. [Getting Started](#start)
1. [Parameters](#param)
1. [Example](#example)
1. [Authors](#authors)

<a name="intro"></a>
## Introduction

Let us consider the imaging model:
> ![equation](https://latex.codecogs.com/gif.latex?%5Cmathbf%7BX%7D%20%3D%20%5Ctext%7BH%7D%20%28%5Cmathbf%7BA%7D%20%5Cmathbf%7BS%7D%29%20&plus;%20%5Cmathbf%7BN%7D),

where:
- ![equation](https://latex.codecogs.com/gif.latex?%5Cmathbf%7BX%7D%20%5Cin%20%5Cmathbb%7BR%7D%5E%7BN_c%20%5Ctimes%20N_p%7D) are the ![equation](https://latex.codecogs.com/gif.latex?N_c) multiwavelength data, flattened and stacked in a matrix,
- ![equation](https://latex.codecogs.com/gif.latex?%5Ctext%7BH%7D) is measurement operator,
- ![equation](https://latex.codecogs.com/gif.latex?%5Cmathbf%7BA%7D%20%5Cin%20%5Cmathbb%7BR%7D%5E%7BN_c%20%5Ctimes%20N_s%7D) is the mixing matrix,
- ![equation](https://latex.codecogs.com/gif.latex?%5Cmathbf%7BS%7D%20%5Cin%20%5Cmathbb%7BR%7D%5E%7BN_s%20%5Ctimes%20N_p%7D) are the ![equation](https://latex.codecogs.com/gif.latex?N_s) sources, flattened and stacked in a matrix,
- ![equation](https://latex.codecogs.com/gif.latex?%5Cmathbf%7BN%7D%20%5Cin%20%5Cmathbb%7BR%7D%5E%7BN_c%20%5Ctimes%20N_p%7D) is the noise.

In the following, the measurement operator is assumed channel-dependant, linear and isotropic. Thus, for channel ![equation](https://latex.codecogs.com/gif.latex?%5Cnu%20%5Cin%20%5B1%2C%20N_c%5D):
> ![equation](https://latex.codecogs.com/gif.latex?%5Cmathbf%7BX%7D_%7B%5Cnu%7D%20%3D%20%28%5Cmathbf%7BA%7D_%7B%5Cnu%7D%20%5Cmathbf%7BS%7D%29*%5Cmathbf%7BH%7D_%7B%5Cnu%7D%20&plus;%20%5Cmathbf%7BN%7D_%7B%5Cnu%7D),

where ![equation](https://latex.codecogs.com/gif.latex?*) denotes the convolution product. The equation can be simplified in Fourier domain:
> ![equation](https://latex.codecogs.com/gif.latex?%5Cmathbf%7B%5Chat%7BX%7D%7D%5Ep_%5Cnu%20%3D%20%5Cmathbf%7B%5Chat%7BH%7D%7D%5Ep_%5Cnu%20%5Cmathbf%7BA%7D_%5Cnu%20%5Cmathbf%7B%5Chat%7BS%7D%7D%5Ep%20&plus;%20%5Cmathbf%7B%5Chat%7BN%7D%7D%5Ep_%5Cnu)

The sources are assumed to be sparse in the starlet representation ![equation](https://latex.codecogs.com/gif.latex?%5Cmathbf%7B%5CPhi%7D). DecGMCA aims at minimizing the following objective function with respect to A and S:
> ![equation](https://latex.codecogs.com/gif.latex?%5Csum%5Climits_%7Bp%2C%5Cnu%7D%20%5Cleft%20%5C%7C%5Cmathbf%7B%5Chat%7BX%7D%7D%5Ep_%5Cnu%20-%20%5Cmathbf%7B%5Chat%7BH%7D%7D%5Ep_%5Cnu%20%5Cmathbf%7BA%7D_%5Cnu%20%5Cmathbf%7B%5Chat%7BS%7D%7D%5Ep%5Cright%20%5C%7C%5E2_2%20%26plus%3B%20%5Cleft%20%5C%7C%20%5Cmathbf%7B%5CLambda%7D%20%5Codot%20%5Cleft%28%20%5Cmathbf%7BS%7D%20%5Cmathbf%7B%5CPhi%7D%5ET%5Cright%29%5Cright%20%5C%7C_%7B%5Cell_1%7D%20%26plus%3B%20%5Cchi_%5Cmathcal%7BO%7D%5Cleft%28%5Cmathbf%7BA%7D%5Cright%29),

where ![equation](https://latex.codecogs.com/gif.latex?%5Codot) denotes the element-wise product, ![equation](https://latex.codecogs.com/gif.latex?%5Cmathbf%7B%5CLambda%7D) are the sparsity regularization parameters and ![equation](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BO%7D) is the oblique ensemble. Moreover, non-negativity constraints on ![equation](https://latex.codecogs.com/gif.latex?%5Cmathbf%7BA%7D) and/or ![equation](https://latex.codecogs.com/gif.latex?%5Cmathbf%7BS%7D) may be added.

<a name="procedure"></a>
## Procedure

The algorithm is built upon a sparsity-enforcing projected alternate least-square procedure, which updates iteratively the sources and the mixing matrix. In a nutshell, when either ![equation](https://latex.codecogs.com/gif.latex?%5Cmathbf%7BA%7D) or ![equation](https://latex.codecogs.com/gif.latex?%5Cmathbf%7BS%7D) is updated, a first least-squares estimate is computed by minimizing the data-fidelity term. This step is then followed by the application of the proximal operator of the corresponding regularization term.

In contrast to standard BSS problems, the least-square update of the sources is not necessarily stable with respect to noise. Thus, an extra Tikhonov regularization is added.

The separation is comprised of two stages. The first stage estimates a first guess of the mixing matrix and the sources (**warm-up**); it is required to provide robustness with respect to the initial point. The second stage refines the separation by employing a more precise Tikhonov regularization strategy (**refinement**). Lastly, the sources are improved during a finale step with the output mixing matrix.

<a name="start"></a>
## Getting Started

### Requirements

DecGMCA has been developed with Python 3.7.

### Prerequisites

The following Python libraries need to be installed to run the code:
- Numpy,
- Matplotlib [optional].

### DecGMCA class

DecGMCA is implemented in a class. The data and the parameters of the separation are provided at the initialization of the object. The separation is performed by running the method `run`. The results are stored in the attributes `S` and `A`.

<a name="param"></a>
## Parameters

Below is the list of the attributes of the DecGMCA class.

| Parameter | Type                            | Information                                                                                | Default value            |
|-----------|---------------------------------|--------------------------------------------------------------------------------------------|--------------------------|
| `X`       | (m,p) float numpy.ndarray       | input data in Healpix representation, each row corresponds to an observation               | N/A                      |
| `Hfft`    | (m,p) float numpy.ndarray       | convolution kernels in Fourier domain (flattened & with 0-frequency shifted to the center) | N/A                      |
| `M`       | (p,) float numpy.ndarray        | mask                                                                                       | None                     |
| `n`       | int                             | number of sources to be estimated                                                          | N/A                      |
| `AInit`   | (m,n) float numpy.ndarray       | initial value for the mixing matrix. If None, PCA-based initialization                     | None                     |
| `nnegA`   | bool                            | non-negativity constraint on A                                                             | False                    |
| `nnegS`   | bool                            | non-negativity constraint on S                                                             | False                    |
| `nneg`    | bool                            | non-negativity constraint on A and S. If not None, overrides nnegA and nnegS               | None                     |
| `wuStrat` | int                             | warm-up Tikhonov regularization strategy (**TO DELETE**)                                   | 3                        |
| `minWuIt` | int                             | minimum number of iterations at warm-up                                                    | 100                      |
| `c_wu`    | float or (2,) numpy.ndarray     | Tikhonov regularization hyperparameter at warm-up                                          | 0.5                      |
| `c_ref`   | float                           | Tikhonov regularization hyperparameter at refinement                                       | 0.5                      |
| `cwuDec`  | int                             | number of iterations for the decrease of c_wu (if c_wu is an array)                        | minWuIt/2                |
| `nStd`    | float                           | noise standard deviation                                                                   | N/A                      |
| `useMad`  | bool                            | True: noise std estimated with MAD, False: noise std estimated analytically                | False                    |
| `nscales` | int                             | number of starlet detail scales                                                            | 3                        |
| `k`       | float                           | parameter of the k-std thresholding                                                        | 3                        |
| `K_max`   | float                           | maximal L0 norm of the sources. Being a percentage, it should be between 0 and 1           | 0.5                      |
| `L1`      | bool                            | if False, L0 rather than L1 penalization                                                   | True                     |
| `doRw`    | bool                            | do L1 reweighing during refinement (only if L1 penalization)                               | True                     |
| `eps`     | (3,) float numpy.ndarray        | stopping criteria of (1) the warm-up, (2) the refinement and (3) the finale refinement ofS | [0.2, 0.5, 0.5]          |
| `verb`    | int                             | verbosity level, from 0 (mute) to 5 (most talkative)                                       | 0                        |
| `S0`      | (n,p) float numpy.ndarray       | ground truth sources (for testing purposes)                                                | None                     |
| `A0`      | (m,n) float numpy.ndarray       | ground truth mixing matrix (for testing purposes)                                          | None                     |
| `iSNR0`   | (n,lmax+1) float numpy.ndarray, | regularization parameters for strategy #4 (for testing purposes)                           | deduced from S0 and nStd |

<a name="example"></a>
## Example

Perform a DBSS on the data `X` with 4 sources.

```python
decgmca = DecGMCA(X, Hfft, n=4, k=3, K_max=0.5, nscales=3, wuStrat=3, c_wu=numpy.array([5, 0.5]), c_ref=0.5, nStd=1e-7)
decgmca.run()
S = decgmca.S.copy()  # estimated sources
A = decgmca.A.copy()  # estimated mixing matrix
```

<a name="authors"></a>
## Authors
