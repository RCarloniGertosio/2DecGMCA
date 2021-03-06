import numpy as np
import matplotlib.pyplot as plt
import ffttools as fftt


# Miscellaneous

def mad(X=0, M=None):
    """Compute median absolute estimator.

    Parameters
    ----------
    X: np.ndarray
        data
    M: np.ndarray
        mask with the same size of x, optional

    Returns
    -------
    float
        mad estimate
        """
    if M is None:
        return np.median(abs(X - np.median(X))) / 0.6735
    xm = X[M == 1]
    return np.median(abs(xm - np.median(xm))) / 0.6735


def geo_mean(X, axis=0):
    """Compute geometric mean.

    Parameters
    ----------
    X: np.ndarray
        data
    axis: int or tuple
        int or tuple of int, axis or axes along which the means are computed

    Returns
    -------
    float
        geometric mean
    """
    return np.exp(np.mean(np.log(X), axis=axis))


def geo_std(X, axis=0, ddof=1):
    """Compute geometric standard deviation.

    Parameters
    ----------
    X: np.ndarray
        data
    axis: int or tuple
        int or tuple of int, axis or axes along which the standard deviations are computed
    ddof: int
        means delta degrees of freedom

    Returns
    -------
    float
        geometric standard deviation
    """
    return np.exp(np.std(np.log(X), axis=axis, ddof=ddof))


def whiten(Y):
    """Whiten a matrix.

    Parameters
    ----------
    Y: np.ndarray
        input matrix

    Returns
    -------
    (np.ndarray, np.ndarray)
        whitened matrix,
        whitening operator
    """
    U, s, _ = np.linalg.svd(Y, full_matrices=False)

    W = np.diag(1 / np.sqrt(s)) @ U.T

    Y_white = W @ Y

    return Y_white, W


# Generate problem

def generate_problem(n=4, m=8, size=256, cutmin=None, cutmax=None, nscales=3, sparseLvl=2., nbIt=25, condn=2., 
                     max0s=None, minResol=None, maxResol=None, infResol=False, snr=10, verb=0):
    """Generate a synthetic DBSS problem.

    Generates the data for a DBSS problem:
        Y[i,:] = H[i,:] * (A @ S[i,:]) + N[i,:]
    where H is the convolution kernels in the direct space, @ denotes the matrix product and * denotes the
    convolution product.

    Parameters
    ----------
    n: int
        number of sources
    m: int
        number of observations
    size: int
        size of the sources
    cutmin: int
        source generation: frequency at which the band-limiting filter starts to cut (default: int(size/6))
    cutmax: int
        source generation: frequency above which the band-limiting filter is 0 (default: max(minResol, size/2))
    nscales: int
        source generation: number of detail scales
    sparseLvl: float
        source generation: desired sparsity level on the wavelet domain (k*mad per scale)
    nbIt: int
        source generation: number of iterations
    condn: float
        mixing matrix: desired condition number
    max0s: int
        mixing matrix: maximum number of zeros (default: no condition)
    minResol: float
        filters: fwhm in Fourier space of the worse-resolved observation (default: int(size/4))
    maxResol: float
        filters: fwhm in Fourier space of the best-resolved observation (default: max(minResol, size/2))
    infResol: bool
        filters: the observations are not convolved, overrides minResol and maxResol
    snr: float
        observations: SNR in dB
    verb: int
        verbosity level

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        mixing matrix ((m,n) float array),
        sources ((n,p) float array),
        filters in Fourier domain ((m,p) float array),
        noise ((m,n) float array),
        observations ((m,p) float array),
        noised observations ((m,p) float array)

    Example
    -------
        A, S, Hl, X, N, Y = generateProblem(n=2, m=4, size=256, condn=2, snr=20, minResol=100, maxResol=384)
    """

    # Initialize some default parameters
    if minResol is None:
        minResol = np.int(size/10)
    if maxResol is None:
        maxResol = np.max(minResol, size/2)

    # Do some checks
    if minResol > maxResol:
        raise ValueError('minResol is greater than maxResol')
    if minResol < 0:
        raise ValueError('minResol is negative')
    if cutmin is not None and cutmax is not None and cutmin > cutmax:
        raise ValueError('cutmin is greater than cutmax')
    if cutmin is not None and cutmin < 0:
        raise ValueError('cutmin is negative')
    if cutmax is not None and cutmax > size/2:
        raise ValueError('cutmax is greater than size/2')
    if condn < 0:
        raise ValueError('condn is negative')

    if verb:
        print("Generating the sources...")
    S = generate_sources(n=n, size=size, cutmin=cutmin, cutmax=cutmax, sparseLvl=sparseLvl, nscales=nscales,
                         nbIt=nbIt)

    if verb:
        print("Generating the mixing matrix...")
    A = generate_mixmat(n=n, m=m, condn=condn, max0s=max0s)

    Hfft = generate_filt(m=m, size=size, minResol=minResol, maxResol=maxResol, infResol=infResol)

    if verb:
        print("Generating the observations...")
    X = fftt.convolve(A@S, Hfft)
    N = np.random.randn(m, size**2) * np.sqrt(np.mean(X ** 2) * 10 ** (-snr / 10))
    Y = X + N

    return A, S, Hfft, X, N, Y


def generate_sources(n=1, size=256, cutmin=None, cutmax=None, sparseLvl=2., nscales=3, nbIt=25, verb=0):
    """Generate synthetic sources.

    Parameters
    ----------
    n: int
        number of sources
    size: int
        size of the sources
    cutmin: int
        frequency at which the band-limiting filter starts to cut (default: int(size/2))
    cutmax: int
        frequency above which the band-limiting filter is 0 (default: max(minResol, size/2))
    nscales: int
        number of detail scales
    sparseLvl: float
        desired sparsity level on the wavelet domain (k*mad per scale)
    nbIt: int
        number of iterations
    verb: int
        verbosity level

    Returns
    -------
    np.ndarray
        (m,n) float array, mixing matrix
    """

    if cutmin is None:
        cutmin = np.int(size/6)
    if cutmax is None:
        cutmax = np.int(size/2)

    if verb > 0:
        print(" cutmin ", cutmin, " - cutmax ", cutmax)

    bl = fftt.getidealbeam(size, cutmin, cutmax)

    S = np.random.randn(n, size**2)
    Sfft = fftt.fft(S)
    Sfft = fftt.fftprod(Sfft, bl)  # band-limited

    for it in range(nbIt):
        if verb > 0:
            print("iteration: ", it+1)
        Swt = fftt.wt_trans(Sfft, nscales=nscales, fft_in=True)
        for i in range(n):
            for j in range(nscales+1):
                threshold = sparseLvl * mad(Swt[i, :, j])
                Swt[i, :, j] = (Swt[i, :, j]-threshold*np.sign(Swt[i, :, j]))*(abs(Swt[i, :, j]) > threshold)
        S = fftt.wt_rec(Swt)
        S = S * (S > 0)  # positivity but not guaranteed after band-limiting constraint
        Sfft = fftt.fft(S)
        Sfft = fftt.fftprod(Sfft, bl)  # band-limited

    if n == 1:
        Sfft = Sfft[0, :]

    return fftt.ifft(Sfft)*20  # multiply by 20 to have sources of approx. unit value


def generate_mixmat(n=2, m=4, condn=2., dcondn=1e-2, max0s=None, forceMax0s=False):
    """Generate a synthetic mixing matrix.

    Parameters
    ----------
    n: int
        number of sources
    m: int
        number of observations
    condn: float
        desired condition number
    dcondn: float
        condition number precision
    max0s: int
        maximum number of zeros (default: no condition)
    forceMax0s: bool
        if False, max0s is incremented every minute, to ensure the convergence of the function

    Returns
    -------
    np.ndarray
        (m,n) float array, mixing matrix
    """

    A = np.random.rand(m, n)
    A = A / np.maximum(1e-24, np.linalg.norm(A, axis=0))
    if max0s is None:
        max0s = n * m
    it = 0
    while True:
        it += 1
        if not forceMax0s and it >= 5e4:  # relax condition on max number of zeros
            it = 0
            max0s += 1
        try:
            U, d, V = np.linalg.svd(A)
        except np.linalg.LinAlgError:  # divergence, new A drawn
            A = np.random.rand(m, n)
            A = A / np.maximum(1e-24, np.linalg.norm(A, axis=0))
        else:
            d = d[-1] * ((d - d[-1]) * (condn - 1) / (d[0] - d[-1]) + 1)
            D = A * 0
            D[:n, :n] = np.diag(d)
            A = U @ D @ V.T
            for i in range(n):
                if sum(A[:, i] > 0) <= np.int(m / 2):  # if there are more negative numbers
                    A[:, i] = np.maximum(-A[:, i], 0)
                else:
                    A[:, i] = np.maximum(A[:, i], 0)
            A = A / np.maximum(1e-24, np.linalg.norm(A, axis=0))
            err = np.abs(np.linalg.cond(A) - condn)
            if err < dcondn or err > 1e10:
                if np.count_nonzero(A == 0) <= max0s:
                    return A
                A = np.random.rand(m, n)
                A = A / np.maximum(1e-24, np.linalg.norm(A, axis=0))


def generate_filt(m=8, size=256, minResol=None, maxResol=None, infResol=False):
    """Generate Gaussian-shaped filters in Fourier domain.

    Parameters
    ----------
    m: int
        number of observations
    size: int
        size of the sources
    minResol: float
        filters: fwhm in Fourier space of the worse-resolved observation (default: int(size/4))
    maxResol: float
        filters: fwhm in Fourier space of the best-resolved observation (default: max(minResol, size/2))
    infResol: bool
        filters: the observations are not convolved, overrides minResol and maxResol

    Returns
    -------
    np.ndarray
         (m,p) float array, convolution kernels in Fourier domain
    """

    if infResol:
        return np.ones((m, size**2))

    if minResol is None:
        minResol = int(size/4)
    if maxResol is None:
        maxResol = int(max(minResol, size/2))

    resol = np.linspace(minResol, maxResol, m)
    sigma2 = resol**2/(2*np.log(2))
    Hfft = np.zeros((m, size**2))
    xx, yy = np.meshgrid(np.arange(0, size), np.arange(0, size))
    for observation in range(m):
        Hfft[observation, :] = np.exp(-((xx-size/2)**2+(yy-size/2)**2)/(2*sigma2[observation])).flatten()
    return Hfft


# View data

def view(X, X2=None, log=False, title='', minimum=None, maximum=None, cbar=True, newfig=True):
    """Plot one or more arrays. If input is complex, plots the amplitude.

    Parameters
    ----------
    X: np.ndarray
        (p,) or (n,p) float array,
    X2: np.ndarray
        (p,) or (n,p) float array, second array or stack of n arrays, optional
    log: bool
        logarithmic scale
    title: str
        title of the plots
    minimum: float
        minimum range value (default: min(maps, maps2))
    maximum: float
        maximum range value (default: max(maps, maps2))
    cbar: bool
        show color bar
    newfig: bool
        create a new figure

    Returns
    -------
    None
    """
    if X.dtype == 'complex' or (X2 is not None and X2.dtype == 'complex'):
        X = np.abs(X)
        if X2 is not None:
            X2 = np.abs(X2)

    if len(np.shape(X)) == 1:
        X = np.expand_dims(X, axis=0)
        if X2 is not None:
            X2 = np.expand_dims(X2, axis=0)
    size = np.int(np.sqrt(np.shape(X)[1]))
    if minimum is None:
        minimum = np.min(X)
        if X2 is not None:
            minimum = np.min([minimum, np.min(X2)])
    if maximum is None:
        maximum = np.max(X)
        if X2 is not None:
            maximum = np.max([maximum, np.max(X2)])
    if not log:
        def f(x):
            return x
    else:
        def f(x):
            return np.log10(x - minimum + 1)
    for i in range(np.shape(X)[0]):
        if newfig:
            plt.figure()
        plt.imshow(np.reshape(f(X[i, :]), (size, size)), vmin=f(minimum), vmax=f(maximum))
        if title != '':
            plt.title(title)
        if cbar:
            plt.colorbar()
        if X2 is not None:
            plt.figure()
            plt.imshow(np.reshape(f(X2[i, :]), (size, size)), vmin=f(minimum), vmax=f(maximum))
            if title != '':
                plt.title(title)
            if cbar:
                plt.colorbar()


def view_data(A, S, Y):
    """View input data.

    Parameters
    ----------
    A: np.ndarray
        (m,n) float array, mixing matrix
    S: np.ndarray
        (n,p) float array, sources
    Y: np.ndarray
        (m,p) float array, observations

    Returns
    -------
    None
    """

    view(S)

    print("A =\n", A)

    view(Y)

    plt.pause(0.1)


def view_interesting_channels(Y, A):
    """View observations with active sources.

    Parameters
    ----------
    Y: np.ndarray
        (m,p) float array, observations
    A: np.ndarray
        (m,n) float array, mixing matrix

    Returns
    -------
    None
    """

    interesting_channels = np.where(np.sum(A, axis=1) >= 0.05*np.shape(A)[1])[0]
    view(Y[interesting_channels, :])


def asses_solution(A0, S0, A, S, corrPerm=False, perScale=False, nscales=3, viewData=True):
    """Print the performance metrics of a solution and asses visually the sources.

    Parameters
    ----------
    A0: np.ndarray
        (m,n) float array, ground truth mixing matrix
    S0: np.ndarray
        (n,p) float array, ground truth sources
    A: np.ndarray
        (m,n) float array, estimated mixing matrix
    S: np.ndarray
        (n,p) float array, estimated sources
    corrPerm: bool
        correct permutation of A and S (in-place updates)
    perScale: bool
        calculate NMSE per wavelet scale
    nscales: int
        number of wavelet detail scales
    viewData: bool
        compare the sources visually

    Returns
    -------
    None
    """

    if not corrPerm:
        A = A.copy()
        S = S.copy()

    if not perScale:
        CA, NMSE = evaluate(A0, S0, A, S, corrPerm=True, perScale=perScale, nscales=nscales)
        print('CA : %.2f | NMSE: %.2f' % (CA, NMSE))
    else:
        CA, NMSE, NMSEScale = evaluate(A0, S0, A, S, corrPerm=True, perScale=perScale, nscales=nscales)
        print('CA : %.2f | overall NMSE: %.2f' % (CA, NMSE))
        for i in range(nscales):
            print('NMSE at detail scale %i: %.2f' % (i+1, NMSEScale[i]))
        print('NMSE at coarse scale: %.2f' % NMSEScale[nscales])

    if viewData:
        view(S0, S)
        view(np.abs(S0-S), log=True)
        plt.pause(0.1)


# Metrics

def evaluate(A0, S0, A, S, corrPerm=False, perScale=False, nscales=3, S0wt=None):
    """Computes the NMSE and the CA.

    Parameters
    ----------
    A0: np.ndarray
        (m,n) float array, ground truth mixing matrix
    S0: np.ndarray
        (n,p) float array, ground truth sources
    A: np.ndarray
        (m,n) float array, estimated mixing matrix
    S: np.ndarray
        (n,p) float array, estimated sources
    corrPerm: bool
        correct permutation of A and S (in-place updates)
    perScale: bool
        calculate NMSE per wavelet scale
    nscales: int
        number of wavelet detail scales
    S0wt: np.ndarray
        (m,n,nscales+1) float array, wavelet transform of S0, optional (to accelerate)

    Returns
    -------
    (float,float) or (float,float,np.ndarray)
        CA,
        NMSE,
        NMSE per scale if perScale ((nscales+1,) float array)
    """

    if not corrPerm:
        A = A.copy()
        S = S.copy()

    n = np.shape(A0)[1]

    corr_perm(A0, S0, A, S, inplace=True)

    CA = -10 * np.log10(np.mean(np.abs(np.dot(np.linalg.pinv(A), A0) - np.eye(n))))
    NMSE = -10 * np.log10(np.sum((S0-S)**2)/np.sum(S0**2))

    if not perScale:
        return CA, NMSE

    if S0wt is not None:
        nscales = np.shape(S0wt)[2]-1
    else:
        S0wt = fftt.wt_trans(S0, nscales=nscales)
    Swt = fftt.wt_trans(S, nscales=nscales)
    NMSEScale = -10 * np.log10(np.sum((S0wt-Swt)**2, axis=(0, 1))/np.sum(S0**2, axis=(0, 1)))

    return CA, NMSE, NMSEScale


def corr_perm(A0, S0, A, S, inplace=False, optInd=False):
    """Correct the permutation of the solution.

    Parameters
    ----------
    A0: np.ndarray
        (m,n) float array, ground truth mixing matrix
    S0: np.ndarray
        (n,p) float array, ground truth sources
    A: np.ndarray
        (m,n) float array, estimated mixing matrix
    S: np.ndarray
        (n,p) float array, estimated sources
    inplace: bool
        in-place update of A and S
    optInd: bool
        return permutation

    Returns
    -------
    None or np.ndarray or (np.ndarray,np.ndarray) or (np.ndarray,np.ndarray,np.ndarray)
        A (if not inplace),
        S (if not inplace),
        ind (if optInd)
    """

    A0 = A0.copy()
    S0 = S0.copy()
    if not inplace:
        A = A.copy()
        S = S.copy()

    n = np.shape(A0)[1]

    for i in range(0, n):
        S[i, :] *= (1e-24 + np.linalg.norm(A[:, i]))
        A[:, i] /= (1e-24 + np.linalg.norm(A[:, i]))
        S0[i, :] *= (1e-24 + np.linalg.norm(A0[:, i]))
        A0[:, i] /= (1e-24 + np.linalg.norm(A0[:, i]))

    try:
        diff = abs(np.dot(np.linalg.inv(np.dot(A0.T, A0)), np.dot(A0.T, A)))
    except np.linalg.LinAlgError:
        diff = abs(np.dot(np.linalg.pinv(A0), A))
        print('Warning! Pseudo-inverse used.')

    ind = np.arange(0, n)

    for i in range(0, n):
        ind[i] = np.where(diff[i, :] == max(diff[i, :]))[0][0]

    A[:] = A[:, ind.astype(int)]
    S[:] = S[ind.astype(int), :]

    for i in range(0, n):
        p = np.sum(S[i, :] * S0[i, :])
        if p < 0:
            S[i, :] = -S[i, :]
            A[:, i] = -A[:, i]

    if inplace and not optInd:
        return None
    elif inplace and optInd:
        return ind
    elif not optInd:
        return A, S
    else:
        return A, S, ind


def nmse(S0, S):
    """Compute the normalized mean square error (NMSE) in dB.

    Parameters
    ----------
    S0: np.ndarray
        (n,p) float array, ground truth sources
    S: np.ndarray
        (n,p) float array, estimated sources

    Returns
    -------
    float
        NMSE (dB)
    """

    return -10 * np.log10(np.sum((S0-S)**2)/np.sum(S0**2))


def ca(A0, A):
    """Compute the criterion on A (CA) in dB.

    Parameters
    ----------
    A0: np.ndarray
        (m,n) float array, ground truth mixing matrix
    A: np.ndarray
        (m,n) float array, estimated mixing matrix

    Returns
    -------
    float
        CA (dB)
    """

    return -10 * np.log10(np.mean(np.abs(np.dot(np.linalg.pinv(A), A0) - np.eye(np.shape(A0)[1]))))
