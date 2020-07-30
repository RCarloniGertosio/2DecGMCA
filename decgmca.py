import numpy as np
import ffttools as fftt
import utils


class DecGMCA:
    """DecGMCA joint deconvolution and blind source separation algorithm.

    The input data are assumed to be flattened. The filters are assumed to have the zero-frequency shifted to the
    center and to be flattened.

    Example
    --------
    decgmca = SDecGMCA(X, Hl, n=2, k=3, K_max=0.5, nscales=3, wuStrat=3, c_wu=numpy.array([5, 0.5]), c_ref=0.5,
    nStd=1e-7)\n
    decgmca.run()\n
    S = decgmca.S.copy()\n
    A = decgmca.A.copy()
    """

    def __init__(self, X, Hfft, M=None, n=4, AInit=None, nnegA=False, nnegS=False, nneg=None, wuStrat=3, minWuIt=100,
                 c_wu=4e-1, c_ref=2e-1, cwuDec=None, nStd=0, useMad=False, nscales=3, k=3, K_max=0.5, L1=True,
                 doRw=True, eps=None, verb=0, S0=None, A0=None, iSNR0=None):
        """Initialize object.

        Parameters
        ----------
        X : np.ndarray
            (m,p) float array, input data, each row corresponds to an observation
        Hfft : np.ndarray
            (m,p) float array, convolution kernels in Fourier domain, with zero-frequency shifted to the center and
            flattened
        M : np.ndarray
            (p,) float array, mask, optional
        n : int
            number of sources to be estimated
        AInit : np.ndarray
            (m,n) float array, initial value for the mixing matrix. If None, PCA-based initialization
        nnegA : bool
            non-negativity constraint on A
        nnegS : bool
            non-negativity constraint on S
        nneg : bool
            non-negativity constraint on A and S. If not None, overrides nnegA and nnegS
        wuStrat : int
            warm-up Tikhonov regularization strategy:
                * 1 : c
                * 2 : c*sigma_max(A.T@diag(Hl**2)@A)
                * 3 : max(0, c-sigma_min(A.T@diag(Hl**2)@A)/(sigma_min(A.T@A)+1e-2))
                * 4 : c/SNR, with SNR = spectraS/nStdFFT where spectraS calculated from a previous estimation of the
                sources)
        minWuIt : int
            minimum number of iterations at warm-up
        c_wu : float or np.ndarray
            float or (2,) float array, Tikhonov regularization hyperparameter at warm-up
        c_ref : float
            Tikhonov regularization hyperparameter at refinement (with regularization strategy #4)
        cwuDec : int
            number of iterations for the decrease of c_wu (if c_wu is an array). If None: minWuIt/2
        nStd : float
            noise standard deviation
        useMad : bool
            True: noise std estimated with MAD, False: noise std estimated analytically)
        nscales : int
            number of detail scales
        k : float
            parameter of the k-std thresholding
        K_max : float
            maximal L0 norm of the sources. Being a percentage, it should be between 0 and 1
        L1 : bool
            if set to 0, L0 rather than L1 penalization
        doRw : bool
            do L1 reweighing during refinement (only if L1 penalization)
        eps : np.ndarray
            (3,) float array, stopping criteria of (1) the warm-up, (2) the refinement and (3) the finale refinement of
            S
        verb : int
            verbosity level, from 0 (mute) to 5 (most talkative)
        S0 : np.ndarray
            (n,p) float array, ground truth sources (for testing purposes)
        A0 : np.ndarray
            (m,n) float array, ground truth mixing matrix (for testing purposes)
        iSNR0 : np.ndarray
            (n,p) float array, regularization parameters for strategy #4 (for testing purposes). Default: deduced from
            S0 and nStd

        Returns
        -------
        DecGMCA
        """

        # Initialize given attributes
        self.X = X
        self.Hfft = Hfft.copy()
        if len(np.shape(self.Hfft)) == 1:  # if Hl is a 1D array, expand it to a 2D array
            self.Hfft = np.expand_dims(self.Hfft, axis=0)
            self.Hfft = np.tile(self.Hfft, [self.m, 1])
        if M is None:
            self.M = np.ones(np.shape(X)[-1])
        else:
            self.M = M.copy()
        self.n = n
        self.AInit = AInit.copy()
        self.nnegA = nnegA
        self.nnegS = nnegS
        if nneg:
            self.nnegA = True
            self.nnegS = True
        self.wuStrat = wuStrat
        self.minWuIt = minWuIt
        self.c_wu = c_wu
        self.c_ref = c_ref
        if cwuDec is None:
            self.cwuDec = int(self.minWuIt/2)
        else:
            self.cwuDec = cwuDec
        self.nStd = nStd
        self.useMad = useMad
        self.nscales = nscales
        self.k = k
        self.K_max = K_max
        self.L1 = L1
        self.doRw = doRw
        if eps is None:
            self.eps = [1e-2, 1e-5, 1e-5]
        else:
            self.eps = eps.copy()
        self.verb = verb
        self.S0 = S0.copy()
        self.A0 = A0.copy()
        self.iSNR0 = iSNR0.copy()

        # Initialize deduced attributes
        self.m = np.shape(self.X)[0]                        # number of observations
        self.p = np.shape(self.X)[1]                        # number of pixels
        self.size = np.int(np.sqrt(self.p))
        self.supp = np.int(np.sum(self.M))                  # support of the mask
        self.wt_filters = fftt.get_wt_filters(size=self.size, nscales=self.nscales)  # wavelet filters
        self.Xfft = fftt.fft(self.X)                        # data in Fourier domain
        self.Xfft_det = fftt.fftprod(self.Xfft, 1 - self.wt_filters[:, -1])  # data in Fourier domain w/ no coarse scale
        self.nStdFFT = self.nStd * np.sqrt(self.p)          # noise std in the Fourier domain
        # For testing purposes:
        if self.S0 is not None and iSNR0 is None:
            self.iSNR0 = self.nStdFFT ** 2 / (np.abs(fftt.fft(self.S0))+1e-10)**2 * self.supp / self.p
        if self.S0 is not None:
            self.S0wt = fftt.wt_trans(self.S0, nscales=self.nscales)         # true sources in the wavelet domain

        # Initialize other attributes
        self.S = np.zeros((self.n, self.p))                         # current estimation of the sources
        self.Sfft = np.zeros((self.n, self.p), dtype=complex)       # current estimation of the sources in Fourier dom.
        self.Sfft_det = np.zeros((self.n, self.p), dtype=complex)   # current est. of the detail scales of S
        # TODO memory optimization: if self.nnegS, self.Sfft==self.Sfft_det during the separation
        self.Swtrw = np.zeros((self.n, self.p, self.nscales))       # weights for the l1 reweighing
        self.A = np.zeros((self.m, self.n))                         # current estimation of the mixing matrix
        self.invOpSp = np.zeros((self.p, self.n))                   # spectra of the inverse operators
        self.lastWuIt = None                                        # last warm-up iteration
        self.lastRefIt = None                                       # last refinement iteration
        self.nmse = None
        self.ca = None
        self.nmseScales = None
        self.aborted = False                                        # True if last resolution has been aborted

    def __str__(self):
        res = '\n'
        res += '  - Number of sources: %i\n' % self.n
        if self.nnegA or self.nnegS:
            res += '  - Non-negativity constraint on '
            if self.nnegA and self.nnegS:
                res += 'A and S\n'
            elif not self.nnegS:
                res += 'A\n'
            else:
                res += 'S\n'
        if self.eps[0] != 0:
            res += '  - Warm-up regularization strategy: %i\n' % self.wuStrat
            res += '  - Minimum number of iterations at warm-up: %i\n' % self.minWuIt
            if np.isscalar(self.c_wu):
                res += '  - Tikhonov regularization hyperparameter at warm-up: %.2f\n' % self.c_wu
            else:
                res += '  - Tikhonov regularization hyperparameter at warm-up decreases between %.2f and %.2f in %i ' \
                       'iterations\n' % (np.max(self.c_wu), np.min(self.c_wu), self.cwuDec)
            res += '  - Tikhonov regularization hyperparameter at refinement: %.2f\n' % self.c_ref
        else:  # TODO for the comparison with ODecGMCA, to delete
            res += '  - Regularization strategy: %i (no refinement with SNR)\n' % self.wuStrat
            if np.isscalar(self.c_wu):
                res += '  - Tikhonov regularization hyperparameter : %.2f\n' % self.c_wu
            else:
                res += '  - Tikhonov regularization hyperparameter decreases between %.2f and %.2f in %i iterations\n' \
                       % (np.min(self.c_wu), np.max(self.c_wu), self.cwuDec)
        if self.useMad:
            res += '  - Noise std estimated with MAD\n'
        else:
            res += '  - Noise std estimated with a noise retroprojected in the source domain\n'
        res += '  - nscales = %i  |  k = %.2f  |  K_max = %i%%\n' % (self.nscales, self.k, self.K_max*100)
        if not self.L1:
            res += '  - L0 penalization\n'
        elif not self.doRw:
            res += '  - L1 penalization\n'
        else:
            res += '  - L1 penalization with L1-reweighting\n'
        return res

    def run(self):
        """Run DecGMCA with the data and the parameters stored in the attributes.

        Parameters
        ----------
        no value

        Returns
        -------
        int
            error code
        """

        if self.verb:
            print(self)
        self.initialize()
        core = self.core()
        if core == 1:
            self.aborted = True
            return 1
        refine_s_end = self.refine_s_end()
        if refine_s_end == 1:
            self.aborted = True
            return 1
        self.terminate()
        return 0

    def initialize(self):
        """Initialize the attributes for a new separation.
        
        Returns
        -------
        int
            error code
        """

        X = fftt.convolve(self.X, self.Hfft[0, :] / (self.Hfft + 1e-10))
        # Initialize A
        if self.AInit is not None:
            self.A = self.AInit.copy()
        else:  # PCA with the deteriorated data
            R = X@X.T
            D, V = np.linalg.eig(R)
            self.A = V[:, 0:self.n]
        self.A /= np.maximum(np.linalg.norm(self.A, axis=0), 1e-24)

        # Initialize other parameters
        self.S = np.zeros((self.n, self.p))
        self.Sfft = np.zeros((self.n, self.p), dtype=complex)
        self.Swtrw = np.zeros((self.n, self.p, self.nscales))
        self.lastWuIt = None
        self.lastRefIt = None
        self.nmse = None
        self.ca = None
        self.nmseScales = None
        self.aborted = False

        return 0

    def core(self):
        """Manage the separation.

        This function handles the alternate updates of S and A, as well as the two stages (warm-up and refinement).

        Parameters
        ----------
        no value

        Returns
        -------
        int
            error code
        """

        stage = "wu"

        Sfft_old = np.zeros((self.n, self.p), dtype=complex)
        A_old = np.zeros((self.m, self.n))
        it = 0

        while True:
            it += 1
            # Get parameters of DecGMCA for the current iteration
            strat, c, K, remCs, doRw, nnegS = self.get_parameters(stage, it)

            if self.verb >= 2:
                print("Iteration #%i" % it)

            # Update S
            update_s = self.update_s(strat, c, K, remCs, doRw, nnegS)
            if update_s:  # error caught
                return 1

            # Update A
            update_a = self.update_a()
            if update_a:  # error caught
                return 1

            # Post processing

            delta_S = np.sqrt(np.sum(np.abs(self.Sfft - Sfft_old)**2) / np.sum(np.abs(self.Sfft))**2)
            delta_A = np.max(abs(1-abs(np.sum(self.A*A_old, axis=0))))
            cond_A = np.linalg.cond(self.A)
            Sfft_old = self.Sfft.copy()
            A_old = self.A.copy()

            if self.A0 is not None and self.S0 is not None and self.verb >= 2:
                Acp, Scp, _ = utils.corr_perm(self.A0, self.S0, self.A, self.S, optInd=True)
                if self.verb >= 2:
                    print("NMSE = %.2f  -  CA = %.2f" % (utils.nmse(self.S0, Scp), utils.ca(self.A0, Acp)))

            if self.verb >= 2:
                print("delta_S = %.2e - delta_A = %.2e - cond(A) = %.2f" % (delta_S, delta_A, cond_A))
            if self.verb >= 5:
                print("A:\n", self.A)

            # Stage update

            if stage == 'wu' and it >= self.minWuIt and delta_S <= self.eps[0]:
                if self.verb >= 2:
                    print("> End of the warm-up (iteration %i)" % it)
                self.lastWuIt = it
                stage = 'ref'

            if stage == 'ref' and (delta_S <= self.eps[1] or it >= self.lastWuIt + 200) and (it >= self.lastWuIt + 25):
                if self.verb >= 2:
                    print("> End of the refinement (iteration %i)" % it)
                self.lastRefIt = it
                return 0
                
            if self.eps[0] != 0 and stage == 'wu' and \
                    (it >= self.minWuIt + 100 or (it >= self.minWuIt+10 and cond_A >= 100)):
                if self.verb >= 2:
                    print("> Algorithm did not converge, abort")
                return 1

            # TODO For the comparison with old DecGMCA, to delete:
            if self.eps[0] == 0 and self.lastWuIt is None and delta_S <= 1e-2 and it >= self.minWuIt:
                self.lastWuIt = it
            if self.eps[0] == 0 and self.lastWuIt is not None and (delta_S <= 1e-5 or it >= self.lastWuIt+100):
                if self.verb >= 2:
                    print("> End (iteration %i)" % it)
                return 0

    def get_parameters(self, stage, it):
        """Get the parameters of DecGMCA.

        Return the parameters of DecGMCA according to the stage and the iteration.

        Parameters
        ----------
        stage : str
            stage ('wu': warm-up, 'ref': refinement)
        it : int
            iteration

        Returns
        -------
        (int, float, float, bool, bool, bool)
            regularization strategy,
            regularization hyperparameter,
            L0 support of the sources,
            remove coarse scales,
            do L1 reweighting,
            apply non-negativity constraint on the sources
        """

        if stage == 'wu':
            strat = self.wuStrat
            if np.isscalar(self.c_wu):
                c = self.c_wu
            else:
                c = np.maximum(np.min(self.c_wu),
                               np.max(self.c_wu) * 10**((np.log10(np.min(self.c_wu))-np.log10(np.max(self.c_wu)))
                                                        * (it-1)/(self.cwuDec-1)))
            K = np.minimum(self.K_max / self.minWuIt * it, self.K_max)
            doRw = False    # no l1 reweighing during warm-up
            nnegS = False   # no non-negativity constraint on S during warm-up
            if self.lastWuIt is not None and it >= self.lastWuIt:  # TODO: to delete
                nnegS = self.nnegS
                doRw = self.doRw
        else:
            strat = 4
            c = self.c_ref
            K = self.K_max
            doRw = self.doRw
            nnegS = self.nnegS
        remCs = not self.nnegS  # coarse scales of the data removed if no non-negativity constraint on the sources
        return strat, c, K, remCs, doRw, nnegS

    def update_s(self, strat, c, K, remCs=True, doRw=None, nnegS=None, calcSfft_det=True, Sfft=None, Sfft_det=None,
                 S=None, A=None, iSNR=None, stds=None, Swtrw=None, oracle=False):
        """Perform the update of the sources.

        Perform the update of the sources, comprising a Tikhonov-regularized least-square, a thresholding in the wavelet
        domain and possibly a projection on the positive orthant.

        Parameters
        ----------
        strat: int
            regularization strategy
        c: float
            regularization hyperparameter
        K: float
            L0 support of the sources
        remCs: bool
            remove coarse scale
        doRw: bool
            do reweighting (default: self.doRw)
        nnegS: bool
            apply non-negativity constraint on the sources (default: self.nnegS)
        calcSfft_det: bool
            project the sources in Fourier domain after the update (result saved in Sfft_det)
        Sfft: np.ndarray
            (n,p) complex array, estimated sources in Fourier domain (in-place update, default: self.Sfft)
        Sfft_det: np.ndarray
            (n,p) complex array, estimated sources with only the detail scales in Fourier domain (in-place update,
            default: self.Sfft_det)
        S: np.ndarray
            (n,p) float array, estimated sources (in-place update, default: self.S)
        A: np.ndarray
            (m,n) float array, mixing matrix (default: self.A)
        iSNR: np.ndarray
            (n,p) float array, regularization parameters for strategy #4 (default: calculated from Sfft)
        stds: np.ndarray
            (n,nscales) float array, std of the noise in the source space, per detail scale (default: mad or analytical
            calculation)
        Swtrw: np.ndarray
            (n,p,nscales) float array, sources in the wavelet domain of previous iteration (default: self.Swtrw)
        oracle: bool
            perform an oracle update (using the ground-truth A and S)

        Returns
        -------
        int
            error code
        """

        if nnegS is None:
            nnegS = self.nnegS
        if remCs and nnegS:
            remCs = False
            print('Warning! remCs forced False (the CS cannot be removed to apply the non-negativity const. on S')
        if doRw is None:
            doRw = self.doRw

        ls_s = self.ls_s(strat, c, remCs, Sfft=Sfft, A=A, iSNR=iSNR, oracle=oracle)
        if ls_s:  # error caught
            return 1

        self.thresholding(K, remCs, doRw, nnegS, calcSfft_det, Sfft=Sfft, Sfft_det=Sfft_det, S=S, stds=stds,
                          Swtrw=Swtrw, oracle=oracle)
        
        return 0

    def ls_s(self, strat, c, remCs, Sfft=None, A=None, iSNR=None, oracle=False):
        """Perform the Tikhonov-regularized least-square update of the sources.

        Parameters
        ----------
        strat: int
            regularization strategy
        c: float
            regularization hyperparameter
        remCs: bool
            remove coarse scale
        Sfft: np.ndarray
            (n,p) complex array, estimated sources in Fourier domain (in-place update, default: self.Sfft)
        A: np.ndarray
            (m,n) float array, mixing matrix (default: self.A)
        iSNR: np.ndarray
            (n,p) float array, regularization parameters for strategy #4 (default: calculated from Sfft)
        oracle: bool
            perform an oracle update (using the ground-truth A and S)

        Returns
        -------
        int
            error code
        """

        if Sfft is None:
            Sfft = self.Sfft
        if A is None:
            if not oracle:
                A = self.A
            else:
                A = self.A0
        if remCs:
            Xfft = self.Xfft_det
        else:
            Xfft = self.Xfft
        if strat == 4 and iSNR is None:
            if not oracle:
                iSNR = self.nStdFFT ** 2 / (np.abs(Sfft)+1e-10)**2 * self.supp / self.p
            else:
                iSNR = self.iSNR0
            if remCs:
                iSNR *= (1-self.wt_filters[:, -1])**2

        if self.verb >= 3:
            print("Regularization strategy: %i - hyperparameter: c = %.5f  " % (strat, c))

        normAA = np.linalg.norm(A.T@A, ord=-2)

        Ra = np.einsum('lj,li,lk', A, self.Hfft**2, A)
        if strat == 1:
            Ra += c*np.eye(self.n)[np.newaxis, :, :]
        elif strat == 2:
            Ra += c*np.linalg.norm(Ra, ord=2, axis=(1, 2))[:, np.newaxis, np.newaxis]*np.eye(self.n)[np.newaxis, :, :]
        elif strat == 3:
            Ra += np.maximum(0, c-np.linalg.norm(Ra, ord=-2, axis=(1, 2))/(normAA+1e-2))[:, np.newaxis, np.newaxis] \
                  * np.eye(self.n)[np.newaxis, :, :]
        elif strat == 4:
            eps = np.zeros((self.p, self.n, self.n))
            diag = np.arange(self.n)
            eps[:, diag, diag] = c*iSNR.T
            Ra += eps
        try:
            Ua, Sa, Va = np.linalg.svd(Ra)
        except np.linalg.LinAlgError:
            if self.verb:
                print('SVD did not converge, abort')
            return 1
        Sa = np.maximum(Sa, np.max(Sa, axis=1)[:, np.newaxis] * 1e-9)
        iRa = np.einsum('...ki,...k,...jk', Va, 1/Sa, Ua)
        piA = np.einsum('ijk,lk,li->ijl', iRa, A, self.Hfft)
        Sfft[:] = np.einsum('ijk,ki->ji', piA, Xfft)
        if not self.useMad:
            self.invOpSp = np.einsum('ijk,ijk->ij', piA, piA)

        return 0

    def thresholding(self, K, remCs, doRw, nnegS, calcSfft_det, Sfft=None, Sfft_det=None, S=None, stds=None, Swtrw=None,
                     oracle=False):
        """Perform the thresholding in the wavelet domain of the sources and possibly a projection on the positive
        orthant.

        Parameters
        ----------
        K: float
            L0 support of the sources
        remCs: bool
            remove coarse scale
        doRw: bool
            do reweighting
        nnegS: bool
            apply non-negativity constraint on the sources
        calcSfft_det: bool
            project the sources in Fourier domain after the update (result saved in Sfft_det)
        Sfft: np.ndarray
            (n,p) complex array, estimated sources in Fourier domain (in-place update, default: self.Sfft)
        Sfft_det: np.ndarray
            (n,p) complex array, estimated sources with only the detail scales in Fourier domain (in-place update,
            default: self.Sfft_det)
        S: np.ndarray
            (n,p) float array, estimated sources (in-place update, default: self.S)
        stds: np.ndarray
            (n,nscales) float array, std of the noise in the source space, per detail scale (default: mad or analytical
            calculation)
        Swtrw: np.ndarray
            (n,p,nscales) float array, sources in the wavelet domain of previous iteration (default: self.Swtrw)
        oracle: bool
            perform an oracle update (using the ground-truth A and S)

        Returns
        -------
        int
            error code
        """

        if Sfft is None:
            Sfft = self.Sfft
        if Sfft_det is None:
            Sfft_det = self.Sfft_det
        if S is None:
            S = self.S
        if Swtrw is None:
            if not oracle:
                Swtrw = self.Swtrw
            else:
                Swtrw = self.S0wt

        if self.verb >= 3:
            print("Maximal L0 norm of the sources: %.1f %%" % (K * 100))

        Swt = fftt.wt_trans(Sfft, nscales=self.nscales, fft_in=True)

        # Thresholding
        for i in range(self.n):
            for j in range(self.nscales):
                Swtij = Swt[i, :, j]
                Swtrwij = Swtrw[i, :, j]
                if stds is not None:
                    std = stds[i, j]
                elif self.useMad:
                    std = utils.mad(Swtij, M=self.M)
                else:
                    if not remCs:
                        std = self.nStd*np.sqrt(np.sum(self.invOpSp[:, i] * self.wt_filters[:, j] ** 2) / self.p)
                    else:
                        std = self.nStd * np.sqrt(np.sum(self.invOpSp[:, i] * self.wt_filters[:, j] ** 2 *
                                                         (1-self.wt_filters[:, -1]) ** 2)/self.p)
                thrd = self.k * std

                # If oracle, threshold Swtrw
                if oracle and self.L1 and doRw:
                    Swtrwij = (Swtrwij - np.sign(Swtrwij) * (thrd - np.sqrt(np.abs(
                        (Swtrwij - thrd * np.sign(Swtrwij)) * (3 * thrd * np.sign(Swtrwij) + Swtrwij))))) / 2 * (
                                           np.abs(Swtrwij) >= thrd)

                # Support based threshold
                if K != 1:
                    npix = np.sum(abs(Swtij) - thrd > 0)
                    Kval = np.maximum(np.int(K*npix), 5)
                    thrd = np.partition(abs(Swtij), self.p-Kval)[self.p-Kval]

                if self.verb == 4 and i == 0:
                    print("Threshold of source %i at scale %i: %.5e" % (i+1, j+1, thrd))
                elif self.verb == 5:
                    print("Threshold of source %i at scale %i: %.5e" % (i+1, j+1, thrd))

                # Adapt the threshold if reweighing demanded
                if doRw and self.L1:
                    thrd = thrd/(np.abs(Swtrwij)/np.maximum(1e-20, self.k*std)+1)
                else:
                    thrd = thrd * np.ones(self.p)

                # Apply the threshold
                Swtij[(abs(Swtij) < thrd)] = 0
                if self.L1:
                    indNZ = np.where(abs(Swtij) > thrd)[0]
                    Swtij[indNZ] = Swtij[indNZ]-thrd[indNZ]*np.sign(Swtij[indNZ])

                Swt[i, :, j] = Swtij

        # Reconstruct S
        if remCs:
            S[:] = fftt.wt_rec(Swt[:, :, :-1])
        else:
            S[:] = fftt.wt_rec(Swt)
            
        # Non-negativity constraint
        if nnegS:
            nneg = S > 0
            S *= nneg
            Swt *= nneg[:, :, np.newaxis]

        if oracle:
            return 0
        
        # Save the wavelet coefficients of S for next iteration
        if doRw and self.L1 and not oracle:
            self.Swtrw = Swt[:, :, :-1]

        # Project S in Fourier domain
        Sfft[:] = fftt.fft(S)
        
        if calcSfft_det:     # project the detail scales of S in Fourier domain
            if not remCs:
                Sfft_det[:] = fftt.fft(fftt.wt_rec(Swt[:, :, :-1]))
            else:
                Sfft_det[:] = Sfft
        return 0

    def update_a(self, Sfft=None, A=None):
        """Perform the least-square update of the mixing matrix (with the detail scales of the data and the sources).

        Parameters
        ----------
        Sfft: np.ndarray
            (n,p) complex array, sources in Fourier domain (default: self.Sfft)
        A: np.ndarray
            (m,n) float array, estimated mixing matrix (in-place update, default: self.A)

        Returns
        -------
        int
            error code
        """

        if Sfft is None:
            Sfft = self.Sfft_det
        if A is None:
            A = self.A

        Rs = np.real(np.einsum('il,jl,kl', self.Hfft**2, Sfft, np.conj(Sfft)))
        try:
            Us, Ss, Vs = np.linalg.svd(Rs)
        except np.linalg.LinAlgError:
            if self.verb:
                print('SVD did not converge, abort')
            return 1
        Ss = np.maximum(Ss, np.max(Ss, axis=1)[:, np.newaxis] * 1e-9)
        iRs = np.einsum('...ij,...j,...jk', Us, 1/Ss, Vs)
        Ws = np.real(np.einsum('ij,kj->ik', self.Xfft_det*self.Hfft, np.conj(Sfft)))
        A[:] = np.einsum('ij,ijk->ik', Ws, iRs)

        # Non-negativity constraint
        if self.nnegA:
            sign = np.sign(np.sum(A, axis=0))
            sign[sign == 0] = 1
            A *= sign
            A[:] = np.maximum(A, 0)

        # Oblique constraint
        A /= np.maximum(np.linalg.norm(A, axis=0), 1e-24)
        return 0

    def refine_s_end(self):
        """Perform the finale refinement of the sources, with K = 1 and the coarse scales of the data.

        Returns
        -------
        int
            error code"""

        if self.verb >= 2:
            print("Finale refinement of the sources with the finale estimation of A...")

        strat = self.wuStrat
        c = np.min(self.c_wu)

        update_s = self.update_s(strat, c, 1, remCs=False, doRw=False, calcSfft_det=False)
        if update_s:  # error caught
            return 1

        # Initialize attributes
        self.Swtrw = np.zeros((self.n, self.p, self.nscales))
        Sfft_old = np.zeros((self.n, self.p), dtype=complex)
        delta_S = np.inf
        it = 0

        if self.eps[0] != 0:  # for the comparison with ODecGMCA, to be deleted
            strat = 4
            c = self.c_ref

        while delta_S >= self.eps[2] and it < 50:
            it += 1

            update_s = self.update_s(strat, c, 1, remCs=False, calcSfft_det=False)
            if update_s:  # error caught
                return 1

            delta_S = np.sqrt(np.sum(np.abs(self.Sfft - Sfft_old) ** 2) / np.sum(np.abs(self.Sfft)) ** 2)
            Sfft_old = self.Sfft.copy()

            if self.A0 is not None and self.S0 is not None and self.verb >= 2:
                Acp, Scp, _ = utils.corr_perm(self.A0, self.S0, self.A, self.S, optInd=True)
                if self.verb >= 2:
                    print("NMSE = %.2f" % utils.nmse(self.S0, Scp))

            if self.verb >= 2:
                print("delta_S = %.2e" % delta_S)

        return 0

    def terminate(self):
        """If ground truth data provided, correct permutations and evaluate solution.

        Returns
        -------
        int
            error code
        """

        if self.A0 is not None and self.S0 is not None:
            self.ca, self.nmse, self.nmseScales = utils.evaluate(self.A0, self.S0, self.A, self.S, corrPerm=True,
                                                                 perScale=True, S0wt=self.S0wt)
            if self.verb:
                print('CA : %.2f | NMSE: %.2f' % (self.ca, self.nmse))

        return 0

    def oracle_dss(self, strat, c, S=None, A0=None, iSNR0=None, Swt0=None):
        """Solve the oracle deconvolution source separation problem.

        Parameters
        ----------
        strat: int
            regularization strategy
        c: float
            regularization hyperparameter
        S: np.ndarray
            (n,p) float array, estimated sources (in-place update, default: self.S)
        A0: np.ndarray
            (m,n) float array, ground truth mixing matrix (default: self.A0)
        iSNR0: np.ndarray
            (n,p) float array, ground truth regularization parameters for strategy #4 (default: self.iSNR0)
        Swt0: np.ndarray
            (n,p,nscales) float array, ground truth sources in the wavelet domain (default: self.Swt0)

        Returns
        -------
        int
            error code
        """

        update_s = self.update_s(strat, c, 1, remCs=False, doRw=True, S=S, A=A0, iSNR=iSNR0, Swtrw=Swt0, oracle=True)
        if update_s:  # error caught
            return 1

        self.nmse = utils.nmse(self.S0, self.S)

        return 0

    def find_optc(self, c_lim=None, strat=4, precision=1e-3):
        """Grid search of the optimal regularization hyperparameter.

        Parameters
        ----------
        c_lim: np.ndarray
            zone of the grid search, in log10 scale (default: np.array([-5, 1]))
        strat: int
            regularization strategy
        precision: float
            precision of the optimal hyperparameter

        Returns
        -------
        (float, float)
            optimal regularization parameter,
            associated NMSE
        """

        if c_lim is None:
            c_lim = np.array([-5, 1])

        c_min = 10 ** c_lim[0]
        c_max = 10 ** c_lim[1]
        c_mid = 10 ** ((c_lim[0] + c_lim[1]) / 2)

        self.oracle_dss(strat, c_mid)
        nmse_mid = self.nmse
        
        it = 0
        while it < 50 and c_max / c_min > 1 + precision:
            it += 1
            c_a = 10 ** ((3 * np.log10(c_min) + np.log10(c_max)) / 4)
            c_b = 10 ** ((np.log10(c_min) + 3 * np.log10(c_max)) / 4)
            self.oracle_dss(strat, c_a)
            nmse_a = self.nmse
            self.oracle_dss(strat, c_b)
            nmse_b = self.nmse
            if self.verb >= 2:
                print("NMSE : %.2f | %.2f | %.2f " % (nmse_a, nmse_mid, nmse_b))
            if nmse_mid <= nmse_b:
                c_min, c_mid = c_mid, c_b
                nmse_mid = nmse_b
            elif nmse_mid < nmse_a:
                c_mid, c_max = c_a, c_mid
                nmse_mid = nmse_a
            else:
                c_min, c_max = c_a, c_b
            if self.verb:
                print("Min bound:", c_min, "& max bound:", c_max)

        if c_mid / 10 ** c_lim[0] < 1.01 or 10 ** c_lim[1] / c_mid < 1.01:
            print("Warning! Opt c is near the boundary of the search zone (strat=%i)" % strat)

        return c_mid, nmse_mid
