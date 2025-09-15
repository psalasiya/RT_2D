from ngsolve import *
import numpy as np

class eigenspectrum(object):
    def __init__(self, ev, ef):
        """
        A class to process and filter eigenspectrum (eigenvalues and eigenfunctions)
        from a QEP solver.

        Attributes:
            ev (list[complex]): Eigenvalues (e.g., kx values).
            ef (list[GridFunction]): Corresponding eigenfunctions from ngsolve.
        """

        
        self.ev = ev
        self.ef = ef
    
    def pick_first_BZ(self):
        """
        Filter eigenvalues and eigenfunctions into the first Brillouin zone.

        Args:
            ev (list): A list of complex eigenvalues (kx).
            ef (list): A list of corresponding ngsolve.GridFunction eigenfunctions.
        """
        
        temp1 = []
        temp2 = []
        for i in range(len(self.ev)):
            if round(self.ev[i].real, 3) > -round(np.pi, 3) and round(self.ev[i].real, 3) <= round(np.pi, 3):
                temp1.append(self.ev[i])
                temp2.append(self.ef[i])

        temp1 = np.array(temp1, dtype=complex)
        self.ev_BZ = temp1
        self.ef_BZ = temp2
    
    def seperate_ev(self):
        """
        Separates eigenvalues/eigenfunctions into three categories:
            - Real
            - Purely imaginary (with damping limit)
            - Complex

        Stores results in:
            ev_r, ef_r: Real eigenvalues and eigenfunctions.
            ev_i, ef_i: Purely imaginary eigenvalues/eigenfunctions.
            ev_c, ef_c: Complex eigenvalues/eigenfunctions.
        """

        # maximum number of eigenvalues in each bins
        num_ev_r = 50
        num_ev_i = 50
        num_ev_c = 50

        # initializ counter for bins
        ctr_r = 0
        ctr_i = 0
        ctr_c = 0

        # initialize each bin
        ev_r = []
        ev_c = []
        ev_i = []
        ef_r = []
        ef_i = []
        ef_c = []
        for i in range(len(self.ev_BZ)):
            # case: eigenvalue is either zero or purely imaginary
            if round(self.ev_BZ[i].real, 4) == 0:
                # subcase: eigenvalue is zero
                if round(self.ev_BZ[i].imag, 4) == 0:
                    if ctr_r < num_ev_r:
                        ev_r.append(0 + 0*1j)
                        ef_r.append(self.ef_BZ[i])
                        ctr_r += 1
                # subcase: eigenvalue is purely imaginary with damping limit of 500
                elif round(self.ev_BZ[i].imag, 4) >= -500 and round(self.ev_BZ[i].imag, 4) <= 500:
                    if ctr_i < num_ev_i:
                        ev_i.append(0 + self.ev_BZ[i].imag * 1j)
                        ef_i.append(self.ef_BZ[i])
                        ctr_i += 1  
            # case: eigenvalue is either purely real or complex 
            else:
                # subcase: eigenvalue is real
                if round(self.ev_BZ[i].imag, 4) == 0:
                    if ctr_r < num_ev_r:
                        ev_r.append(self.ev_BZ[i].real + 0*1j)
                        ef_r.append(self.ef_BZ[i])
                        ctr_r += 1
                # subcase: eigenvalue is complex with damping limit of 500
                elif round(self.ev_BZ[i].imag, 4) >= -500 and round(self.ev_BZ[i].imag, 4) <= 500:
                    if ctr_c < num_ev_c:
                        ev_c.append(self.ev_BZ[i])
                        ef_c.append(self.ef_BZ[i])
                        ctr_c += 1

        # sorting imaginary eigenvalues according to their imaginary part
        ev_i_srt = np.sort(ev_i)
        ind_i    = np.argsort(ev_i)
        ef_i_srt = []
        for i in range(len(ev_i_srt)):
            ef_i_srt.append(ef_i[ind_i[i]])
        
        # sorting complex eigenvalues accoring to their imaginary part
        ind_c = np.argsort(np.imag(ev_c))
        ev_c_srt = []
        ef_c_srt = []
        for i in range(len(ev_c_srt)):
            ev_c_srt.append(ev_c[ind_c[i]])
            ef_c_srt.append(ef_c[ind_c[i]])
        
        self.ev_r = ev_r
        self.ef_r = ef_r
        self.ev_i = ev_i_srt
        self.ef_i = ef_i_srt
        self.ev_c = ev_c_srt
        self.ef_c = ef_c_srt
    
    def generate_gradient(self, mu):
        """ this method generates gradient of the previously generated eigenfunctions """

        g1 = []
        g2 = []
        g3 = []
        for i in range(len(self.ef_r)):
            temp = mu * (grad(self.ef_r[i]) * CoefficientFunction((1, 0)) + 1j * self.ev_r[i] * self.ef_r[i])
            g1.append(temp)
        for i in range(len(self.ef_i)):
            temp = mu * (grad(self.ef_i[i]) * CoefficientFunction((1, 0)) + 1j * self.ev_i[i] * self.ef_i[i])
            g2.append(temp)
        for i in range(len(self.ef_c)):
            temp = mu * (grad(self.ef_c[i]) * CoefficientFunction((1, 0)) + 1j * self.ev_c[i] * self.ef_c[i])
            g3.append(temp)
        
        self.gef_r = g1
        self.gef_i = g2
        self.gef_c = g3
    
    def generate_eigenspectrum(self, trunc, ky, a):
        """ This method captures the eigenvalues whose imaginary part < trunc. It then separtes the left and right propagating/evanescent/decating waves and then creates a list of eigenvalues arranged as left-eigenvalues coming before the right-eigenvalues. """

        # creating a tempory list of eigenvalues, eigenfunctions and their gradients s.t. imaginary part of the values < trunc
        kxn_temp = []
        phin_temp = []
        psin_temp = []
        for i in range(len(self.ev_r)):
            if np.abs(np.imag(self.ev_r[i])) <= trunc:
                kxn_temp.append(self.ev_r[i])
                phin_temp.append(self.ef_r[i])
                psin_temp.append(self.gef_r[i])
        for i in range(len(self.ev_i)):
            if np.abs(np.imag(self.ev_i[i])) <= trunc:
                kxn_temp.append(self.ev_i[i])
                phin_temp.append(self.ef_i[i])
                psin_temp.append(self.gef_i[i])
        for i in range(len(self.ev_c)):
            if np.abs(np.imag(self.ev_c[i])) <= trunc:
                kxn_temp.append(self.ev_c[i])
                phin_temp.append(self.ef_c[i])
                psin_temp.append(self.gef_c[i])
        
        # separating the eigenvalues according to left- or right-going waves
        # A small tolerance is used via rounding to handle floating point inaccuracies.
        kxnl = []
        kxnr = []
        phinl = []
        phinr = []
        psinl = []
        psinr = []
        for i in range(len(kxn_temp)):
            # left-decaying modes
            if np.round(np.real(kxn_temp[i]), 4) == 0 and np.imag(kxn_temp[i]) < 0:
                kxnl.append(kxn_temp[i])
                phinl.append(phin_temp[i])
                psinl.append(psin_temp[i])
            # right-decaying modes
            elif np.round(np.real(kxn_temp[i]), 4) == 0 and np.imag(kxn_temp[i]) > 0:
                kxnr.append(kxn_temp[i])
                phinr.append(phin_temp[i])
                psinr.append(psin_temp[i])
            # right-propagating/evanescent modes
            elif np.imag(kxn_temp[i]) > 0:
                kxnr.append(kxn_temp[i])
                phinr.append(phin_temp[i])
                psinr.append(psin_temp[i])
            # left-propagating/evanescent modes
            elif np.imag(kxn_temp[i]) < 0:
                kxnl.append(kxn_temp[i])
                phinl.append(phin_temp[i])
                psinl.append(psin_temp[i])
            # right-decaying modes (edge of the first BZ)
            elif np.round(np.real(kxn_temp[i]), 4) == np.round(np.pi/a, 4) and np.imag(kxn_temp[i]) > 0:
                kxnr.append(kxn_temp[i])
                phinr.append(phin_temp[i])
                psinr.append(psin_temp[i])
            # left-decaying modes(edge of the first BZ)
            elif np.round(np.real(kxn_temp[i]), 4) == np.round(np.pi/a, 4) and np.imag(kxn_temp[i]) < 0:
                kxnl.append(kxn_temp[i]-2*np.pi)
                phinl.append(phin_temp[i])
                psinl.append(psin_temp[i])
            # left-propagating modes
            elif np.real(kxn_temp[i]) < 0 and np.round(np.imag(kxn_temp[i]), 4) == 0:
                kxnl.append(kxn_temp[i])
                phinl.append(phin_temp[i])
                psinl.append(psin_temp[i])
            # right-propagating modes
            elif np.real(kxn_temp[i]) > 0 and np.round(np.imag(kxn_temp[i]), 4) == 0:
                kxnr.append(kxn_temp[i])
                phinr.append(phin_temp[i])
                psinr.append(psin_temp[i])

        # combing left- and right-going eigenvalues, eigenfunctions and their gradients.
        kxn = kxnl + kxnr
        krt_vec = [(kxn[n], ky) for n in range(kxn)]
        self.krt = krt_vec
        self.phin = phinl + phinr
        self.psin = psinl + psinr