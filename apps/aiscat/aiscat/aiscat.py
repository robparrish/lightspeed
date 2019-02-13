import lightspeed as ls
import est
import numpy as np
# Internal library submodules
from . import plugin as ls2 # Scattering codes (CPU/GPU)
from . import formfactor # IAM form factors

class AISCAT(object):

    """ Class AISCAT manages the computation of ab initio and IAM elastic
        scattering cross sectionsfor TRXD and UED. Becke grids are used to
        describe the electronic density field for ab initio scattering.
    """

    @staticmethod
    def default_options():
        if hasattr(AISCAT, '_default_options'): return AISCAT._default_options.copy()
        opt = est.Options() 

        opt.add_option(
            key='lot',
            required=True,
            doc='Level of Theory')
        opt.add_option(
            key='grid',
            required=True,
            allowed_types=[ls.BeckeGrid],
            doc='Becke grid for density collocation')
        opt.add_option(
            key='mode',
            allowed_values=['ued', 'xray'],
            doc='ued or xray scattering?')
        opt.add_option(
            key='L',
            required=True,
            allowed_types=[float],
            doc='DeBroglie wavelength (Angstrom)')
        opt.add_option(
            key='s',
            required=True,
            allowed_types=[ls.Tensor],
            doc='Reciprocal scattering radii (Angstrom^-1). Tensor of shape (ns,)')
        opt.add_option(
            key='eta',
            required=True,
            allowed_types=[ls.Tensor],
            doc='Azimuthal angles (radian). Tensor of shape (neta,)')
        opt.add_option(
            key='anisotropy',
            required=True,
            allowed_values=['aligned', 'isotropic', 'parallel', 'perpendicular'],
            doc='Rotational anisotropy')
        opt.add_option(
            key='algorithm',
            required=True,
            allowed_values=['cpu', 'gpu_f_f', 'gpu_f_d', 'gpu_d_d'],
            doc='Algorithm/precision for computation. gpu_d_d recommended for aligned gpu_f_d otherwise')
        opt.add_option(
            key='thre_dens',
            value=1.0E-14,
            allowed_types=[float],
            doc='Grid density collocation cutoff')
        opt.add_option(
            key='hash_R',
            value=5.0,
            allowed_types=[float],
            doc='HashedGrid box size')
            
        AISCAT._default_options = opt
        return AISCAT._default_options.copy()

    def __init__(
        self,
        options,
        ):
        
        self.options = options

    @staticmethod
    def from_options(**kwargs):
        """ Return an instance of this class with default options updated from values in kwargs. """
        return AISCAT(AISCAT.default_options().set_values(kwargs))

    @property
    def lot(self):
        return self.options['lot']

    @property
    def grid(self):
        return self.options['grid']

    @property
    def L(self):
        return self.options['L']

    @property
    def s(self):
        return self.options['s']

    @property
    def eta(self):
        return self.options['eta']

    @property
    def mode(self):
        return self.options['mode']

    @property
    def anisotropy(self):
        return self.options['anisotropy']

    @property
    def algorithm(self):
        return self.options['algorithm']

    # => Compute Routines <= #

    def diffraction_moments(
        self,
        S,
        index1,
        index2,
        ):

        """ Compute ab initio elastic or inelastic diffraction moments for xray
            or ued probes using a variety of anistropy/computational techniques.
    
        The full detector diffraction pattern is:
        
            I(s,eta) = M_0(s) + M_2(s)*cos(2*eta)
    
        This code computes the moments M_0(s) and M_2(s), which are easier to
        compute and store than full diffraction patterns.

        This function only works for cases where the diffraction pattern is
        exactly resolved by M0 and M2 (isotropic, parallel, perpendicular),
        throws otherwise.
    
        Params:
            S (int) - spin index
            index1 (int) - index of bra state
            index2 (int) - index of ket state

        Elastic scattering is assumed if index1==index2 (scattering off of total state density)
        Inelastic scattering is assumed if index1!=index2 (scattering off of transition density)

        Returns: 
            IM (ls.Tensor of shape (ns, 2)) - Detector moments, M_0 in first
                column, M_2 in second column.
        """

        if self.anisotropy not in ['isotropic', 'parallel', 'perpendicular']: 
            raise RuntimeError('Invalid anisotropy for moments: %s' % self.anisotropy)
        # Grid density collocation
        xyzq = self.compute_xyzq(
            D=self.lot.compute_opdm_ao(S, index1, index2),
            include_nuclei=self.mode=='ued' and index1==index2,
            )
        print('Grid Density: %12.6f\n' % np.sum(xyzq[:,3]))
        # Moment function
        if self.anisotropy == 'isotropic' : fun=ls2.isotropic_moments
        elif self.anisotropy == 'parallel' : fun=ls2.parallel_moments
        elif self.anisotropy == 'perpendicular' and self.algorithm == 'cpu': fun=ls2.perpendicular_moments
        elif self.anisotropy == 'perpendicular' and self.algorithm == 'gpu_f_f': fun=ls2.perpendicular_moments_gpu_f_f
        elif self.anisotropy == 'perpendicular' and self.algorithm == 'gpu_f_d': fun=ls2.perpendicular_moments_gpu_f_d
        elif self.anisotropy == 'perpendicular' and self.algorithm == 'gpu_d_d': fun=ls2.perpendicular_moments_gpu_d_d
        else: raise RuntimeError('Invalid anisotropy/algorithm combination: %s, %s' % (self.anisotropy, self.algorithm))
        # Moment computation
        IM = fun(
            self.lot.geometry.resources,
            self.L, 
            self.s,
            xyzq,
            )
        # 1/s^4 weight in UED
        # if self.mode == 'ued':
            # IM[...] /= np.outer(self.s, np.ones(2))**4
        return IM

    def diffraction_pattern(
        self,
        S,
        index1,
        index2,
        ):

        """ Compute ab initio elastic or inelastic diffraction pattern for xray
            or ued probes, using a variety of computational methods.

        If it is tractable to use moments to exactly compute the scattering
        signal (isotropic, parallel, perpendicular cases) this is done first.
    
        If this is not tractable, the full diffraction pattern is directly
        computed.

        Params:
            S (int) - spin index
            index1 (int) - index of bra state
            index2 (int) - index of ket state

        Elastic scattering is assumed if index1==index2 (scattering off of total state density)
        Inelastic scattering is assumed if index1!=index2 (scattering off of transition density)

        Returns: 
            I (ls.Tensor of shape (ns,neta)) - diffraction pattern.
        """
        
        # First try to compute via moments
        if self.anisotropy in ['isotropic', 'parallel', 'perpendicular']:
            IM = self.diffraction_moments(S=S,index1=index1,index2=index2)
            return AISCAT.pattern_from_moments(IM,eta=self.eta)
        # Otherwise, must compute aligned scattering signal
        # Grid density collocation
        xyzq = self.compute_xyzq(
            D=self.lot.compute_opdm_ao(S, index1, index2),
            include_nuclei=self.mode=='ued' and index1==index2,
            )
        print('Grid Density: %12.6f\n' % np.sum(xyzq[:,3]))
        # Scattering collocation points
        theta = 2.0 * np.arcsin(self.s[...] * self.L / (4.0 * np.pi))
        tt, ee = np.meshgrid(theta, self.eta, indexing='ij')
        ss, ee = np.meshgrid(self.s, self.eta, indexing='ij')
        # Compute scattering vectors
        sx = ss * np.cos(tt / 2.0) * np.sin(ee)
        sy = ss * np.sin(tt / 2.0) 
        sz = ss * np.cos(tt / 2.0) * np.cos(ee)
        # Pack for LS 
        sxyz = ls.Tensor((sx.size,3))
        sxyz[:,0] = np.ravel(sx)
        sxyz[:,1] = np.ravel(sy)
        sxyz[:,2] = np.ravel(sz)
        # Diffraction function
        if self.algorithm == 'cpu': fun = ls2.aligned_diffraction2
        elif self.algorithm == 'gpu_f_f': fun = ls2.aligned_diffraction_gpu_f_f
        elif self.algorithm == 'gpu_f_d': fun = ls2.aligned_diffraction_gpu_f_d
        elif self.algorithm == 'gpu_d_d': fun = ls2.aligned_diffraction_gpu_d_d
        # Compute diffraction pattern
        I = fun(
            self.lot.geometry.resources,
            sxyz,
            xyzq,
            )
        # Make sure return is (ns, neta)
        I = ls.Tensor.array(np.reshape(I, (self.s.shape[0], self.eta.shape[0])))
        # 1/s^4 weight in UED
        if self.mode == 'ued':
            I[...] /= ss**4
        return I
        
    def diffraction_moments_iam(
        self,
        ABpairs=None,
        ):

        """ Compute elastic diffraction moments for xray or ued probes via the
            independent atom model (IAM)
    
        The full detector diffraction pattern is:
        
            I(s,eta) = M_0(s) + M_2(s)*cos(2*eta)
    
        This code computes the moments M_0(s) and M_2(s), which are easier to
        compute and store than full diffraction patterns.

        This function only works for cases where the diffraction pattern is
        exactly resolved by M0 and M2 (isotropic, parallel, perpendicular),
        throws otherwise.

        Params: 
            ABpairs (list of pairs of int) - indices of A-B atom pairs to
            include in computation. Defaults to all atom pairs. If explicitly
            specified, note that AB and BA pairs must both be specified, e.g.,
            (0,0), (0,1), (1,0), and (1,1) must all be specified to get full
            set of interactions between atoms 0 and 1.
    
        Returns: 
            IM (ls.Tensor of shape (ns, 2)) - Detector moments, M_0 in first
                column, M_2 in second column.
        """

        if self.anisotropy not in ['isotropic', 'parallel', 'perpendicular']: 
            raise RuntimeError('Invalid anisotropy for moments: %s' % self.anisotropy)

        # Geometry
        molecule = self.lot.molecule
        # Scattering centers 
        xyz = molecule.xyz[...] * ls.units['ang_per_bohr']
        # Atom pairs to include (all pairs if not otherwise specified)
        if ABpairs is None:
            ABpairs = []
            for A in range(molecule.natom):
                for B in range(molecule.natom):
                    ABpairs.append((A,B))
        # Atomic form factors
        factors = [formfactor.AtomicFormFactor.build_factor(atom.symbol, atom.Z, mode=self.mode) for atom in molecule.atoms]

        # Collocate the atomic form factors
        f = np.zeros((len(factors), self.s.size))
        for A, factor in enumerate(factors):
            f[A,:] = factor.evaluate(qx=0.0,qy=0.0,qz=self.s[...])
        # Compute scattering angles via Bragg equation
        theta = 2.0 * np.arcsin(self.s[...] * self.L / (4.0 * np.pi))
        # Target
        I0 = np.zeros_like(self.s)
        I2 = np.zeros_like(self.s)
        with np.errstate(all='ignore'):
            for A, B in ABpairs:
                # Geometry
                d = xyz[A,:] - xyz[B,:]
                r2 = np.sum(d**2)
                r = np.sqrt(r2)
                sg2 = np.sum(d[:2]**2) / r2 if r2 != 0.0 else 1.0
                sr = self.s[...]*r
                # Bessel functions
                J0 = np.sin(sr) / sr
                J0[sr == 0.0] = 1.0
                J1sr = np.sin(sr) / sr**3 - np.cos(sr) / sr**2
                J1sr[sr == 0.0] = 1.0/3.0
                J2 = (3.0 / sr**2 - 1.0) * np.sin(sr) / sr - 3.0 * np.cos(sr) / sr**2
                J2[sr == 0.0] = 0.0
                # Kernels
                if self.anisotropy == 'isotropic':
                    I0 += f[A,:] * f[B,:] * J0
                elif self.anisotropy == 'perpendicular':
                    Iz = f[A,:] * f[B,:] * (J1sr - (sg2 + (2.0 - 3.0 * sg2) * np.cos(0.5 * theta)**2) * J2 / 2.0)
                    Ix = f[A,:] * f[B,:] * (J1sr - (sg2) * J2 / 2.0)
                    I0 += 0.5 * (Iz + Ix)
                    I2 += 0.5 * (Iz - Ix)
                elif self.anisotropy == 'parallel':
                    I0 += f[A,:] * f[B,:] * (J1sr - (sg2 + (2.0 - 3.0 * sg2) * np.sin(0.5 * theta)**2) * J2 / 2.0)

        I = ls.Tensor.array(np.vstack((I0,I2)).T)
        return I

    def diffraction_pattern_iam(
        self,
        ABpairs=None,
        ):

        """ Compute elastic diffraction pattern for xray or ued probes, using
            the independent atom model (IAM)

        If it is tractable to use moments to exactly compute the scattering
        signal (isotropic, parallel, perpendicular cases) this is done first.
    
        If this is not tractable, the full diffraction pattern is directly
        computed.

        Params: 
            ABpairs (list of pairs of int) - indices of A-B atom pairs to
            include in computation. Defaults to all atom pairs. If explicitly
            specified, note that AB and BA pairs must both be specified, e.g.,
            (0,0), (0,1), (1,0), and (1,1) must all be specified to get full
            set of interactions between atoms 0 and 1.

        Returns: 
            I (ls.Tensor of shape (ns,neta)) - diffraction pattern.
        """

        # First try to compute via moments
        if self.anisotropy in ['isotropic', 'parallel', 'perpendicular']:
            IM = self.diffraction_moments_iam(ABpairs=ABpairs)
            return AISCAT.pattern_from_moments(IM,eta=self.eta)
        # Otherwise, must compute aligned scattering signal
        # Geometry
        molecule = self.lot.molecule
        # Scattering centers 
        xyz = molecule.xyz[...] * ls.units['ang_per_bohr']
        # Atom pairs to include (all pairs if not otherwise specified)
        if ABpairs is None:
            ABpairs = []
            for A in range(molecule.natom):
                for B in range(molecule.natom):
                    ABpairs.append((A,B))
        # Atomic form factors
        factors = [formfactor.AtomicFormFactor.build_factor(atom.symbol, atom.Z, mode=self.mode) for atom in molecule.atoms]
        # Collocate the atomic form factors
        f = np.zeros((len(factors), self.s.size))
        for A, factor in enumerate(factors):
            f[A,:] = factor.evaluate(qx=0.0,qy=0.0,qz=self.s[...])
        # Compute scattering angles via Bragg equation
        theta = 2.0 * np.arcsin(self.s[...] * self.L / (4.0 * np.pi))
        tt, ee = np.meshgrid(theta, self.eta, indexing='ij')
        ss, ee = np.meshgrid(self.s, self.eta, indexing='ij')
        # Compute scattering vectors
        sx = ss * np.cos(tt / 2.0) * np.sin(ee)
        sy = ss * np.sin(tt / 2.0) 
        sz = ss * np.cos(tt / 2.0) * np.cos(ee)
        # Target 
        I = np.zeros_like(ss)
        for A, B in ABpairs:
            d = xyz[A,:] - xyz[B,:]
            sr = sx * d[0] + sy * d[1] + sz * d[2] 
            I += np.outer(f[A,:] * f[B,:], np.ones_like(self.eta)) * np.cos(sr)
        I = ls.Tensor.array(I)
        return I

    def diffraction_moments_at(
        self,
        Ainds=None,
        ):

        """ Compute elastic diffraction moments for xray or ued probes,
            returning only the diagonal contribution from the IAM.

        Params: 
            Ainds (list of int) - atom indices to include in computation. All
                atoms are used if None.
        Returns: 
            IM (ls.Tensor of shape (ns, 2)) - Detector moments, M_0 in first
                column, M_2 in second column.
        """

        if Ainds is None:
            ABpairs = []
            for A in range(self.lot.molecule.natom):
                ABpairs.append((A,A))
        else:
            for A in Ainds:
                ABpairs.append((A,A))
        return self.diffraction_moments_iam(ABpairs=ABpairs)

    def diffraction_pattern_at(
        self,
        Ainds=None,
        ):

        """ Compute elastic diffraction pattern for xray or ued probes,
            returning only the diagonal contribution from the IAM.

        Params: 
            Ainds (list of int) - atom indices to include in computation. All
                atoms are used if None.
        Returns: 
            I (ls.Tensor of shape (ns,neta)) - diffraction pattern.
        """

        if Ainds is None:
            ABpairs = []
            for A in range(self.lot.molecule.natom):
                ABpairs.append((A,A))
        else:
            for A in Ainds:
                ABpairs.append((A,A))
        return self.diffraction_pattern_iam(ABpairs=ABpairs)

    # => Internal Utility Routines <= #

    def compute_xyzq(
        self,
        D,
        include_nuclei=False,
        ):

        """ Compute rho on the BeckeGrid from D.
    
        Params:
            D (ls.Tensor of shape (nao, nao)) - Total OPDM
            include_nuclei (bool) - Include nuclei with opposite charge?
        Returns:
            xyzq (ls.Tensor of shape (ngrid, 4)) - x, y, z grid coordinates in
                Angstrom and grid charges q in fundamental charge units.
         """

        # Setup a HashedGrid for collocation
        hashed = ls.HashedGrid(
            self.grid.xyz,
            self.options['hash_R'],
            )
        # Collocate density to grid
        rho = ls.GridBox.ldaDensity(
            self.lot.geometry.resources,
            self.lot.geometry.pairlist,
            D,
            hashed,
            self.options['thre_dens'],
            ) 
        # Point charge field for diffraction
        xyzq = ls.Tensor.zeros_like(self.grid.xyzw)
        xyzq[:,:3] = ls.units['ang_per_bohr'] * self.grid.xyzw[:,:3] # We'll be doing the computation in Angstrom from here out
        xyzq[:,3] = rho * self.grid.xyzw[:,3]
        # Augment grid electronic charge with nuclei if requested
        if include_nuclei:
            xyz2 = self.lot.geometry.qm_molecule.xyzZ
            xyz2[:,:3] *= ls.units['ang_per_bohr']
            xyzq = ls.Tensor.array(np.vstack((-xyzq[...], xyz2)))

        return xyzq 

    # => Utility/Plotting Routines <= #

    @staticmethod
    def pattern_from_moments(
        M,
        eta=ls.Tensor.array(np.linspace(0.0, 2.0*np.pi, 120)),
        ):
    
        """ Get detector pattern from previously-computed moments.
    
        Params:
            M (ls.Tensor of shape (ns, 2)) - Detector moments M_0 and M_2.
            eta (ls.Tensor of shape (neta, 2)) - Detector azimuthal angles
                (radian). For detector plots, it is recommended that this make a
                complete circle, i.e., that 2*pi be the last point in eta.
        Return:  
            I (ls.Tensor of shape (ns, neta)) - Detector pattern
    
        """
    
        return ls.Tensor.array(np.outer(M[:,0], np.ones_like(eta)) + np.outer(M[:,1], np.cos(2.0 * eta[...])))

    @staticmethod
    def pattern_plot(
        s,
        eta,
        I,
        filename=None,
        clf=True,
        cmap=None,
        levels=None,
        nlevel=127,
        twosided=False,
        colorbar=False,
        cticks=None,
        smin=None,
        smax=None,
        borders=False,
        ):
    
        """ Make a matplotlib polar contourf plot of the detector pattern.
    
        Params:
            s (np.ndarray of shape (ns,)) - reciprocal scatter vector norms
                (Angstrom^-1).
            eta (np.ndarray of shape (neta,)) - reciprocal scatter vector
                azimuthal angles (radians).
            I (np.ndarray of shape (ns, neta)) - scattering signal.
            filename (str or None) - filename to save plot to (plot not saved if None)
            clf (bool) - call plt.clf() (True) or not (False)?
            cmap (colormap) - Colormap to use. Seismic, coolwarm, or bwr are good
                for two-sided data. Viridis (default) is good for one-sided data.
            levels (np.ndarray of shape (nlevel,)) - levels for contours. nlevel
                evenly spaced contours are used if None.
            nlevel (int) - Number of contour levels to use if levels is None
            twosided (bool) - Is the data twosided or not? Only used if levels if
                None. In this case levels are chosen from -Imax to +Imax if
                twosided, Imin to Imax otherwise.
            colorbar (bool) - Add a colorbar to the plot (True) or not (False)?
            cticks (list) - List of color ticks or None
            smin (float) - Minimum s to plot. min(ss) is used if None
            smax (float) - Maximum s to plot. max(ss) is used if None
            borders (bool) - Add borders to smin/smax (True) or not (False)?
        Return:
            plt (matplotlib.pyplot) - the finished plot environment, for additional
            changes by the user.
        
        """
        import matplotlib.pyplot as plt
    
        ss, ee = np.meshgrid(s, eta, indexing='ij')

        if cmap is None:
            cmap = plt.get_cmap('viridis')
        
        # Levels if needed
        if levels is None:
            if twosided:
                levels = np.linspace(-np.max(np.abs(I)), +np.max(np.abs(I)), nlevel)
            else:
                levels = np.linspace(np.min(I),np.max(I), nlevel)
    
        # Edge clamps
        if smin is None: smin = np.min(ss)
        if smax is None: smax = np.max(ss)
    
        # Detector plot
        plt.clf()
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
        h = ax.contourf(ee, ss, I, levels=levels, cmap=cmap, extend='both')
        if borders:
            th = np.linspace(0.0, 2.0*np.pi, 1000)
            # ax.plot(th, smin * np.ones_like(th), '-k', linewidth=1.0)
            ax.plot(th, smax * np.ones_like(th), '-k', linewidth=1.0)
        ax.set_theta_offset(np.pi / 2.0)
        ax.set_rlim([smin, smax])
        ax.set_rticks([])
        ax.set_thetagrids([])
        plt.axis('off')
        if colorbar:
            cbar = plt.colorbar(h)
            if cticks:
                cbar.set_ticks(cticks)
        if filename:
            plt.savefig(filename, transparent=True, bbox_inches='tight')
        return plt
    
