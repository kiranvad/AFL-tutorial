import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

class BlockCopolymerSAS:
    """
    Model for Small-Angle Scattering of disordered block copolymers
    using Random Phase Approximation (RPA)
    """
    
    def __init__(self, N=1000, f=0.5, b_A=1.0, b_B=1.0, rho=1.0):
        """
        Initialize block copolymer parameters
        
        Parameters:
        -----------
        N : int
            Total degree of polymerization
        f : float
            Volume fraction of block A (0 < f < 1)
        b_A, b_B : float
            Scattering length densities of blocks A and B
        rho : float
            Monomer density
        """
        self.N = N
        self.f = f
        self.b_A = b_A
        self.b_B = b_B
        self.rho = rho
        
        # Calculate block lengths
        self.N_A = int(N * f)
        self.N_B = N - self.N_A
        
        # Statistical segment length (assumed equal for both blocks)
        self.a = 1.0  # Kuhn length
        
    def debye_function(self, x):
        """
        Debye scattering function: g(x) = 2(e^(-x) + x - 1)/x^2
        With proper handling of x→0 limit
        """
        # Handle x=0 case
        mask = np.abs(x) < 1e-6
        result = np.zeros_like(x)
        
        # For small x, use series expansion: g(x) ≈ 1 - x/3 + x²/12 - ...
        result[mask] = 1.0 - x[mask]/3.0 + x[mask]**2/12.0
        
        # For finite x
        x_nonzero = x[~mask]
        result[~mask] = 2 * (np.exp(-x_nonzero) + x_nonzero - 1) / x_nonzero**2
        
        return result
    
    def form_factors(self, q):
        """
        Calculate form factors for blocks A and B
        """
        # Radius of gyration squared for each block
        Rg2_A = self.N_A * self.a**2 / 6
        Rg2_B = self.N_B * self.a**2 / 6
        Rg2_total = self.N * self.a**2 / 6
        
        # Arguments for Debye functions
        x_A = q**2 * Rg2_A
        x_B = q**2 * Rg2_B
        x_AB = q**2 * Rg2_total
        
        # Form factors
        g_AA = self.N_A * self.debye_function(x_A)
        g_BB = self.N_B * self.debye_function(x_B)
        g_AB = np.sqrt(self.N_A * self.N_B) * self.debye_function(x_AB)
        
        return g_AA, g_BB, g_AB
    
    def structure_factor_rpa(self, q, chi_N):
        """
        Calculate structure factor using Random Phase Approximation
        
        Parameters:
        -----------
        q : array_like
            Scattering vector magnitude
        chi_N : float
            Flory-Huggins parameter times degree of polymerization
        """
        g_AA, g_BB, g_AB = self.form_factors(q)
        
        # RPA denominator
        denominator = 2 * chi_N - (g_AA + g_BB - 2 * g_AB)
        
        # Avoid division by zero near spinodal
        denominator = np.maximum(denominator, 1e-10)
        
        # Structure factor
        S = self.N / denominator
        
        return S
    
    def scattering_intensity(self, q, chi_N, contrast=None):
        """
        Calculate scattering intensity I(q)
        
        Parameters:
        -----------
        q : array_like
            Scattering vector magnitude
        chi_N : float
            Flory-Huggins parameter times degree of polymerization
        contrast : float, optional
            Scattering contrast. If None, uses (b_A - b_B)²
        """
        if contrast is None:
            contrast = (self.b_A - self.b_B)**2
        
        S = self.structure_factor_rpa(q, chi_N)
        
        # Scattering intensity
        I = contrast * self.rho * S
        
        return I
    
    def correlation_length(self, chi_N):
        """
        Calculate correlation length in disorder phase
        """
        # At q=0 limit
        g_AA_0 = self.N_A
        g_BB_0 = self.N_B
        g_AB_0 = np.sqrt(self.N_A * self.N_B)
        
        # Correlation length
        xi_inv_sq = (2 * chi_N - (g_AA_0 + g_BB_0 - 2 * g_AB_0)) / (self.N * self.a**2 / 6)
        xi = 1 / np.sqrt(np.maximum(xi_inv_sq, 1e-10))
        
        return xi
    
    def spinodal_chi_N(self):
        """
        Calculate spinodal point (order-disorder transition)
        """
        return (1/np.sqrt(self.f) + 1/np.sqrt(1-self.f))**2 / 2

def plot_sas_curves():
    """
    Example: Plot SAS curves for different chi_N values
    """
    # Initialize block copolymer
    bcp = BlockCopolymerSAS(N=400, f=0.5, b_A=1.0, b_B=0.0)
    
    # q range
    q = np.logspace(-3, 0, 200)
    
    # Different chi_N values
    chi_N_values = [5, 8, 10, 12]
    spinodal = bcp.spinodal_chi_N()
    
    plt.figure(figsize=(12, 8))
    
    # Plot SAS curves
    plt.subplot(2, 2, 1)
    for chi_N in chi_N_values:
        I = bcp.scattering_intensity(q, chi_N)
        x_axis = (q**2)*
        plt.loglog(q, I, label=f'χN = {chi_N}')
    
    plt.axvline(1.9/np.sqrt(bcp.N * bcp.a**2 / 6), color='k', linestyle='--', alpha=0.5, label='q* ≈ 1.9/Rg')
    plt.xlabel('q (Å⁻¹)')
    plt.ylabel('I(q)')
    plt.title('SAS Curves in Disorder Phase')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot structure factor
    plt.subplot(2, 2, 2)
    for chi_N in chi_N_values:
        S = bcp.structure_factor_rpa(q, chi_N)
        plt.loglog(q, S, label=f'χN = {chi_N}')
    
    plt.xlabel('q (Å⁻¹)')
    plt.ylabel('S(q)')
    plt.title('Structure Factor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot correlation length vs chi_N
    plt.subplot(2, 2, 3)
    chi_N_range = np.linspace(5, 15, 100)
    xi_values = [bcp.correlation_length(chi_N) for chi_N in chi_N_range]
    
    plt.plot(chi_N_range, xi_values, 'b-', linewidth=2)
    plt.axvline(spinodal, color='r', linestyle='--', label=f'Spinodal χN = {spinodal:.2f}')
    plt.xlabel('χN')
    plt.ylabel('Correlation Length ξ')
    plt.title('Correlation Length vs χN')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot peak intensity vs chi_N
    plt.subplot(2, 2, 4)
    q_peak = 1.9 / np.sqrt(bcp.N * bcp.a**2 / 6)
    peak_intensities = [bcp.scattering_intensity(q_peak, chi_N) for chi_N in chi_N_range]
    
    plt.semilogy(chi_N_range, peak_intensities, 'g-', linewidth=2)
    plt.axvline(spinodal, color='r', linestyle='--', label=f'Spinodal χN = {spinodal:.2f}')
    plt.xlabel('χN')
    plt.ylabel('Peak Intensity I(q*)')
    plt.title('Peak Intensity vs χN')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def compare_compositions():
    """
    Compare SAS curves for different block compositions
    """
    # q range
    q = np.logspace(-3, 0, 200)
    chi_N = 10
    
    compositions = [0.3, 0.5, 0.7]
    
    plt.figure(figsize=(10, 6))
    
    for f in compositions:
        bcp = BlockCopolymerSAS(N=1000, f=f, b_A=1.0, b_B=0.0)
        I = bcp.scattering_intensity(q, chi_N)
        plt.loglog(q, I, label=f'f = {f}')
    
    plt.xlabel('q (Å⁻¹)')
    plt.ylabel('I(q)')
    plt.title(f'SAS Curves for Different Compositions (χN = {chi_N})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    # Run examples
    plot_sas_curves()
    compare_compositions()
    
    # Print some key parameters
    bcp = BlockCopolymerSAS(N=1000, f=0.5)
    print(f"Spinodal χN = {bcp.spinodal_chi_N():.2f}")
    print(f"Correlation length at χN=10: {bcp.correlation_length(10):.2f}")