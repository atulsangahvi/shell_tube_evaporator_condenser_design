import streamlit as st
import numpy as np
import pandas as pd
import math
from scipy.optimize import fsolve
import plotly.graph_objects as go
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Password protection
def check_password():
    """Password protection for the app"""
    def password_entered():
        if st.session_state["password"] == "Semaanju":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input(
            "Enter Password", type="password", on_change=password_entered, key="password"
        )
        st.write("*Please enter the password to access the design tool*")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input(
            "Enter Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        return True

# Page configuration
st.set_page_config(
    page_title="DX Shell & Tube HX Designer",
    page_icon="🌡️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #1E3A8A;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E5E7EB;
    }
    .result-box {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #3B82F6;
    }
    .warning-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #F59E0B;
    }
    .dx-badge {
        display: inline-block;
        background-color: #3B82F6;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-weight: bold;
        font-size: 0.85rem;
        margin-left: 0.5rem;
    }
    .condenser-badge {
        display: inline-block;
        background-color: #8B5CF6;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-weight: bold;
        font-size: 0.85rem;
        margin-left: 0.5rem;
    }
    .velocity-good {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 0.25rem 0.5rem;
        border-radius: 0.5rem;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .velocity-low {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.25rem 0.5rem;
        border-radius: 0.5rem;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .velocity-high {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.25rem 0.5rem;
        border-radius: 0.5rem;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .glycol-ethylene {
        background-color: #E0F2FE;
        color: #0C4A6E;
        padding: 0.25rem 0.5rem;
        border-radius: 0.5rem;
        font-size: 0.85rem;
    }
    .glycol-propylene {
        background-color: #F0FDF4;
        color: #166534;
        padding: 0.25rem 0.5rem;
        border-radius: 0.5rem;
        font-size: 0.85rem;
    }
    .stButton>button {
        width: 100%;
    }
    .input-with-buttons {
        display: flex;
        align-items: center;
        gap: 5px;
        margin-bottom: 10px;
    }
    .input-label {
        font-weight: bold;
        margin-bottom: 5px;
        color: #374151;
    }
    .number-input {
        width: 100px;
        text-align: center;
    }
    .kw-comparison {
        background-color: #F0F9FF;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #3B82F6;
    }
    .kw-match {
        background-color: #D1FAE5;
        border-left: 4px solid #10B981;
    }
    .kw-mismatch {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
    }
    .temp-comparison {
        background-color: #FEFCE8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #F59E0B;
    }
    .region-box {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #94A3B8;
    }
    .region-desuperheat {
        border-left: 4px solid #F59E0B;
    }
    .region-condense {
        border-left: 4px solid #3B82F6;
    }
    .region-subcool {
        border-left: 4px solid #10B981;
    }
</style>
""", unsafe_allow_html=True)

class DXHeatExchangerDesign:
    """DX (Direct Expansion) Shell & Tube Heat Exchanger Design"""
    
    # Enhanced refrigerant properties database with saturation enthalpies approximation
    REFRIGERANTS = {
        "R134a": {
            "cp_vapor": 0.852,  # kJ/kg·K at 5°C
            "cp_liquid": 1.434,
            "h_fg": 198.7,  # kJ/kg at ~5°C (will be adjusted for temperature)
            "rho_vapor": 14.43,  # kg/m³
            "rho_liquid": 1277.8,
            "mu_vapor": 1.11e-5,  # Pa·s
            "mu_liquid": 2.04e-4,
            "k_vapor": 0.0116,  # W/m·K
            "k_liquid": 0.0845,
            "pr_vapor": 0.815,
            "pr_liquid": 3.425,
            "sigma": 0.00852,  # Surface tension N/m
            "T_critical": 101.1  # °C
        },
        "R404A": {
            "cp_vapor": 0.823,
            "cp_liquid": 1.553,
            "h_fg": 163.3,
            "rho_vapor": 33.16,
            "rho_liquid": 1131.8,
            "mu_vapor": 1.23e-5,
            "mu_liquid": 1.98e-4,
            "k_vapor": 0.0108,
            "k_liquid": 0.0718,
            "pr_vapor": 0.938,
            "pr_liquid": 4.257,
            "sigma": 0.00682,
            "T_critical": 72.1
        },
        "R407C": {
            "cp_vapor": 1.246,
            "cp_liquid": 1.448,
            "h_fg": 200.0,
            "rho_vapor": 30.02,
            "rho_liquid": 1149.7,
            "mu_vapor": 1.25e-5,
            "mu_liquid": 1.90e-4,
            "k_vapor": 0.0125,
            "k_liquid": 0.0768,
            "pr_vapor": 0.789,
            "pr_liquid": 2.901,
            "sigma": 0.00751,
            "T_critical": 86.1
        },
        "R410A": {
            "cp_vapor": 1.301,
            "cp_liquid": 1.553,
            "h_fg": 189.6,
            "rho_vapor": 35.04,
            "rho_liquid": 1119.6,
            "mu_vapor": 1.10e-5,
            "mu_liquid": 1.70e-4,
            "k_vapor": 0.0130,
            "k_liquid": 0.0759,
            "pr_vapor": 0.809,
            "pr_liquid": 2.702,
            "sigma": 0.00653,
            "T_critical": 71.4
        },
        "R22": {
            "cp_vapor": 0.665,
            "cp_liquid": 1.256,
            "h_fg": 183.4,
            "rho_vapor": 25.52,
            "rho_liquid": 1208.3,
            "mu_vapor": 1.15e-5,
            "mu_liquid": 1.95e-4,
            "k_vapor": 0.0110,
            "k_liquid": 0.0862,
            "pr_vapor": 0.782,
            "pr_liquid": 3.101,
            "sigma": 0.00821,
            "T_critical": 96.2
        },
        "R32": {
            "cp_vapor": 0.816,
            "cp_liquid": 1.423,
            "h_fg": 236.5,
            "rho_vapor": 38.21,
            "rho_liquid": 949.8,
            "mu_vapor": 1.20e-5,
            "mu_liquid": 1.65e-4,
            "k_vapor": 0.0139,
            "k_liquid": 0.1081,
            "pr_vapor": 0.719,
            "pr_liquid": 2.297,
            "sigma": 0.00582,
            "T_critical": 78.1
        },
        "R1234yf": {
            "cp_vapor": 0.884,
            "cp_liquid": 1.352,
            "h_fg": 148.2,
            "rho_vapor": 37.82,
            "rho_liquid": 1084.7,
            "mu_vapor": 1.18e-5,
            "mu_liquid": 2.05e-4,
            "k_vapor": 0.0120,
            "k_liquid": 0.0709,
            "pr_vapor": 0.849,
            "pr_liquid": 4.102,
            "sigma": 0.00621,
            "T_critical": 94.7
        },
        "Ammonia (R717)": {
            "cp_vapor": 2.182,
            "cp_liquid": 4.685,
            "h_fg": 1261.0,
            "rho_vapor": 4.256,
            "rho_liquid": 625.2,
            "mu_vapor": 9.9e-6,
            "mu_liquid": 1.35e-4,
            "k_vapor": 0.0246,
            "k_liquid": 0.5015,
            "pr_vapor": 0.878,
            "pr_liquid": 1.261,
            "sigma": 0.02342,
            "T_critical": 132.3
        }
    }
    
    # Water/glycol properties (same as before)
    ETHYLENE_GLYCOL_PROPERTIES = {
        0: {"cp": 4.186, "rho": 998.2, "mu": 0.00100, "k": 0.598, "pr": 7.01, "freeze_point": 0.0},
        10: {"cp": 4.080, "rho": 1022.0, "mu": 0.00132, "k": 0.570, "pr": 9.45, "freeze_point": -3.5},
        20: {"cp": 3.950, "rho": 1040.0, "mu": 0.00180, "k": 0.540, "pr": 13.15, "freeze_point": -7.5},
        30: {"cp": 3.780, "rho": 1057.0, "mu": 0.00258, "k": 0.510, "pr": 19.10, "freeze_point": -14.0},
        40: {"cp": 3.600, "rho": 1069.0, "mu": 0.00400, "k": 0.470, "pr": 30.60, "freeze_point": -23.0},
        50: {"cp": 3.420, "rho": 1077.0, "mu": 0.00680, "k": 0.430, "pr": 54.10, "freeze_point": -36.0},
        60: {"cp": 3.200, "rho": 1082.0, "mu": 0.01200, "k": 0.390, "pr": 98.50, "freeze_point": -52.0}
    }
    
    PROPYLENE_GLYCOL_PROPERTIES = {
        0: {"cp": 4.186, "rho": 998.2, "mu": 0.00100, "k": 0.598, "pr": 7.01, "freeze_point": 0.0},
        10: {"cp": 4.100, "rho": 1016.0, "mu": 0.00145, "k": 0.575, "pr": 10.34, "freeze_point": -3.0},
        20: {"cp": 4.000, "rho": 1028.0, "mu": 0.00210, "k": 0.555, "pr": 15.14, "freeze_point": -7.0},
        30: {"cp": 3.880, "rho": 1037.0, "mu": 0.00320, "k": 0.535, "pr": 22.18, "freeze_point": -13.0},
        40: {"cp": 3.720, "rho": 1043.0, "mu": 0.00520, "k": 0.515, "pr": 33.60, "freeze_point": -21.0},
        50: {"cp": 3.550, "rho": 1045.0, "mu": 0.00890, "k": 0.495, "pr": 53.40, "freeze_point": -33.0},
        60: {"cp": 3.350, "rho": 1044.0, "mu": 0.01600, "k": 0.475, "pr": 90.10, "freeze_point": -48.0}
    }
    
    # Tube materials properties
    TUBE_MATERIALS = {
        "Copper": {"k": 386, "density": 8960, "cost_factor": 1.0, "corrosion_resistance": "Excellent"},
        "Cu-Ni 90/10": {"k": 40, "density": 8940, "cost_factor": 1.8, "corrosion_resistance": "Excellent"},
        "Steel": {"k": 50, "density": 7850, "cost_factor": 0.6, "corrosion_resistance": "Poor"},
        "Aluminum Brass": {"k": 100, "density": 8300, "cost_factor": 1.2, "corrosion_resistance": "Good"},
        "Stainless Steel 304": {"k": 16, "density": 8000, "cost_factor": 2.5, "corrosion_resistance": "Excellent"},
        "Stainless Steel 316": {"k": 16, "density": 8000, "cost_factor": 3.0, "corrosion_resistance": "Superior"},
        "Titanium": {"k": 22, "density": 4500, "cost_factor": 8.0, "corrosion_resistance": "Superior"}
    }
    
    # Tube sizes
    TUBE_SIZES = {
        "1/4\"": 0.00635,
        "3/8\"": 0.009525,
        "1/2\"": 0.0127,
        "5/8\"": 0.015875,
        "3/4\"": 0.01905,
        "1\"": 0.0254,
        "1.25\"": 0.03175,
        "1.5\"": 0.0381
    }
    
    # Recommended velocities
    RECOMMENDED_VELOCITIES = {
        "water_tubes": {"min": 0.6, "opt": 1.2, "max": 2.5},
        "water_shell": {"min": 0.3, "opt": 0.8, "max": 1.5},
        "glycol_tubes": {"min": 0.4, "opt": 0.9, "max": 2.0},
        "glycol_shell": {"min": 0.2, "opt": 0.6, "max": 1.2},
        "refrigerant_two_phase": {"min": 0.5, "opt": 1.0, "max": 3.0},
        "refrigerant_vapor": {"min": 5.0, "opt": 10.0, "max": 30.0},
        "refrigerant_liquid": {"min": 0.5, "opt": 1.0, "max": 2.0}
    }
    
    def __init__(self):
        self.results = {}
        self.glycol_type = "ethylene"
    
    def calculate_water_glycol_properties(self, temperature: float, glycol_percentage: int, 
                                        glycol_type: str = "ethylene") -> Dict:
        """Calculate water/glycol mixture properties"""
        if glycol_type.lower() == "propylene":
            base_props = self.PROPYLENE_GLYCOL_PROPERTIES.get(glycol_percentage, 
                                                           self.PROPYLENE_GLYCOL_PROPERTIES[0])
        else:
            base_props = self.ETHYLENE_GLYCOL_PROPERTIES.get(glycol_percentage, 
                                                           self.ETHYLENE_GLYCOL_PROPERTIES[0])
        
        self.glycol_type = glycol_type.lower()
        
        # Temperature corrections
        T_ref = 20.0
        cp_factor = 1.0 - 0.0005 * (temperature - T_ref)
        rho_factor = 1.0 - 0.0002 * (temperature - T_ref)
        
        if temperature > T_ref:
            mu_factor = math.exp(-0.025 * (temperature - T_ref))
        else:
            mu_factor = math.exp(0.035 * (T_ref - temperature))
        
        k_factor = 1.0 + 0.0018 * (temperature - T_ref)
        
        cp_corrected = base_props["cp"] * cp_factor * 1000
        rho_corrected = base_props["rho"] * rho_factor
        mu_corrected = base_props["mu"] * mu_factor
        k_corrected = base_props["k"] * k_factor
        pr_corrected = mu_corrected * cp_corrected / k_corrected
        
        return {
            "cp": cp_corrected,
            "rho": rho_corrected,
            "mu": mu_corrected,
            "k": k_corrected,
            "pr": pr_corrected,
            "freeze_point": base_props["freeze_point"],
            "glycol_type": glycol_type,
            "glycol_percentage": glycol_percentage
        }
    
    def calculate_freeze_point(self, glycol_percentage: int, glycol_type: str) -> float:
        """Calculate freeze point based on glycol percentage"""
        if glycol_type.lower() == "propylene":
            props = self.PROPYLENE_GLYCOL_PROPERTIES.get(glycol_percentage, 
                                                       self.PROPYLENE_GLYCOL_PROPERTIES[0])
        else:
            props = self.ETHYLENE_GLYCOL_PROPERTIES.get(glycol_percentage, 
                                                       self.ETHYLENE_GLYCOL_PROPERTIES[0])
        return props["freeze_point"]
    
    # ==================== HEAT TRANSFER CORRELATIONS ====================
    
    def gnielinski_single_phase(self, Re: float, Pr: float, f: float = None) -> float:
        """Gnielinski correlation for single-phase turbulent flow"""
        if Re < 2300:
            # Laminar flow - constant heat flux
            return 4.36
        elif Re < 3000:
            # Transition region
            Nu_lam = 4.36
            Nu_3000 = 0.023 * 3000**0.8 * Pr**0.4
            return Nu_lam + (Re - 2300) / 700 * (Nu_3000 - Nu_lam)
        else:
            # Turbulent flow
            if f is None:
                f = (0.79 * math.log(Re) - 1.64)**-2
            Nu = (f/8) * (Re - 1000) * Pr / (1 + 12.7 * (f/8)**0.5 * (Pr**(2/3) - 1))
            return max(Nu, 4.36)
    
    def shah_evaporation(self, Re_l: float, Pr_l: float, x: float, 
                        rho_l: float, rho_v: float, D: float, G: float, 
                        h_fg: float, k_l: float) -> float:
        """Shah correlation for flow boiling in tubes"""
        if x <= 0:
            return self.gnielinski_single_phase(Re_l, Pr_l) * k_l / D
        
        Co = ((1 - x) / x)**0.8 * (rho_v / rho_l)**0.5 if x > 0 else 1e6
        Bo = G * h_fg / (k_l * 273)
        
        if Co <= 0.65:
            N = Co
        else:
            N = 0.38 * Co**-0.3
        
        if Bo > 0.0011:
            F = 14.7 * Bo**0.56 * N
        else:
            F = 15.43 * Bo**0.56 * N
        
        if Re_l < 2300:
            Nu_l = 4.36
        else:
            f_l = (0.79 * math.log(Re_l) - 1.64)**-2
            Nu_l = (f_l/8) * (Re_l - 1000) * Pr_l / (1 + 12.7 * (f_l/8)**0.5 * (Pr_l**(2/3) - 1))
        
        Nu_tp = Nu_l * (1 + 2.4e4 * Bo**1.16 + 1.37 * Co**-0.86) if x > 0 else Nu_l
        
        return Nu_tp * k_l / D
    
    def dobson_chato_condensation(self, G: float, D: float, T_sat: float, 
                                rho_l: float, rho_v: float, mu_l: float, 
                                k_l: float, cp_l: float, h_fg: float, 
                                x: float, quality_type: str = "average") -> float:
        """
        Dobson & Chato correlation for condensation in horizontal tubes
        Returns HTC in W/m²K
        """
        # Wall temperature assumption (5K below saturation for condensation)
        delta_T = 5.0  # K
        
        # Modified latent heat (accounts for liquid subcooling in film)
        h_fg_prime = h_fg * 1000 + 0.68 * cp_l * delta_T
        
        # Gravitational acceleration
        g = 9.81  # m/s²
        
        # Calculate void fraction for quality adjustment
        alpha = 1 / (1 + ((1 - x) / x) * (rho_v / rho_l)**0.667)
        
        # Nusselt number for film condensation (simplified)
        Nu_film = 0.555 * ((g * rho_l * (rho_l - rho_v) * k_l**3 * h_fg_prime) / 
                          (mu_l * D * delta_T))**0.25
        
        # Convective enhancement for higher mass flux and quality
        if quality_type == "inlet" and x > 0.7:
            # At inlet where quality is high
            Re_v = G * D / mu_l * x if mu_l > 0 else 0
            if Re_v > 0:
                enhancement = 1 + 0.8 * (x**0.8) * (Re_v/1000)**0.2
                Nu_film *= enhancement
        
        h_tp = Nu_film * k_l / D
        
        # Ensure reasonable bounds for condensation HTC
        h_tp = max(h_tp, 1000)  # Minimum 1000 W/m²K for condensation
        h_tp = min(h_tp, 8000)  # Maximum 8000 W/m²K
        
        return h_tp
    
    def calculate_single_phase_htc(self, m_dot: float, D: float, rho: float, 
                                 mu: float, k: float, cp: float, 
                                 n_passes: int = 1) -> float:
        """Calculate single-phase HTC using Gnielinski correlation"""
        # Flow area per pass
        A_flow = math.pi * D**2 / 4
        
        # Velocity
        if n_passes > 0:
            v = m_dot / (rho * A_flow * n_passes)
        else:
            v = m_dot / (rho * A_flow)
        
        # Reynolds number
        Re = rho * v * D / mu if mu > 0 else 0
        
        # Prandtl number
        Pr = mu * cp / k if k > 0 else 0
        
        # Nusselt number
        if Re > 0 and Pr > 0:
            Nu = self.gnielinski_single_phase(Re, Pr)
            h = Nu * k / D
        else:
            h = 100  # Default low value
        
        return h
    
    # ==================== ε-NTU METHOD IMPLEMENTATION ====================
    
    def epsilon_ntu_counterflow(self, NTU: float, C_r: float) -> float:
        """ε-NTU relationship for counterflow heat exchanger"""
        if C_r < 1e-6:  # C_min/C_max ≈ 0 (phase change)
            epsilon = 1 - math.exp(-NTU)
        elif abs(1 - C_r) < 1e-6:  # C_r = 1
            epsilon = NTU / (1 + NTU)
        else:
            epsilon = (1 - math.exp(-NTU * (1 - C_r))) / (1 - C_r * math.exp(-NTU * (1 - C_r)))
        return epsilon
    
    def calculate_condenser_three_region(self, m_dot_ref: float, m_dot_water: float,
                                       T_ref_in: float, T_cond: float, T_subcool_req: float,
                                       T_water_in: float, ref_props: Dict, water_props: Dict,
                                       tube_id: float, shell_h: float, tube_k: float,
                                       n_tubes: int, tube_length: float, n_passes: int,
                                       R_fouling: float = 0.00035) -> Dict:
        """
        Calculate condenser performance using ε-NTU method with three regions
        """
        # 1. Calculate required heat duties
        Q_desuperheat_req = m_dot_ref * ref_props["cp_vapor"] * 1000 * (T_ref_in - T_cond)  # W
        Q_latent_req = m_dot_ref * ref_props["h_fg"] * 1000  # W
        Q_subcool_req = m_dot_ref * ref_props["cp_liquid"] * 1000 * (T_cond - (T_cond - T_subcool_req))  # W
        Q_total_req = Q_desuperheat_req + Q_latent_req + Q_subcool_req
        
        # Subcooled outlet temperature
        T_subcooled_req = T_cond - T_subcool_req
        
        # 2. Calculate heat capacity rates
        C_ref_desuperheat = m_dot_ref * ref_props["cp_vapor"] * 1000  # W/K
        C_ref_subcool = m_dot_ref * ref_props["cp_liquid"] * 1000  # W/K
        C_water = m_dot_water * water_props["cp"]  # W/K
        
        # 3. Total heat transfer area
        tube_od = tube_id * 1.2  # Estimate OD from ID
        A_total = math.pi * tube_od * tube_length * n_tubes
        
        # 4. Calculate HTCs for each region
        
        # Mass flux for tube-side calculations
        A_flow_tube = (math.pi * tube_id**2 / 4) * n_tubes / max(n_passes, 1)
        G_ref = m_dot_ref / A_flow_tube if A_flow_tube > 0 else 0
        
        # Region 1: Desuperheating (vapor cooling)
        h_desuperheat = self.calculate_single_phase_htc(
            m_dot_ref, tube_id, ref_props["rho_vapor"], 
            ref_props["mu_vapor"], ref_props["k_vapor"],
            ref_props["cp_vapor"] * 1000, n_passes
        )
        
        # Region 2: Condensing (two-phase)
        # Use quality = 0.8 (near inlet where most condensation occurs)
        h_condense = self.dobson_chato_condensation(
            G_ref, tube_id, T_cond,
            ref_props["rho_liquid"], ref_props["rho_vapor"],
            ref_props["mu_liquid"], ref_props["k_liquid"],
            ref_props["cp_liquid"] * 1000, ref_props["h_fg"] * 1000,
            x=0.8, quality_type="inlet"
        )
        
        # Region 3: Subcooling (liquid cooling)
        h_subcool = self.calculate_single_phase_htc(
            m_dot_ref, tube_id, ref_props["rho_liquid"], 
            ref_props["mu_liquid"], ref_props["k_liquid"],
            ref_props["cp_liquid"] * 1000, n_passes
        )
        
        # Shell-side HTC (same for all regions)
        h_shell = shell_h
        
        # 5. Calculate overall U for each region
        R_wall = tube_od * math.log(tube_od / tube_id) / (2 * tube_k) if tube_k > 0 else 0
        
        U_desuperheat = 1 / (1/h_desuperheat + 1/h_shell + R_wall + R_fouling)
        U_condense = 1 / (1/h_condense + 1/h_shell + R_wall + R_fouling)
        U_subcool = 1 / (1/h_subcool + 1/h_shell + R_wall + R_fouling)
        
        # 6. Estimate area distribution based on heat duty
        if Q_total_req > 0:
            A_desuperheat = A_total * Q_desuperheat_req / Q_total_req
            A_condense = A_total * Q_latent_req / Q_total_req
            A_subcool = A_total * Q_subcool_req / Q_total_req
        else:
            # Default distribution if Q_total is 0
            A_desuperheat = A_total * 0.1
            A_condense = A_total * 0.8
            A_subcool = A_total * 0.1
        
        # Ensure areas sum to A_total
        A_sum = A_desuperheat + A_condense + A_subcool
        if A_sum > 0:
            A_desuperheat = A_desuperheat * A_total / A_sum
            A_condense = A_condense * A_total / A_sum
            A_subcool = A_subcool * A_total / A_sum
        
        # 7. ε-NTU calculation for each region (iterative approach)
        
        # Initialize temperatures
        T_water_1 = T_water_in
        Q1_achieved = 0
        Q2_achieved = 0
        Q3_achieved = 0
        
        # Region 1: Desuperheating (both streams finite capacity)
        C_min1 = min(C_ref_desuperheat, C_water)
        C_max1 = max(C_ref_desuperheat, C_water)
        C_r1 = C_min1 / C_max1 if C_max1 > 0 else 0
        
        NTU1 = U_desuperheat * A_desuperheat / C_min1 if C_min1 > 0 else 0
        epsilon1 = self.epsilon_ntu_counterflow(NTU1, C_r1)
        
        Q_max1 = C_min1 * (T_ref_in - T_water_1)
        Q1_achieved = epsilon1 * Q_max1
        
        # Water temperature after region 1
        T_water_2 = T_water_1 + Q1_achieved / C_water if C_water > 0 else T_water_1
        
        # Refrigerant temperature after region 1 (should be T_cond)
        T_ref_after1 = T_ref_in - Q1_achieved / C_ref_desuperheat if C_ref_desuperheat > 0 else T_cond
        T_ref_after1 = max(T_ref_after1, T_cond)  # Cannot go below condensing temp
        
        # Region 2: Condensing (refrigerant capacity → ∞)
        # For condensation, C_min = C_water, C_r = 0
        NTU2 = U_condense * A_condense / C_water if C_water > 0 else 0
        epsilon2 = 1 - math.exp(-NTU2)
        
        Q_max2 = C_water * (T_cond - T_water_2)
        Q2_achieved = epsilon2 * Q_max2
        
        # Water temperature after region 2
        T_water_3 = T_water_2 + Q2_achieved / C_water if C_water > 0 else T_water_2
        
        # Region 3: Subcooling (both streams finite capacity)
        C_min3 = min(C_ref_subcool, C_water)
        C_max3 = max(C_ref_subcool, C_water)
        C_r3 = C_min3 / C_max3 if C_max3 > 0 else 0
        
        NTU3 = U_subcool * A_subcool / C_min3 if C_min3 > 0 else 0
        epsilon3 = self.epsilon_ntu_counterflow(NTU3, C_r3)
        
        Q_max3 = C_min3 * (T_cond - T_water_3)
        Q3_achieved = epsilon3 * Q_max3
        
        # Water outlet temperature
        T_water_out = T_water_3 + Q3_achieved / C_water if C_water > 0 else T_water_3
        
        # Refrigerant outlet temperature (subcooled)
        Q_subcool_actual = min(Q3_achieved, Q_subcool_req)  # Cannot subcool more than available
        T_ref_out = T_cond - Q_subcool_actual / C_ref_subcool if C_ref_subcool > 0 else T_subcooled_req
        
        # Total achieved heat transfer
        Q_total_achieved = Q1_achieved + Q2_achieved + Q3_achieved
        
        # Overall effectiveness
        Q_max_total = C_water * (T_ref_in - T_water_in) if C_water > 0 else 0
        epsilon_overall = Q_total_achieved / Q_max_total if Q_max_total > 0 else 0
        
        # Overall NTU
        U_avg = (U_desuperheat * A_desuperheat + U_condense * A_condense + U_subcool * A_subcool) / A_total
        NTU_overall = U_avg * A_total / C_water if C_water > 0 else 0
        
        # Calculate LMTD for reporting (simplified)
        if T_ref_in > T_water_out and T_ref_out > T_water_in:
            dt1 = T_ref_in - T_water_out
            dt2 = T_ref_out - T_water_in
            if abs(dt1 - dt2) > 1e-6:
                LMTD = (dt1 - dt2) / math.log(dt1 / dt2)
            else:
                LMTD = (dt1 + dt2) / 2
        else:
            LMTD = 0
        
        return {
            # Heat duties
            "Q_total_req": Q_total_req / 1000,  # kW
            "Q_total_achieved": Q_total_achieved / 1000,  # kW
            "Q_desuperheat_req": Q_desuperheat_req / 1000,
            "Q_latent_req": Q_latent_req / 1000,
            "Q_subcool_req": Q_subcool_req / 1000,
            "Q_desuperheat_achieved": Q1_achieved / 1000,
            "Q_latent_achieved": Q2_achieved / 1000,
            "Q_subcool_achieved": Q3_achieved / 1000,
            
            # Temperatures
            "T_water_in": T_water_in,
            "T_water_out": T_water_out,
            "T_ref_in": T_ref_in,
            "T_ref_out": T_ref_out,
            "T_ref_out_req": T_subcooled_req,
            "T_cond": T_cond,
            
            # Heat transfer coefficients
            "h_desuperheat": h_desuperheat,
            "h_condense": h_condense,
            "h_subcool": h_subcool,
            "h_shell": h_shell,
            "U_desuperheat": U_desuperheat,
            "U_condense": U_condense,
            "U_subcool": U_subcool,
            "U_avg": U_avg,
            
            # Areas
            "A_total": A_total,
            "A_desuperheat": A_desuperheat,
            "A_condense": A_condense,
            "A_subcool": A_subcool,
            
            # Performance metrics
            "epsilon_overall": epsilon_overall,
            "NTU_overall": NTU_overall,
            "LMTD": LMTD,
            "NTU1": NTU1,
            "NTU2": NTU2,
            "NTU3": NTU3,
            "epsilon1": epsilon1,
            "epsilon2": epsilon2,
            "epsilon3": epsilon3,
            
            # Additional info
            "C_water": C_water,
            "C_ref_desuperheat": C_ref_desuperheat,
            "C_ref_subcool": C_ref_subcool,
            "G_ref": G_ref
        }
    
    def calculate_evaporator_two_region(self, m_dot_ref: float, m_dot_water: float,
                                      T_evap: float, inlet_quality: float, superheat_req: float,
                                      T_water_in: float, ref_props: Dict, water_props: Dict,
                                      tube_id: float, shell_h: float, tube_k: float,
                                      n_tubes: int, tube_length: float, n_passes: int,
                                      R_fouling: float = 0.00035) -> Dict:
        """
        Calculate evaporator performance using ε-NTU method with two regions
        """
        # Convert quality from percentage to fraction
        x_in = inlet_quality / 100.0
        
        # 1. Calculate required heat duties
        # Latent heat: evaporate remaining liquid (1 - x_in)
        Q_latent_req = m_dot_ref * (1 - x_in) * ref_props["h_fg"] * 1000  # W
        
        # Superheating
        T_superheated_req = T_evap + superheat_req
        Q_superheat_req = m_dot_ref * ref_props["cp_vapor"] * 1000 * superheat_req  # W
        
        Q_total_req = Q_latent_req + Q_superheat_req
        
        # 2. Calculate heat capacity rates
        # For evaporation region: refrigerant capacity → ∞ (phase change)
        # For superheat region: finite capacity
        C_ref_superheat = m_dot_ref * ref_props["cp_vapor"] * 1000  # W/K
        C_water = m_dot_water * water_props["cp"]  # W/K
        
        # 3. Total heat transfer area
        tube_od = tube_id * 1.2  # Estimate OD from ID
        A_total = math.pi * tube_od * tube_length * n_tubes
        
        # 4. Calculate HTCs for each region
        
        # Mass flux for tube-side calculations
        A_flow_tube = (math.pi * tube_id**2 / 4) * n_tubes / max(n_passes, 1)
        G_ref = m_dot_ref / A_flow_tube if A_flow_tube > 0 else 0
        
        # Region 1: Evaporation (two-phase)
        # Use average quality
        x_avg = (x_in + 1.0) / 2.0
        Re_l = G_ref * tube_id / ref_props["mu_liquid"] if ref_props["mu_liquid"] > 0 else 0
        Pr_l = ref_props["pr_liquid"]
        
        h_evap = self.shah_evaporation(
            Re_l, Pr_l, x_avg,
            ref_props["rho_liquid"], ref_props["rho_vapor"],
            tube_id, G_ref, ref_props["h_fg"] * 1000,
            ref_props["k_liquid"]
        )
        
        # Region 2: Superheating (vapor heating)
        h_superheat = self.calculate_single_phase_htc(
            m_dot_ref, tube_id, ref_props["rho_vapor"], 
            ref_props["mu_vapor"], ref_props["k_vapor"],
            ref_props["cp_vapor"] * 1000, n_passes
        )
        
        # Shell-side HTC
        h_shell = shell_h
        
        # 5. Calculate overall U for each region
        R_wall = tube_od * math.log(tube_od / tube_id) / (2 * tube_k) if tube_k > 0 else 0
        
        U_evap = 1 / (1/h_evap + 1/h_shell + R_wall + R_fouling)
        U_superheat = 1 / (1/h_superheat + 1/h_shell + R_wall + R_fouling)
        
        # 6. Estimate area distribution based on heat duty
        if Q_total_req > 0:
            A_evap = A_total * Q_latent_req / Q_total_req
            A_superheat = A_total * Q_superheat_req / Q_total_req
        else:
            A_evap = A_total * 0.8
            A_superheat = A_total * 0.2
        
        # Ensure areas sum to A_total
        A_sum = A_evap + A_superheat
        if A_sum > 0:
            A_evap = A_evap * A_total / A_sum
            A_superheat = A_superheat * A_total / A_sum
        
        # 7. ε-NTU calculation for each region
        
        # Initialize temperatures (counterflow assumed)
        # For evaporator: water enters hot, leaves cold
        
        # Region 1: Evaporation (refrigerant capacity → ∞)
        # Water temperature entering evaporation region (after superheat region in counterflow)
        # For simplicity, assume linear temperature distribution
        
        # First pass: calculate with estimated temperatures
        T_water_mid_est = T_water_in - (Q_latent_req / C_water) if C_water > 0 else T_water_in
        T_water_mid_est = max(T_water_mid_est, T_evap + 5)  # Minimum approach
        
        # Evaporation region (C_ref → ∞)
        NTU_evap = U_evap * A_evap / C_water if C_water > 0 else 0
        epsilon_evap = 1 - math.exp(-NTU_evap)
        
        Q_max_evap = C_water * (T_water_mid_est - T_evap)
        Q_evap_achieved = epsilon_evap * Q_max_evap
        
        # Water temperature after evaporation
        T_water_after_evap = T_water_mid_est - Q_evap_achieved / C_water if C_water > 0 else T_water_mid_est
        
        # Region 2: Superheating (both streams finite capacity)
        C_min_superheat = min(C_ref_superheat, C_water)
        C_max_superheat = max(C_ref_superheat, C_water)
        C_r_superheat = C_min_superheat / C_max_superheat if C_max_superheat > 0 else 0
        
        NTU_superheat = U_superheat * A_superheat / C_min_superheat if C_min_superheat > 0 else 0
        epsilon_superheat = self.epsilon_ntu_counterflow(NTU_superheat, C_r_superheat)
        
        Q_max_superheat = C_min_superheat * (T_water_after_evap - T_evap)
        Q_superheat_achieved = epsilon_superheat * Q_max_superheat
        
        # Water outlet temperature
        T_water_out = T_water_after_evap - Q_superheat_achieved / C_water if C_water > 0 else T_water_after_evap
        
        # Refrigerant outlet temperature (superheated)
        T_ref_out = T_evap + Q_superheat_achieved / C_ref_superheat if C_ref_superheat > 0 else T_superheated_req
        
        # Total achieved heat transfer
        Q_total_achieved = Q_evap_achieved + Q_superheat_achieved
        
        # Overall effectiveness
        Q_max_total = C_water * (T_water_in - T_evap) if C_water > 0 else 0
        epsilon_overall = Q_total_achieved / Q_max_total if Q_max_total > 0 else 0
        
        # Overall NTU
        U_avg = (U_evap * A_evap + U_superheat * A_superheat) / A_total
        NTU_overall = U_avg * A_total / C_water if C_water > 0 else 0
        
        # Calculate LMTD
        dt1 = T_water_in - T_ref_out
        dt2 = T_water_out - T_evap
        if dt1 > 0 and dt2 > 0 and abs(dt1 - dt2) > 1e-6:
            LMTD = (dt1 - dt2) / math.log(dt1 / dt2)
        else:
            LMTD = (dt1 + dt2) / 2
        
        return {
            # Heat duties
            "Q_total_req": Q_total_req / 1000,  # kW
            "Q_total_achieved": Q_total_achieved / 1000,  # kW
            "Q_latent_req": Q_latent_req / 1000,
            "Q_superheat_req": Q_superheat_req / 1000,
            "Q_latent_achieved": Q_evap_achieved / 1000,
            "Q_superheat_achieved": Q_superheat_achieved / 1000,
            
            # Temperatures
            "T_water_in": T_water_in,
            "T_water_out": T_water_out,
            "T_ref_in": T_evap,  # Evaporating temperature
            "T_ref_out": T_ref_out,
            "T_ref_out_req": T_superheated_req,
            "T_evap": T_evap,
            
            # Heat transfer coefficients
            "h_evap": h_evap,
            "h_superheat": h_superheat,
            "h_shell": h_shell,
            "U_evap": U_evap,
            "U_superheat": U_superheat,
            "U_avg": U_avg,
            
            # Areas
            "A_total": A_total,
            "A_evap": A_evap,
            "A_superheat": A_superheat,
            
            # Performance metrics
            "epsilon_overall": epsilon_overall,
            "NTU_overall": NTU_overall,
            "LMTD": LMTD,
            "NTU_evap": NTU_evap,
            "NTU_superheat": NTU_superheat,
            "epsilon_evap": epsilon_evap,
            "epsilon_superheat": epsilon_superheat,
            
            # Additional info
            "C_water": C_water,
            "C_ref_superheat": C_ref_superheat,
            "G_ref": G_ref,
            "inlet_quality": inlet_quality
        }
    
    # ==================== MAIN DESIGN FUNCTIONS ====================
    
    def design_dx_evaporator(self, inputs: Dict) -> Dict:
        """Design DX evaporator using ε-NTU method"""
        
        # Extract inputs
        refrigerant = inputs["refrigerant"]
        m_dot_ref = inputs["m_dot_ref"]  # kg/s
        T_evap = inputs["T_ref"]
        superheat_req = inputs["delta_T_sh_sc"]
        inlet_quality = inputs.get("inlet_quality", 20)  # percentage
        
        # Water/glycol side
        glycol_percent = inputs["glycol_percentage"]
        glycol_type = inputs.get("glycol_type", "ethylene")
        m_dot_sec_L = inputs["m_dot_sec"] / 3600  # L/s to kg/s
        T_sec_in = inputs["T_sec_in"]
        
        # Geometry
        tube_size = inputs["tube_size"]
        tube_material = inputs["tube_material"]
        tube_thickness = inputs["tube_thickness"] / 1000  # m
        tube_pitch = inputs["tube_pitch"] / 1000  # m
        n_passes = inputs["n_passes"]
        n_baffles = inputs["n_baffles"]
        n_tubes = inputs["n_tubes"]
        tube_length = inputs["tube_length"]
        tube_layout = inputs["tube_layout"].lower()
        
        # Get fluid properties
        ref_props = self.REFRIGERANTS[refrigerant]
        sec_props = self.calculate_water_glycol_properties(T_sec_in, glycol_percent, glycol_type)
        
        # Convert secondary flow to kg/s
        m_dot_sec_kg = m_dot_sec_L * sec_props["rho"] / 1000
        
        # Tube dimensions
        tube_od = self.TUBE_SIZES[tube_size]
        tube_id = max(tube_od - 2 * tube_thickness, tube_od * 0.8)
        
        # Calculate shell diameter and shell-side HTC
        shell_diameter = self.calculate_shell_diameter(tube_od, n_tubes, tube_pitch, tube_layout)
        
        # Calculate equivalent diameter for shell side
        if tube_layout == "triangular":
            D_e = 4 * (0.866 * tube_pitch**2 - 0.5 * math.pi * tube_od**2) / (math.pi * tube_od)
        else:
            D_e = 4 * (tube_pitch**2 - 0.25 * math.pi * tube_od**2) / (math.pi * tube_od)
        
        # Calculate shell-side flow area and velocity
        shell_cross_area = math.pi * shell_diameter**2 / 4
        tube_bundle_area = n_tubes * math.pi * tube_od**2 / 4
        shell_flow_area = (shell_cross_area - tube_bundle_area) * 0.4
        
        # Shell-side mass flux and Reynolds
        G_sec = m_dot_sec_kg / shell_flow_area if shell_flow_area > 0 else 0
        Re_shell = G_sec * D_e / sec_props["mu"]
        
        # Shell-side HTC
        h_shell = self.calculate_shell_side_htc(Re_shell, sec_props["pr"], D_e, sec_props["k"], tube_layout)
        
        # Tube material thermal conductivity
        tube_k = self.TUBE_MATERIALS[tube_material]["k"]
        
        # Calculate evaporator performance using ε-NTU method
        results = self.calculate_evaporator_two_region(
            m_dot_ref, m_dot_sec_kg, T_evap, inlet_quality, superheat_req,
            T_sec_in, ref_props, sec_props, tube_id, h_shell, tube_k,
            n_tubes, tube_length, n_passes
        )
        
        # Calculate additional parameters
        baffle_spacing = tube_length / (n_baffles + 1)
        
        # Flow velocities
        A_flow_tube = (math.pi * tube_id**2 / 4) * n_tubes / max(n_passes, 1)
        G_ref = m_dot_ref / A_flow_tube if A_flow_tube > 0 else 0
        
        # For two-phase flow, use homogeneous density
        x_avg = (inlet_quality/100.0 + 1.0) / 2.0
        rho_tp = 1 / (x_avg/ref_props["rho_vapor"] + (1-x_avg)/ref_props["rho_liquid"])
        v_ref = G_ref / rho_tp
        v_sec = G_sec / sec_props["rho"]
        
        # Pressure drops (simplified)
        # Tube side (two-phase evaporation) - simplified calculation
        Re_l = G_ref * tube_id / ref_props["mu_liquid"] if ref_props["mu_liquid"] > 0 else 0
        if Re_l > 2300:
            f_tube = (0.79 * math.log(Re_l) - 1.64)**-2
        else:
            f_tube = 64 / Re_l if Re_l > 0 else 0.05
        
        # Two-phase multiplier (simplified)
        phi_tp = 1 + 2.5 / x_avg if x_avg > 0 else 1
        dp_tube = f_tube * (tube_length * n_passes / tube_id) * (rho_tp * v_ref**2 / 2) * phi_tp
        
        # Shell side pressure drop
        if Re_shell < 2300:
            f_shell = 64 / Re_shell if Re_shell > 0 else 0.2
        else:
            f_shell = 0.2 * Re_shell**-0.2
        
        dp_shell = f_shell * (tube_length / D_e) * n_baffles * (sec_props["rho"] * v_sec**2 / 2)
        
        # Check velocities
        sec_velocity_status = self.check_velocity_status(v_sec, glycol_percent, "shell")
        ref_velocity_status = self.check_velocity_status(v_ref, 0, "refrigerant_two_phase")
        
        # Refrigerant distribution check
        m_dot_per_tube = m_dot_ref / n_tubes * 3600
        distribution_status = "Good" if m_dot_per_tube >= 3.6 else "Marginal" if m_dot_per_tube >= 2.0 else "Poor"
        
        # Freeze protection check
        freeze_point = self.calculate_freeze_point(glycol_percent, glycol_type)
        freeze_risk = "High" if results["T_water_out"] < freeze_point + 2 else "Medium" if results["T_water_out"] < freeze_point + 3 else "Low"
        
        # Calculate area ratio (achieved vs required)
        # For required area, use simplified LMTD method
        Q_required = results["Q_total_req"] * 1000  # W
        U_avg = results["U_avg"]
        LMTD = results["LMTD"] if results["LMTD"] > 0 else 5.0  # Default
        
        A_required = Q_required / (U_avg * LMTD) if U_avg > 0 and LMTD > 0 else 0
        area_ratio = results["A_total"] / A_required if A_required > 0 else 0
        
        # Design status
        design_status = self.determine_design_status(
            results["epsilon_overall"], results["A_total"], A_required,
            results["Q_total_achieved"], results["Q_total_req"]
        )
        
        # Store comprehensive results
        self.results = {
            # Basic info
            "heat_exchanger_type": "DX Evaporator",
            "design_method": "Mass Flow Input",
            
            # Thermal performance
            "heat_duty_required_kw": results["Q_total_req"],
            "heat_duty_achieved_kw": results["Q_total_achieved"],
            "kw_difference": results["Q_total_achieved"] - results["Q_total_req"],
            "kw_match_percentage": (results["Q_total_achieved"] / results["Q_total_req"] * 100) if results["Q_total_req"] > 0 else 0,
            
            # Heat duty breakdown
            "q_latent_req_kw": results["Q_latent_req"],
            "q_superheat_req_kw": results["Q_superheat_req"],
            "q_latent_achieved_kw": results["Q_latent_achieved"],
            "q_superheat_achieved_kw": results["Q_superheat_achieved"],
            
            # ε-NTU results
            "effectiveness": results["epsilon_overall"],
            "ntu": results["NTU_overall"],
            "ntu_evap": results["NTU_evap"],
            "ntu_superheat": results["NTU_superheat"],
            "epsilon_evap": results["epsilon_evap"],
            "epsilon_superheat": results["epsilon_superheat"],
            
            # Heat transfer coefficients
            "overall_u": results["U_avg"],
            "h_tube_evap": results["h_evap"],
            "h_tube_superheat": results["h_superheat"],
            "h_shell": results["h_shell"],
            "u_evap": results["U_evap"],
            "u_superheat": results["U_superheat"],
            "lmtd": results["LMTD"],
            
            # Temperatures
            "t_sec_in": results["T_water_in"],
            "t_sec_out": results["T_water_out"],
            "t_ref_in": results["T_evap"],
            "t_ref_out_required": results["T_ref_out_req"],
            "t_ref_out_achieved": results["T_ref_out"],
            "superheat_difference": results["T_ref_out"] - results["T_ref_out_req"],
            "water_deltaT": abs(results["T_water_out"] - results["T_water_in"]),
            "superheat_req": superheat_req,
            
            # Flow rates
            "refrigerant_mass_flow_kg_s": m_dot_ref,
            "refrigerant_mass_flow_kg_hr": m_dot_ref * 3600,
            "inlet_quality_percent": inlet_quality,
            "water_vol_flow_L_hr": m_dot_sec_L * 3600,
            "water_mass_flow_kg_hr": m_dot_sec_kg * 3600,
            "flow_per_tube_kg_hr": m_dot_per_tube,
            
            # Geometry
            "shell_diameter_m": shell_diameter,
            "tube_pitch_mm": tube_pitch * 1000,
            "pitch_ratio": tube_pitch / tube_od if tube_od > 0 else 0,
            "tube_od_mm": tube_od * 1000,
            "tube_id_mm": tube_id * 1000,
            "area_total_m2": results["A_total"],
            "area_evap_m2": results["A_evap"],
            "area_superheat_m2": results["A_superheat"],
            "area_required_m2": A_required,
            "area_ratio": area_ratio,
            "baffle_spacing_m": baffle_spacing,
            "n_baffles": n_baffles,
            
            # Flow parameters
            "velocity_tube_ms": v_ref,
            "velocity_shell_ms": v_sec,
            "velocity_shell_status": sec_velocity_status,
            "velocity_tube_status": ref_velocity_status,
            "dp_tube_kpa": dp_tube / 1000,
            "dp_shell_kpa": dp_shell / 1000,
            "reynolds_tube": Re_l,
            "reynolds_shell": Re_shell,
            "mass_flux_tube": G_ref,
            "mass_flux_shell": G_sec,
            
            # Capacity rates
            "c_water": results["C_water"],
            "c_ref_superheat": results["C_ref_superheat"],
            
            # DX-specific
            "distribution_status": distribution_status,
            "freeze_point_c": freeze_point,
            "freeze_risk": freeze_risk,
            "glycol_type": glycol_type,
            "glycol_percentage": glycol_percent,
            
            # Design status
            "design_status": design_status,
        }
        
        return self.results
    
    def design_condenser(self, inputs: Dict) -> Dict:
        """Design condenser using ε-NTU method with three regions"""
        
        # Extract inputs
        refrigerant = inputs["refrigerant"]
        m_dot_ref = inputs["m_dot_ref"]  # kg/s
        T_ref_in_superheated = inputs["T_ref_in_superheated"]
        T_cond = inputs["T_ref"]
        subcool_req = inputs["delta_T_sh_sc"]
        
        # Water/glycol side
        glycol_percent = inputs["glycol_percentage"]
        glycol_type = inputs.get("glycol_type", "ethylene")
        m_dot_sec_L = inputs["m_dot_sec"] / 3600  # L/s
        T_sec_in = inputs["T_sec_in"]
        
        # Geometry
        tube_size = inputs["tube_size"]
        tube_material = inputs["tube_material"]
        tube_thickness = inputs["tube_thickness"] / 1000  # m
        tube_pitch = inputs["tube_pitch"] / 1000  # m
        n_passes = inputs["n_passes"]
        n_baffles = inputs["n_baffles"]
        n_tubes = inputs["n_tubes"]
        tube_length = inputs["tube_length"]
        tube_layout = inputs["tube_layout"].lower()
        
        # Get fluid properties
        ref_props = self.REFRIGERANTS[refrigerant]
        sec_props = self.calculate_water_glycol_properties(T_sec_in, glycol_percent, glycol_type)
        
        # Convert secondary flow to kg/s
        m_dot_sec_kg = m_dot_sec_L * sec_props["rho"] / 1000
        
        # Tube dimensions
        tube_od = self.TUBE_SIZES[tube_size]
        tube_id = max(tube_od - 2 * tube_thickness, tube_od * 0.8)
        
        # Calculate shell diameter and shell-side HTC
        shell_diameter = self.calculate_shell_diameter(tube_od, n_tubes, tube_pitch, tube_layout)
        
        # Calculate equivalent diameter for shell side
        if tube_layout == "triangular":
            D_e = 4 * (0.866 * tube_pitch**2 - 0.5 * math.pi * tube_od**2) / (math.pi * tube_od)
        else:
            D_e = 4 * (tube_pitch**2 - 0.25 * math.pi * tube_od**2) / (math.pi * tube_od)
        
        # Calculate shell-side flow area and velocity
        shell_cross_area = math.pi * shell_diameter**2 / 4
        tube_bundle_area = n_tubes * math.pi * tube_od**2 / 4
        shell_flow_area = (shell_cross_area - tube_bundle_area) * 0.4
        
        # Shell-side mass flux and Reynolds
        G_sec = m_dot_sec_kg / shell_flow_area if shell_flow_area > 0 else 0
        Re_shell = G_sec * D_e / sec_props["mu"]
        
        # Shell-side HTC
        h_shell = self.calculate_shell_side_htc(Re_shell, sec_props["pr"], D_e, sec_props["k"], tube_layout)
        
        # Tube material thermal conductivity
        tube_k = self.TUBE_MATERIALS[tube_material]["k"]
        
        # Calculate condenser performance using ε-NTU method
        results = self.calculate_condenser_three_region(
            m_dot_ref, m_dot_sec_kg, T_ref_in_superheated, T_cond, subcool_req,
            T_sec_in, ref_props, sec_props, tube_id, h_shell, tube_k,
            n_tubes, tube_length, n_passes
        )
        
        # Calculate additional parameters
        baffle_spacing = tube_length / (n_baffles + 1)
        
        # Flow velocities
        A_flow_tube = (math.pi * tube_id**2 / 4) * n_tubes / max(n_passes, 1)
        G_ref = results["G_ref"]
        
        # For two-phase flow in condensation region, use average density
        # Quality ~0.5 in condensation region
        rho_tp = 1 / (0.5/ref_props["rho_vapor"] + 0.5/ref_props["rho_liquid"])
        v_ref = G_ref / rho_tp
        v_sec = G_sec / sec_props["rho"]
        
        # Pressure drops (simplified)
        # Tube side - simplified for condensation
        Re_l = G_ref * tube_id / ref_props["mu_liquid"] if ref_props["mu_liquid"] > 0 else 0
        if Re_l > 2300:
            f_tube = (0.79 * math.log(Re_l) - 1.64)**-2
        else:
            f_tube = 64 / Re_l if Re_l > 0 else 0.05
        
        # Two-phase multiplier for condensation (simplified)
        phi_tp = 1 + 1.5 / 0.5 if 0.5 > 0 else 1  # For average quality 0.5
        dp_tube = f_tube * (tube_length * n_passes / tube_id) * (rho_tp * v_ref**2 / 2) * phi_tp
        
        # Shell side pressure drop
        if Re_shell < 2300:
            f_shell = 64 / Re_shell if Re_shell > 0 else 0.2
        else:
            f_shell = 0.2 * Re_shell**-0.2
        
        dp_shell = f_shell * (tube_length / D_e) * n_baffles * (sec_props["rho"] * v_sec**2 / 2)
        
        # Check velocities
        sec_velocity_status = self.check_velocity_status(v_sec, glycol_percent, "shell")
        ref_velocity_status = self.check_velocity_status(v_ref, 0, "refrigerant_two_phase")
        
        # Calculate area ratio (achieved vs required)
        Q_required = results["Q_total_req"] * 1000  # W
        U_avg = results["U_avg"]
        LMTD = results["LMTD"] if results["LMTD"] > 0 else 5.0  # Default
        
        A_required = Q_required / (U_avg * LMTD) if U_avg > 0 and LMTD > 0 else 0
        area_ratio = results["A_total"] / A_required if A_required > 0 else 0
        
        # Design status
        design_status = self.determine_design_status(
            results["epsilon_overall"], results["A_total"], A_required,
            results["Q_total_achieved"], results["Q_total_req"]
        )
        
        # Store comprehensive results
        self.results = {
            # Basic info
            "heat_exchanger_type": "Condenser",
            "design_method": "Mass Flow Input",
            
            # Thermal performance
            "heat_duty_required_kw": results["Q_total_req"],
            "heat_duty_achieved_kw": results["Q_total_achieved"],
            "kw_difference": results["Q_total_achieved"] - results["Q_total_req"],
            "kw_match_percentage": (results["Q_total_achieved"] / results["Q_total_req"] * 100) if results["Q_total_req"] > 0 else 0,
            
            # Heat duty breakdown
            "q_desuperheat_req_kw": results["Q_desuperheat_req"],
            "q_latent_req_kw": results["Q_latent_req"],
            "q_subcool_req_kw": results["Q_subcool_req"],
            "q_desuperheat_achieved_kw": results["Q_desuperheat_achieved"],
            "q_latent_achieved_kw": results["Q_latent_achieved"],
            "q_subcool_achieved_kw": results["Q_subcool_achieved"],
            
            # ε-NTU results
            "effectiveness": results["epsilon_overall"],
            "ntu": results["NTU_overall"],
            "ntu_desuperheat": results["NTU1"],
            "ntu_condense": results["NTU2"],
            "ntu_subcool": results["NTU3"],
            "epsilon_desuperheat": results["epsilon1"],
            "epsilon_condense": results["epsilon2"],
            "epsilon_subcool": results["epsilon3"],
            
            # Heat transfer coefficients
            "overall_u": results["U_avg"],
            "h_tube_desuperheat": results["h_desuperheat"],
            "h_tube_condense": results["h_condense"],
            "h_tube_subcool": results["h_subcool"],
            "h_shell": results["h_shell"],
            "u_desuperheat": results["U_desuperheat"],
            "u_condense": results["U_condense"],
            "u_subcool": results["U_subcool"],
            "lmtd": results["LMTD"],
            
            # Temperatures
            "t_sec_in": results["T_water_in"],
            "t_sec_out": results["T_water_out"],
            "t_ref_in_superheated": results["T_ref_in"],
            "t_ref_condensing": results["T_cond"],
            "t_ref_out_required": results["T_ref_out_req"],
            "t_ref_out_achieved": results["T_ref_out"],
            "subcool_difference": results["T_ref_out"] - results["T_ref_out_req"],
            "water_deltaT": abs(results["T_water_out"] - results["T_water_in"]),
            "subcool_req": subcool_req,
            
            # Flow rates
            "refrigerant_mass_flow_kg_s": m_dot_ref,
            "refrigerant_mass_flow_kg_hr": m_dot_ref * 3600,
            "water_vol_flow_L_hr": m_dot_sec_L * 3600,
            "water_mass_flow_kg_hr": m_dot_sec_kg * 3600,
            
            # Geometry
            "shell_diameter_m": shell_diameter,
            "tube_pitch_mm": tube_pitch * 1000,
            "pitch_ratio": tube_pitch / tube_od if tube_od > 0 else 0,
            "tube_od_mm": tube_od * 1000,
            "tube_id_mm": tube_id * 1000,
            "area_total_m2": results["A_total"],
            "area_desuperheat_m2": results["A_desuperheat"],
            "area_condense_m2": results["A_condense"],
            "area_subcool_m2": results["A_subcool"],
            "area_required_m2": A_required,
            "area_ratio": area_ratio,
            "baffle_spacing_m": baffle_spacing,
            "n_baffles": n_baffles,
            
            # Flow parameters
            "velocity_tube_ms": v_ref,
            "velocity_shell_ms": v_sec,
            "velocity_shell_status": sec_velocity_status,
            "velocity_tube_status": ref_velocity_status,
            "dp_tube_kpa": dp_tube / 1000,
            "dp_shell_kpa": dp_shell / 1000,
            "reynolds_tube": Re_l,
            "reynolds_shell": Re_shell,
            "mass_flux_tube": G_ref,
            "mass_flux_shell": G_sec,
            
            # Capacity rates
            "c_water": results["C_water"],
            "c_ref_desuperheat": results["C_ref_desuperheat"],
            "c_ref_subcool": results["C_ref_subcool"],
            
            # Glycol info
            "glycol_type": glycol_type,
            "glycol_percentage": glycol_percent,
            
            # Design status
            "design_status": design_status,
        }
        
        return self.results
    
    # ==================== HELPER FUNCTIONS ====================
    
    def calculate_shell_diameter(self, tube_od: float, n_tubes: int, pitch: float,
                               tube_layout: str = "triangular") -> float:
        """Calculate shell diameter based on tube count and pitch"""
        if tube_layout.lower() == "triangular":
            tubes_per_row = math.sqrt(n_tubes / 0.866)
            bundle_width = tubes_per_row * pitch
        else:
            tubes_per_row = math.sqrt(n_tubes)
            bundle_width = tubes_per_row * pitch
        
        if bundle_width < 0.3:
            clearance = 0.010
        elif bundle_width < 0.6:
            clearance = 0.015
        else:
            clearance = 0.020
        
        shell_diameter = bundle_width + 2 * clearance
        
        return max(shell_diameter, 0.1)
    
    def calculate_shell_side_htc(self, Re: float, Pr: float, D_e: float,
                               k: float, tube_layout: str) -> float:
        """Calculate shell-side HTC using Bell-Delaware method (simplified)"""
        if Re < 100:
            if tube_layout == "triangular":
                Nu = 1.0
            else:
                Nu = 0.9
        elif Re < 1000:
            if tube_layout == "triangular":
                Nu = 0.6 * Re**0.5 * Pr**0.33
            else:
                Nu = 0.5 * Re**0.5 * Pr**0.33
        else:
            if tube_layout == "triangular":
                Nu = 0.36 * Re**0.55 * Pr**0.33
            else:
                Nu = 0.31 * Re**0.6 * Pr**0.33
        
        return Nu * k / D_e
    
    def check_velocity_status(self, velocity: float, glycol_percent: int, flow_type: str) -> Dict:
        """Check velocity against recommended ranges"""
        if flow_type == "shell":
            if glycol_percent > 0:
                rec = self.RECOMMENDED_VELOCITIES["glycol_shell"]
            else:
                rec = self.RECOMMENDED_VELOCITIES["water_shell"]
        elif flow_type == "tubes":
            if glycol_percent > 0:
                rec = self.RECOMMENDED_VELOCITIES["glycol_tubes"]
            else:
                rec = self.RECOMMENDED_VELOCITIES["water_tubes"]
        else:
            rec = self.RECOMMENDED_VELOCITIES.get(flow_type, self.RECOMMENDED_VELOCITIES["water_shell"])
        
        if velocity < rec["min"]:
            status = "Too Low"
            color = "red"
        elif velocity < rec["opt"]:
            status = "Low"
            color = "orange"
        elif velocity <= rec["max"]:
            status = "Optimal"
            color = "green"
        else:
            status = "Too High"
            color = "red"
        
        return {
            "velocity": velocity,
            "status": status,
            "color": color,
            "min": rec["min"],
            "opt": rec["opt"],
            "max": rec["max"]
        }
    
    def determine_design_status(self, effectiveness: float, area_total: float, 
                              area_required: float, kw_achieved: float, 
                              kw_required: float) -> str:
        """Determine overall design status"""
        area_ratio = area_total / area_required if area_required > 0 else 0
        kw_ratio = kw_achieved / kw_required if kw_required > 0 else 0
        
        # Check key performance indicators
        kw_match = abs(kw_ratio - 1.0) < 0.15  # Within 15%
        area_adequate = area_ratio >= 0.9
        effective_enough = effectiveness >= 0.6
        
        if kw_match and area_adequate and effective_enough:
            return "Adequate"
        elif kw_ratio >= 0.8 and area_ratio >= 0.8 and effectiveness >= 0.5:
            return "Marginal"
        else:
            return "Inadequate"

# Helper function for number input with +/- buttons
def number_input_with_buttons(label: str, min_value: float, max_value: float, 
                            value: float, step: float, key: str, format: str = "%.1f",
                            help_text: str = None) -> float:
    """Create a number input with +/- buttons"""
    
    if key not in st.session_state:
        st.session_state[key] = value
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("−", key=f"{key}_minus"):
            st.session_state[key] = max(min_value, st.session_state[key] - step)
    
    with col2:
        st.markdown(f"<div class='input-label'>{label}</div>", unsafe_allow_html=True)
        if help_text:
            st.caption(help_text)
        value_input = st.number_input(
            label="",
            min_value=min_value,
            max_value=max_value,
            value=st.session_state[key],
            step=step,
            key=f"{key}_input",
            label_visibility="collapsed",
            format=format
        )
    
    with col3:
        if st.button("＋", key=f"{key}_plus"):
            st.session_state[key] = min(max_value, st.session_state[key] + step)
    
    return value_input

def create_input_section():
    """Create input section for DX evaporator and condenser design"""
    st.sidebar.header("⚙️ DX Heat Exchanger Design")
    
    inputs = {}
    
    # Heat exchanger type
    inputs["hex_type"] = st.sidebar.radio(
        "Heat Exchanger Type",
        ["DX Evaporator", "Condenser"],
        help="DX Evaporator: Refrigerant evaporates in tubes, water in shell\nCondenser: Refrigerant condenses in tubes, water in shell"
    )
    
    # Display badge
    if inputs["hex_type"] == "DX Evaporator":
        st.sidebar.markdown('<span class="dx-badge">DX Type</span>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<span class="condenser-badge">Condenser</span>', unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Design method - Only Mass Flow Input
    st.sidebar.subheader("📊 Design Input Method")
    st.sidebar.info("**Mass Flow Input**\nEnter refrigerant mass flow from compressor specs")
    
    # Initialize designer for refrigerant list
    designer = DXHeatExchangerDesign()
    
    # Refrigerant parameters
    st.sidebar.subheader("🔧 Refrigerant Parameters")
    
    inputs["refrigerant"] = st.sidebar.selectbox(
        "Refrigerant Type",
        list(designer.REFRIGERANTS.keys())
    )
    
    # Refrigerant mass flow in kg/s
    inputs["m_dot_ref"] = number_input_with_buttons(
        label="Refrigerant Mass Flow (kg/s)",
        min_value=0.01,
        max_value=10.0,
        value=0.5,
        step=0.01,
        key="m_dot_ref",
        format="%.3f",
        help_text="From compressor specification sheet"
    )
    
    # Temperature parameters - CONDENSER
    if inputs["hex_type"] == "Condenser":
        st.sidebar.subheader("🌡️ Condenser Parameters")
        
        # Superheated refrigerant inlet temperature
        inputs["T_ref_in_superheated"] = number_input_with_buttons(
            label="Superheated Refrigerant Inlet (°C)",
            min_value=50.0,
            max_value=150.0,
            value=80.0,
            step=1.0,
            key="T_ref_superheated",
            format="%.1f",
            help_text="Temperature of superheated vapor from compressor"
        )
        
        inputs["T_ref"] = number_input_with_buttons(
            label="Condensing Temperature (°C)",
            min_value=20.0,
            max_value=80.0,
            value=45.0,
            step=1.0,
            key="T_cond",
            format="%.1f"
        )
        
        inputs["delta_T_sh_sc"] = number_input_with_buttons(
            label="Required Subcool at Exit (K)",
            min_value=0.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
            key="subcool",
            format="%.1f"
        )
    
    # Temperature parameters - EVAPORATOR
    else:
        st.sidebar.subheader("🌡️ Evaporator Parameters")
        
        inputs["T_ref"] = number_input_with_buttons(
            label="Evaporating Temperature (°C)",
            min_value=-50.0,
            max_value=20.0,
            value=5.0,
            step=1.0,
            key="T_evap",
            format="%.1f"
        )
        
        # Inlet quality from expansion valve
        inputs["inlet_quality"] = number_input_with_buttons(
            label="Inlet Quality from TXV (%)",
            min_value=0.0,
            max_value=100.0,
            value=20.0,
            step=1.0,
            key="inlet_quality",
            format="%.1f",
            help_text="Quality of refrigerant entering evaporator (0-100%)"
        )
        
        inputs["delta_T_sh_sc"] = number_input_with_buttons(
            label="Required Superheat at Exit (K)",
            min_value=3.0,
            max_value=15.0,
            value=5.0,
            step=0.5,
            key="superheat",
            format="%.1f",
            help_text="DX evaporators require 3-8K superheat for proper TXV operation"
        )
    
    st.sidebar.markdown("---")
    
    # Water/Glycol parameters
    st.sidebar.subheader("💧 Water/Glycol Side")
    
    glycol_options = ["Water Only", "Water + Ethylene Glycol", "Water + Propylene Glycol (Food Grade)"]
    glycol_choice = st.sidebar.radio(
        "Fluid Type",
        glycol_options
    )
    
    if "Ethylene" in glycol_choice:
        inputs["glycol_type"] = "ethylene"
        glycol_label = "EG"
        bg_color = "#E0F2FE"
        text_color = "#0C4A6E"
    elif "Propylene" in glycol_choice:
        inputs["glycol_type"] = "propylene"
        glycol_label = "PG"
        bg_color = "#F0FDF4"
        text_color = "#166534"
    else:
        inputs["glycol_type"] = "water"
        glycol_label = "Water"
        bg_color = "#EFF6FF"
        text_color = "#1E40AF"
    
    st.sidebar.markdown(f"""
    <div style="background-color: {bg_color}; color: {text_color}; padding: 0.5rem; 
                border-radius: 0.5rem; text-align: center; font-weight: bold; margin: 0.5rem 0;">
        {glycol_label}
    </div>
    """, unsafe_allow_html=True)
    
    # Glycol percentage
    if "Glycol" in glycol_choice:
        inputs["glycol_percentage"] = number_input_with_buttons(
            label="Glycol Percentage",
            min_value=0,
            max_value=60,
            value=20,
            step=5,
            key="glycol_percent",
            format="%.0f",
            help_text="Higher percentage = lower freeze point, higher viscosity"
        )
        
        freeze_point = designer.calculate_freeze_point(inputs["glycol_percentage"], inputs["glycol_type"])
        st.sidebar.caption(f"Freeze point: {freeze_point:.1f}°C")
    else:
        inputs["glycol_percentage"] = 0
    
    # Water inlet temperature
    inputs["T_sec_in"] = number_input_with_buttons(
        label="Water Inlet Temperature (°C)",
        min_value=-20.0 if "Glycol" in glycol_choice else 0.0,
        max_value=80.0,
        value=25.0 if inputs["hex_type"] == "Condenser" else 12.0,
        step=1.0,
        key="T_water_in",
        format="%.1f"
    )
    
    # Water flow rate
    inputs["m_dot_sec"] = number_input_with_buttons(
        label="Water Flow Rate (L/hr)",
        min_value=100.0,
        max_value=100000.0,
        value=5000.0,
        step=100.0,
        key="water_flow",
        format="%.0f"
    )
    
    # Flow arrangement (now informational only, calculations use counterflow)
    st.sidebar.radio(
        "Flow Arrangement",
        ["Counter", "Parallel"],
        index=0,
        disabled=True,
        help="ε-NTU method assumes counterflow (most efficient)"
    )
    inputs["flow_arrangement"] = "counter"
    
    st.sidebar.markdown("---")
    
    # Geometry parameters
    st.sidebar.subheader("📐 Geometry Parameters")
    
    inputs["tube_size"] = st.sidebar.selectbox(
        "Tube Size",
        list(designer.TUBE_SIZES.keys())
    )
    
    inputs["tube_material"] = st.sidebar.selectbox(
        "Tube Material",
        list(designer.TUBE_MATERIALS.keys()),
        help="Copper: Best heat transfer, expensive\nCu-Ni: Corrosion resistant for seawater\nStainless: Corrosion resistant, lower heat transfer"
    )
    
    inputs["tube_thickness"] = number_input_with_buttons(
        label="Tube Thickness (mm)",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
        key="tube_thickness",
        format="%.1f"
    )
    
    inputs["tube_pitch"] = number_input_with_buttons(
        label="Tube Pitch (mm)",
        min_value=15.0,
        max_value=100.0,
        value=25.0,
        step=0.5,
        key="tube_pitch",
        format="%.1f"
    )
    
    # Calculate and display pitch ratio
    tube_od = designer.TUBE_SIZES[inputs["tube_size"]] * 1000  # mm
    pitch_ratio = inputs["tube_pitch"] / tube_od if tube_od > 0 else 0
    st.sidebar.caption(f"Pitch/OD ratio: {pitch_ratio:.2f}")
    
    if pitch_ratio < 1.25:
        st.sidebar.warning("⚠️ Pitch ratio < 1.25 may be too tight")
    elif pitch_ratio > 1.5:
        st.sidebar.info("ℹ️ Pitch ratio > 1.5 is good for cleaning")
    
    inputs["n_passes"] = st.sidebar.selectbox(
        "Tube Passes",
        [1, 2, 4, 6],
        help="Number of times refrigerant passes through tubes"
    )
    
    inputs["n_baffles"] = int(number_input_with_buttons(
        label="Number of Baffles",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        key="n_baffles",
        format="%.0f",
        help_text="More baffles = better heat transfer but higher pressure drop"
    ))
    
    inputs["n_tubes"] = int(number_input_with_buttons(
        label="Number of Tubes",
        min_value=1,
        max_value=500,
        value=100,
        step=1,
        key="n_tubes",
        format="%.0f"
    ))
    
    inputs["tube_length"] = number_input_with_buttons(
        label="Tube Length (m)",
        min_value=0.5,
        max_value=10.0,
        value=3.0,
        step=0.1,
        key="tube_length",
        format="%.1f"
    )
    
    inputs["tube_layout"] = st.sidebar.radio(
        "Tube Layout",
        ["Triangular", "Square"],
        help="Triangular: Higher heat transfer, more compact\nSquare: Easier cleaning, lower pressure drop"
    )
    
    return inputs

def display_velocity_indicator(velocity: float, status: Dict):
    """Display velocity with color-coded indicator"""
    if status["color"] == "green":
        css_class = "velocity-good"
    elif status["color"] == "orange":
        css_class = "velocity-low"
    else:
        css_class = "velocity-high"
    
    return f'<span class="{css_class}">{velocity:.2f} m/s ({status["status"]})</span>'

def display_glycol_badge(glycol_type: str, percentage: int):
    """Display glycol type badge"""
    if glycol_type == "ethylene":
        return '<span class="glycol-ethylene">Ethylene Glycol {:.0f}%</span>'.format(percentage)
    elif glycol_type == "propylene":
        return '<span class="glycol-propylene">Propylene Glycol {:.0f}%</span>'.format(percentage)
    else:
        return '<span style="background-color: #EFF6FF; color: #1E40AF; padding: 0.25rem 0.5rem; border-radius: 0.5rem; font-size: 0.85rem;">Water Only</span>'

def display_kw_comparison(kw_required: float, kw_achieved: float):
    """Display kW comparison with color coding"""
    diff = kw_achieved - kw_required
    percent_diff = (diff / kw_required * 100) if kw_required > 0 else 0
    
    if abs(percent_diff) <= 10:  # Within 10%
        css_class = "kw-comparison kw-match"
        status_icon = "✅"
    else:
        css_class = "kw-comparison kw-mismatch"
        status_icon = "⚠️"
    
    html = f"""
    <div class="{css_class}">
        <strong>{status_icon} Heat Duty Comparison</strong><br>
        <strong>Required:</strong> {kw_required:.1f} kW<br>
        <strong>Achieved:</strong> {kw_achieved:.1f} kW<br>
        <strong>Difference:</strong> {diff:+.1f} kW ({percent_diff:+.1f}%)
    </div>
    """
    return html

def display_temp_comparison(temp_required: float, temp_achieved: float, label: str):
    """Display temperature comparison"""
    diff = temp_achieved - temp_required
    
    if abs(diff) <= 1:
        status = "✅ Good match"
        color = "#10B981"
    elif abs(diff) <= 3:
        status = "⚠️ Close"
        color = "#F59E0B"
    else:
        status = "❌ Significant difference"
        color = "#EF4444"
    
    html = f"""
    <div class="temp-comparison">
        <strong>{label}</strong><br>
        <strong>Required:</strong> {temp_required:.1f} °C<br>
        <strong>Achieved:</strong> {temp_achieved:.1f} °C<br>
        <strong>Difference:</strong> {diff:+.1f} °C<br>
        <span style="color: {color}; font-weight: bold;">{status}</span>
    </div>
    """
    return html

def display_region_performance(results: Dict, hex_type: str):
    """Display region-by-region performance"""
    
    if hex_type == "Condenser":
        st.markdown("### 📊 Three-Region Condenser Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="region-box region-desuperheat">', unsafe_allow_html=True)
            st.markdown("**🔴 Desuperheating Region**")
            st.write(f"Area: {results['area_desuperheat_m2']:.1f} m²")
            st.write(f"HTC: {results['h_tube_desuperheat']:.0f} W/m²K")
            st.write(f"U: {results['u_desuperheat']:.0f} W/m²K")
            st.write(f"NTU: {results['ntu_desuperheat']:.2f}")
            st.write(f"ε: {results['epsilon_desuperheat']:.3f}")
            st.write(f"Heat: {results['q_desuperheat_achieved_kw']:.1f} kW")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="region-box region-condense">', unsafe_allow_html=True)
            st.markdown("**🔵 Condensing Region**")
            st.write(f"Area: {results['area_condense_m2']:.1f} m²")
            st.write(f"HTC: {results['h_tube_condense']:.0f} W/m²K")
            st.write(f"U: {results['u_condense']:.0f} W/m²K")
            st.write(f"NTU: {results['ntu_condense']:.2f}")
            st.write(f"ε: {results['epsilon_condense']:.3f}")
            st.write(f"Heat: {results['q_latent_achieved_kw']:.1f} kW")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="region-box region-subcool">', unsafe_allow_html=True)
            st.markdown("**🟢 Subcooling Region**")
            st.write(f"Area: {results['area_subcool_m2']:.1f} m²")
            st.write(f"HTC: {results['h_tube_subcool']:.0f} W/m²K")
            st.write(f"U: {results['u_subcool']:.0f} W/m²K")
            st.write(f"NTU: {results['ntu_subcool']:.2f}")
            st.write(f"ε: {results['epsilon_subcool']:.3f}")
            st.write(f"Heat: {results['q_subcool_achieved_kw']:.1f} kW")
            st.markdown('</div>', unsafe_allow_html=True)
    
    else:  # Evaporator
        st.markdown("### 📊 Two-Region Evaporator Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="region-box region-condense">', unsafe_allow_html=True)
            st.markdown("**🔵 Evaporation Region**")
            st.write(f"Area: {results['area_evap_m2']:.1f} m²")
            st.write(f"HTC: {results['h_tube_evap']:.0f} W/m²K")
            st.write(f"U: {results['u_evap']:.0f} W/m²K")
            st.write(f"NTU: {results['ntu_evap']:.2f}")
            st.write(f"ε: {results['epsilon_evap']:.3f}")
            st.write(f"Heat: {results['q_latent_achieved_kw']:.1f} kW")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="region-box region-desuperheat">', unsafe_allow_html=True)
            st.markdown("**🔴 Superheating Region**")
            st.write(f"Area: {results['area_superheat_m2']:.1f} m²")
            st.write(f"HTC: {results['h_tube_superheat']:.0f} W/m²K")
            st.write(f"U: {results['u_superheat']:.0f} W/m²K")
            st.write(f"NTU: {results['ntu_superheat']:.2f}")
            st.write(f"ε: {results['epsilon_superheat']:.3f}")
            st.write(f"Heat: {results['q_superheat_achieved_kw']:.1f} kW")
            st.markdown('</div>', unsafe_allow_html=True)

def display_results(results: Dict, inputs: Dict):
    """Display calculation results"""
    
    # Header with badges
    if results["heat_exchanger_type"] == "DX Evaporator":
        header_html = f"""
        <div style='display: flex; align-items: center;'>
            <h2>📊 DX Evaporator Design Results</h2>
            <span class="dx-badge">DX Type</span>
            {display_glycol_badge(results['glycol_type'], results['glycol_percentage'])}
            <span style="margin-left: 10px; background-color: #FEF3C7; color: #92400E; padding: 0.25rem 0.5rem; border-radius: 0.5rem; font-size: 0.85rem;">
                Mass Flow Input | ε-NTU Method
            </span>
        </div>
        """
    else:
        header_html = f"""
        <div style='display: flex; align-items: center;'>
            <h2>📊 Condenser Design Results</h2>
            <span class="condenser-badge">Condenser</span>
            {display_glycol_badge(results['glycol_type'], results['glycol_percentage'])}
            <span style="margin-left: 10px; background-color: #FEF3C7; color: #92400E; padding: 0.25rem 0.5rem; border-radius: 0.5rem; font-size: 0.85rem;">
                Mass Flow Input | ε-NTU Method
            </span>
        </div>
        """
    
    st.markdown(header_html, unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Required Heat Duty", f"{results['heat_duty_required_kw']:.1f} kW")
        st.caption(f"Achieved: {results['heat_duty_achieved_kw']:.1f} kW")
    
    with col2:
        status_color = "normal" if results['design_status'] == "Adequate" else "off" if results['design_status'] == "Marginal" else "inverse"
        st.metric("Design Status", results['design_status'], delta_color=status_color)
        st.caption(f"kW Match: {results['kw_match_percentage']:.1f}%")
    
    with col3:
        st.metric("Effectiveness", f"{results['effectiveness']:.3f}")
        st.caption(f"NTU: {results['ntu']:.2f}")
    
    with col4:
        st.metric("Area Ratio", f"{results['area_ratio']:.2f}")
        st.caption(f"{results['area_total_m2']:.1f} m² / {results['area_required_m2']:.1f} m²")
    
    st.markdown("---")
    
    # kW Comparison Section
    st.markdown("### 🔥 Heat Duty Comparison")
    st.markdown(display_kw_comparison(
        results['heat_duty_required_kw'], 
        results['heat_duty_achieved_kw']
    ), unsafe_allow_html=True)
    
    # Temperature Comparison Section
    st.markdown("### 🌡️ Temperature Comparison")
    
    if results["heat_exchanger_type"] == "DX Evaporator":
        st.markdown(display_temp_comparison(
            results['t_ref_out_required'],
            results['t_ref_out_achieved'],
            "Outlet Superheated Temperature"
        ), unsafe_allow_html=True)
    else:
        st.markdown(display_temp_comparison(
            results['t_ref_out_required'],
            results['t_ref_out_achieved'],
            "Outlet Subcooled Temperature"
        ), unsafe_allow_html=True)
    
    # Region-by-region performance
    display_region_performance(results, results["heat_exchanger_type"])
    
    st.markdown("---")
    
    # Flow parameters
    st.markdown("### 💧 Flow Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Refrigerant Side (Tubes)")
        st.write(f"**Mass Flow:** {results['refrigerant_mass_flow_kg_s']:.3f} kg/s ({results['refrigerant_mass_flow_kg_hr']:.1f} kg/hr)")
        
        if results["heat_exchanger_type"] == "DX Evaporator":
            st.write(f"**Inlet Quality:** {results['inlet_quality_percent']:.1f}%")
            st.write(f"**Evaporating Temp:** {results['t_ref_in']:.1f} °C")
            st.write(f"**Outlet Temp Required:** {results['t_ref_out_required']:.1f} °C")
            st.write(f"**Outlet Temp Achieved:** {results['t_ref_out_achieved']:.1f} °C")
        else:
            st.write(f"**Inlet Temp (Superheated):** {results['t_ref_in_superheated']:.1f} °C")
            st.write(f"**Condensing Temp:** {results['t_ref_condensing']:.1f} °C")
            st.write(f"**Outlet Temp Required:** {results['t_ref_out_required']:.1f} °C")
            st.write(f"**Outlet Temp Achieved:** {results['t_ref_out_achieved']:.1f} °C")
        
        st.markdown(f"**Velocity:** {display_velocity_indicator(results['velocity_tube_ms'], results['velocity_tube_status'])}", unsafe_allow_html=True)
        st.write(f"**Pressure Drop:** {results['dp_tube_kpa']:.2f} kPa")
        st.write(f"**Reynolds:** {results['reynolds_tube']:,.0f}")
        st.write(f"**Mass Flux:** {results['mass_flux_tube']:.1f} kg/m²s")
        
        if results["heat_exchanger_type"] == "DX Evaporator":
            st.write(f"**Distribution:** {results['distribution_status']}")
            st.write(f"**Flow per Tube:** {results['flow_per_tube_kg_hr']:.1f} kg/hr")
    
    with col2:
        st.markdown("#### Water/Glycol Side (Shell)")
        st.write(f"**Volumetric Flow:** {results['water_vol_flow_L_hr']:,.0f} L/hr")
        st.write(f"**Mass Flow:** {results['water_mass_flow_kg_hr']:,.0f} kg/hr")
        st.write(f"**Inlet Temp:** {results['t_sec_in']:.1f} °C")
        st.write(f"**Outlet Temp:** {results['t_sec_out']:.1f} °C")
        st.write(f"**ΔT:** {results['water_deltaT']:.1f} K")
        
        st.markdown(f"**Velocity:** {display_velocity_indicator(results['velocity_shell_ms'], results['velocity_shell_status'])}", unsafe_allow_html=True)
        st.write(f"**Pressure Drop:** {results['dp_shell_kpa']:.2f} kPa")
        st.write(f"**Reynolds:** {results['reynolds_shell']:,.0f}")
        st.write(f"**Mass Flux:** {results['mass_flux_shell']:.1f} kg/m²s")
        
        if results["heat_exchanger_type"] == "DX Evaporator" and results['glycol_percentage'] > 0:
            st.write(f"**Freeze Point:** {results['freeze_point_c']:.1f}°C")
            st.write(f"**Freeze Risk:** {results['freeze_risk']}")
    
    st.markdown("---")
    
    # Thermal performance
    st.markdown("### ⚙️ Thermal Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Heat Transfer")
        st.write(f"**Overall U:** {results['overall_u']:.1f} W/m²K")
        st.write(f"**Shell HTC:** {results['h_shell']:.0f} W/m²K")
        st.write(f"**LMTD:** {results['lmtd']:.1f} K")
        
        if results["heat_exchanger_type"] == "Condenser":
            st.write(f"**Desuperheat HTC:** {results['h_tube_desuperheat']:.0f} W/m²K")
            st.write(f"**Condense HTC:** {results['h_tube_condense']:.0f} W/m²K")
            st.write(f"**Subcool HTC:** {results['h_tube_subcool']:.0f} W/m²K")
        else:
            st.write(f"**Evaporation HTC:** {results['h_tube_evap']:.0f} W/m²K")
            st.write(f"**Superheat HTC:** {results['h_tube_superheat']:.0f} W/m²K")
    
    with col2:
        st.markdown("#### Geometry")
        st.write(f"**Shell Diameter:** {results['shell_diameter_m']*1000:.0f} mm")
        st.write(f"**Tube OD:** {results['tube_od_mm']:.1f} mm")
        st.write(f"**Tube ID:** {results['tube_id_mm']:.1f} mm")
        st.write(f"**Tube Pitch:** {results['tube_pitch_mm']:.1f} mm")
        st.write(f"**Pitch/OD Ratio:** {results['pitch_ratio']:.2f}")
        st.write(f"**Tube Layout:** {inputs['tube_layout']}")
        st.write(f"**Baffle Spacing:** {results['baffle_spacing_m']:.2f} m")
        st.write(f"**Number of Baffles:** {results['n_baffles']}")
    
    with col3:
        st.markdown("#### Capacity Rates")
        st.write(f"**Water Capacity:** {results['c_water']/1000:.1f} kW/K")
        
        if results["heat_exchanger_type"] == "Condenser":
            st.write(f"**Refrigerant (Desuperheat):** {results.get('c_ref_desuperheat', 0)/1000:.1f} kW/K")
            st.write(f"**Refrigerant (Subcool):** {results.get('c_ref_subcool', 0)/1000:.1f} kW/K")
        else:
            st.write(f"**Refrigerant (Superheat):** {results.get('c_ref_superheat', 0)/1000:.1f} kW/K")
        
        st.markdown("#### Flow Characteristics")
        st.write(f"**Area Total:** {results['area_total_m2']:.2f} m²")
        st.write(f"**Area Required:** {results['area_required_m2']:.2f} m²")
        st.write(f"**Area Ratio:** {results['area_ratio']:.2f}")
    
    st.markdown("---")
    
    # Design recommendations
    st.markdown("### 💡 Design Recommendations")
    
    if results['design_status'] == "Inadequate":
        st.error(f"""
        **DESIGN INADEQUATE**
        
        **Issues:**
        1. Heat duty mismatch: {results['kw_match_percentage']:.1f}% (target 85-115%)
        2. Effectiveness: {results['effectiveness']:.3f} (target ≥0.6)
        3. Area ratio: {results['area_ratio']:.2f} (target ≥0.9)
        
        **Solutions:**
        1. {'Increase water flow rate' if results['heat_duty_achieved_kw'] < results['heat_duty_required_kw'] else 'Decrease water flow rate'}
        2. Add more tubes or increase tube length
        3. Reduce tube pitch to fit more tubes
        4. Consider enhanced tube surfaces
        5. {'Increase refrigerant mass flow if compressor allows' if results['heat_duty_achieved_kw'] < results['heat_duty_required_kw'] else 'Check if compressor can handle lower load'}
        """)
    elif results['design_status'] == "Marginal":
        st.warning(f"""
        **DESIGN MARGINAL**
        
        **Considerations:**
        1. Heat duty match: {results['kw_match_percentage']:.1f}% (target 85-115%)
        2. Effectiveness: {results['effectiveness']:.3f} (target ≥0.6)
        3. Area ratio: {results['area_ratio']:.2f} (target ≥0.9)
        
        **Recommendations:**
        1. Fine-tune water flow rate (±10-20%)
        2. Consider minor geometry adjustments
        3. Monitor performance in operation
        4. Consider safety factor of 10-20%
        """)
    else:
        st.success(f"""
        **DESIGN ADEQUATE** ✅
        
        **Performance Summary:**
        1. Heat duty match: {results['kw_match_percentage']:.1f}% (good)
        2. Effectiveness: {results['effectiveness']:.3f} (good)
        3. Area ratio: {results['area_ratio']:.2f} (adequate)
        4. Temperature match: {abs(results.get('superheat_difference', results.get('subcool_difference', 0))):.1f}°C difference
        
        **Design is ready for detailed engineering.**
        """)
    
    # Velocity optimization
    if results['velocity_shell_status']['color'] == "red":
        st.info(f"""
        **VELOCITY OPTIMIZATION NEEDED**
        
        Current water velocity: {results['velocity_shell_ms']:.2f} m/s ({results['velocity_shell_status']['status']})
        
        **Target range:** {results['velocity_shell_status']['min']:.1f}-{results['velocity_shell_status']['max']:.1f} m/s
        **Optimal:** {results['velocity_shell_status']['opt']:.1f} m/s
        
        **Adjust water flow rate accordingly.**
        """)
    
    # DX-specific warnings
    if results["heat_exchanger_type"] == "DX Evaporator":
        st.markdown("### ⚠️ DX-Specific Considerations")
        
        if results.get('distribution_status', 'Good') == "Poor":
            st.error(f"""
            **POOR REFRIGERANT DISTRIBUTION**
            
            Flow per tube ({results['flow_per_tube_kg_hr']:.1f} kg/hr) is too low for good distribution.
            
            **Solutions:**
            1. Reduce number of tubes
            2. Increase refrigerant flow (if compressor allows)
            3. Use enhanced distributor design
            4. Consider individual TXVs per circuit
            """)
        
        if results.get('freeze_risk', 'Low') in ["High", "Medium"]:
            st.warning(f"""
            **FREEZE RISK DETECTED**
            
            Water outlet temperature ({results['t_sec_out']:.1f}°C) is close to freeze point ({results['freeze_point_c']:.1f}°C).
            
            **Recommendations:**
            1. Increase glycol percentage
            2. Increase water flow rate
            3. Add freeze protection controls
            4. Monitor temperature closely
            """)
        
        if results.get('superheat_difference', 0) < -2.0:
            st.warning(f"""
            **LOW ACHIEVED SUPERHEAT** ({results.get('superheat_difference', 0):.1f} K difference)
            
            TXV requires 3-8K superheat for proper operation.
            
            **Causes:**
            1. Oversized evaporator
            2. High water flow rate
            3. TXV malfunction
            
            **Solutions:**
            1. Adjust TXV setting
            2. Reduce water flow
            3. Check refrigerant charge
            """)
    
    # Export results
    st.markdown("---")
    st.markdown("### 📥 Export Design Report")
    
    if st.button("Download Engineering Report", type="primary"):
        # Create comprehensive report
        report_data = []
        
        if results["heat_exchanger_type"] == "DX Evaporator":
            report_data = {
                "Parameter": [
                    "Heat Exchanger Type", "Design Method", "Refrigerant",
                    "Refrigerant Mass Flow (kg/s)", "Refrigerant Mass Flow (kg/hr)",
                    "Inlet Quality (%)", "Evaporating Temperature (°C)",
                    "Required Superheat (K)", "Required Outlet Temp (°C)",
                    "Achieved Outlet Temp (°C)", "Temperature Difference (°C)",
                    "Required Heat Duty (kW)", "Achieved Heat Duty (kW)",
                    "Heat Duty Difference (kW)", "Heat Duty Match (%)",
                    "Required Latent Heat (kW)", "Achieved Latent Heat (kW)",
                    "Required Superheat Duty (kW)", "Achieved Superheat Duty (kW)",
                    "Glycol Type", "Glycol Percentage", "Freeze Point (°C)",
                    "Effectiveness (ε)", "NTU", "Overall U (W/m²K)",
                    "Evaporation HTC (W/m²K)", "Superheat HTC (W/m²K)", 
                    "Shell HTC (W/m²K)", "LMTD (K)",
                    "Water Flow Rate (L/hr)", "Water Mass Flow (kg/hr)",
                    "Water Inlet Temp (°C)", "Water Outlet Temp (°C)",
                    "Water ΔT (K)", "Shell Diameter (mm)", "Tube OD (mm)",
                    "Tube ID (mm)", "Tube Pitch (mm)", "Pitch/OD Ratio",
                    "Tube Layout", "Number of Tubes", "Tube Length (m)",
                    "Tube Passes", "Number of Baffles", "Baffle Spacing (m)",
                    "Total Area (m²)", "Evaporation Area (m²)", "Superheat Area (m²)",
                    "Required Area (m²)", "Area Ratio",
                    "Water Velocity (m/s)", "Velocity Status", "Refrigerant Velocity (m/s)",
                    "Shell ΔP (kPa)", "Tube ΔP (kPa)", "Shell Reynolds",
                    "Tube Reynolds", "Flow per Tube (kg/hr)", "Distribution Status",
                    "Freeze Risk", "Design Status"
                ],
                "Value": [
                    results["heat_exchanger_type"], "Mass Flow Input (ε-NTU)", inputs["refrigerant"],
                    f"{results['refrigerant_mass_flow_kg_s']:.3f}", f"{results['refrigerant_mass_flow_kg_hr']:.1f}",
                    f"{results['inlet_quality_percent']:.1f}", f"{results['t_ref_in']:.1f}",
                    f"{results.get('superheat_req', 0):.1f}", f"{results['t_ref_out_required']:.1f}",
                    f"{results['t_ref_out_achieved']:.1f}", f"{results.get('superheat_difference', 0):.1f}",
                    f"{results['heat_duty_required_kw']:.1f}", f"{results['heat_duty_achieved_kw']:.1f}",
                    f"{results['kw_difference']:.1f}", f"{results['kw_match_percentage']:.1f}",
                    f"{results['q_latent_req_kw']:.1f}", f"{results['q_latent_achieved_kw']:.1f}",
                    f"{results['q_superheat_req_kw']:.1f}", f"{results['q_superheat_achieved_kw']:.1f}",
                    results['glycol_type'].title(), f"{results['glycol_percentage']}%", f"{results.get('freeze_point_c', 0):.1f}",
                    f"{results['effectiveness']:.3f}", f"{results['ntu']:.2f}", f"{results['overall_u']:.1f}",
                    f"{results['h_tube_evap']:.0f}", f"{results['h_tube_superheat']:.0f}",
                    f"{results['h_shell']:.0f}", f"{results['lmtd']:.1f}",
                    f"{results['water_vol_flow_L_hr']:,.0f}", f"{results['water_mass_flow_kg_hr']:,.0f}",
                    f"{results['t_sec_in']:.1f}", f"{results['t_sec_out']:.1f}",
                    f"{results['water_deltaT']:.1f}", f"{results['shell_diameter_m']*1000:.0f}",
                    f"{results['tube_od_mm']:.1f}", f"{results['tube_id_mm']:.1f}",
                    f"{results['tube_pitch_mm']:.1f}", f"{results['pitch_ratio']:.2f}",
                    inputs['tube_layout'], str(inputs['n_tubes']), f"{inputs['tube_length']}",
                    str(inputs['n_passes']), str(results['n_baffles']), f"{results['baffle_spacing_m']:.3f}",
                    f"{results['area_total_m2']:.2f}", f"{results.get('area_evap_m2', 0):.2f}",
                    f"{results.get('area_superheat_m2', 0):.2f}", f"{results['area_required_m2']:.2f}",
                    f"{results['area_ratio']:.2f}", f"{results['velocity_shell_ms']:.2f}",
                    results['velocity_shell_status']['status'], f"{results['velocity_tube_ms']:.2f}",
                    f"{results['dp_shell_kpa']:.2f}", f"{results['dp_tube_kpa']:.2f}",
                    f"{results['reynolds_shell']:,.0f}", f"{results['reynolds_tube']:,.0f}",
                    f"{results.get('flow_per_tube_kg_hr', 0):.1f}", results.get('distribution_status', 'N/A'),
                    results.get('freeze_risk', 'N/A'), results['design_status']
                ],
                "Unit": [
                    "", "", "",
                    "kg/s", "kg/hr",
                    "%", "°C",
                    "K", "°C",
                    "°C", "°C",
                    "kW", "kW",
                    "kW", "%",
                    "kW", "kW",
                    "kW", "kW",
                    "", "", "°C",
                    "-", "-", "W/m²K",
                    "W/m²K", "W/m²K",
                    "W/m²K", "K",
                    "L/hr", "kg/hr",
                    "°C", "°C",
                    "K", "mm",
                    "mm", "mm", "mm",
                    "-", "", "",
                    "m", "", "",
                    "m", "m²", "m²",
                    "m²", "m²", "-",
                    "m/s", "", "m/s",
                    "kPa", "kPa",
                    "-", "-", "kg/hr",
                    "", "", ""
                ]
            }
        else:
            report_data = {
                "Parameter": [
                    "Heat Exchanger Type", "Design Method", "Refrigerant",
                    "Refrigerant Mass Flow (kg/s)", "Refrigerant Mass Flow (kg/hr)",
                    "Superheated Inlet Temp (°C)", "Condensing Temperature (°C)",
                    "Required Subcool (K)", "Required Outlet Temp (°C)",
                    "Achieved Outlet Temp (°C)", "Temperature Difference (°C)",
                    "Required Heat Duty (kW)", "Achieved Heat Duty (kW)",
                    "Heat Duty Difference (kW)", "Heat Duty Match (%)",
                    "Required Desuperheat Duty (kW)", "Achieved Desuperheat Duty (kW)",
                    "Required Latent Heat (kW)", "Achieved Latent Heat (kW)",
                    "Required Subcool Duty (kW)", "Achieved Subcool Duty (kW)",
                    "Glycol Type", "Glycol Percentage",
                    "Effectiveness (ε)", "NTU", "Overall U (W/m²K)",
                    "Desuperheat HTC (W/m²K)", "Condense HTC (W/m²K)", 
                    "Subcool HTC (W/m²K)", "Shell HTC (W/m²K)", "LMTD (K)",
                    "Water Flow Rate (L/hr)", "Water Mass Flow (kg/hr)",
                    "Water Inlet Temp (°C)", "Water Outlet Temp (°C)",
                    "Water ΔT (K)", "Shell Diameter (mm)", "Tube OD (mm)",
                    "Tube ID (mm)", "Tube Pitch (mm)", "Pitch/OD Ratio",
                    "Tube Layout", "Number of Tubes", "Tube Length (m)",
                    "Tube Passes", "Number of Baffles", "Baffle Spacing (m)",
                    "Total Area (m²)", "Desuperheat Area (m²)", "Condense Area (m²)",
                    "Subcool Area (m²)", "Required Area (m²)", "Area Ratio",
                    "Water Velocity (m/s)", "Velocity Status", "Refrigerant Velocity (m/s)",
                    "Shell ΔP (kPa)", "Tube ΔP (kPa)", "Shell Reynolds",
                    "Tube Reynolds", "Design Status"
                ],
                "Value": [
                    results["heat_exchanger_type"], "Mass Flow Input (ε-NTU)", inputs["refrigerant"],
                    f"{results['refrigerant_mass_flow_kg_s']:.3f}", f"{results['refrigerant_mass_flow_kg_hr']:.1f}",
                    f"{results['t_ref_in_superheated']:.1f}", f"{results['t_ref_condensing']:.1f}",
                    f"{results['subcool_req']:.1f}", f"{results['t_ref_out_required']:.1f}",
                    f"{results['t_ref_out_achieved']:.1f}", f"{results['subcool_difference']:.1f}",
                    f"{results['heat_duty_required_kw']:.1f}", f"{results['heat_duty_achieved_kw']:.1f}",
                    f"{results['kw_difference']:.1f}", f"{results['kw_match_percentage']:.1f}",
                    f"{results['q_desuperheat_req_kw']:.1f}", f"{results['q_desuperheat_achieved_kw']:.1f}",
                    f"{results['q_latent_req_kw']:.1f}", f"{results['q_latent_achieved_kw']:.1f}",
                    f"{results['q_subcool_req_kw']:.1f}", f"{results['q_subcool_achieved_kw']:.1f}",
                    results['glycol_type'].title(), f"{results['glycol_percentage']}%",
                    f"{results['effectiveness']:.3f}", f"{results['ntu']:.2f}", f"{results['overall_u']:.1f}",
                    f"{results['h_tube_desuperheat']:.0f}", f"{results['h_tube_condense']:.0f}",
                    f"{results['h_tube_subcool']:.0f}", f"{results['h_shell']:.0f}", f"{results['lmtd']:.1f}",
                    f"{results['water_vol_flow_L_hr']:,.0f}", f"{results['water_mass_flow_kg_hr']:,.0f}",
                    f"{results['t_sec_in']:.1f}", f"{results['t_sec_out']:.1f}",
                    f"{results['water_deltaT']:.1f}", f"{results['shell_diameter_m']*1000:.0f}",
                    f"{results['tube_od_mm']:.1f}", f"{results['tube_id_mm']:.1f}",
                    f"{results['tube_pitch_mm']:.1f}", f"{results['pitch_ratio']:.2f}",
                    inputs['tube_layout'], str(inputs['n_tubes']), f"{inputs['tube_length']}",
                    str(inputs['n_passes']), str(results['n_baffles']), f"{results['baffle_spacing_m']:.3f}",
                    f"{results['area_total_m2']:.2f}", f"{results.get('area_desuperheat_m2', 0):.2f}",
                    f"{results.get('area_condense_m2', 0):.2f}", f"{results.get('area_subcool_m2', 0):.2f}",
                    f"{results['area_required_m2']:.2f}", f"{results['area_ratio']:.2f}",
                    f"{results['velocity_shell_ms']:.2f}", results['velocity_shell_status']['status'],
                    f"{results['velocity_tube_ms']:.2f}", f"{results['dp_shell_kpa']:.2f}",
                    f"{results['dp_tube_kpa']:.2f}", f"{results['reynolds_shell']:,.0f}",
                    f"{results['reynolds_tube']:,.0f}", results['design_status']
                ],
                "Unit": [
                    "", "", "",
                    "kg/s", "kg/hr",
                    "°C", "°C",
                    "K", "°C",
                    "°C", "°C",
                    "kW", "kW",
                    "kW", "%",
                    "kW", "kW",
                    "kW", "kW",
                    "kW", "kW",
                    "", "",
                    "-", "-", "W/m²K",
                    "W/m²K", "W/m²K",
                    "W/m²K", "W/m²K", "K",
                    "L/hr", "kg/hr",
                    "°C", "°C",
                    "K", "mm",
                    "mm", "mm", "mm",
                    "-", "", "",
                    "m", "", "",
                    "m", "m²", "m²",
                    "m²", "m²", "m²",
                    "-", "m/s", "",
                    "m/s", "kPa", "kPa",
                    "-", "-", ""
                ]
            }
        
        df_report = pd.DataFrame(report_data)
        csv = df_report.to_csv(index=False)
        
        st.download_button(
            label="📊 Download CSV Report",
            data=csv,
            file_name="dx_heat_exchanger_design.csv",
            mime="text/csv"
        )

def main():
    """Main application function"""
    
    # Check password first
    if not check_password():
        st.stop()
    
    st.markdown("<h1 class='main-header'>🌡️ DX Shell & Tube Heat Exchanger Designer</h1>", unsafe_allow_html=True)
    st.markdown("### Direct Expansion (DX) Evaporator & Condenser | ε-NTU Method | Mass Flow Input")
    
    # Important note
    st.info("""
    **🔧 This tool designs DX (Direct Expansion) type shell & tube heat exchangers using ε-NTU method:**
    
    **Key Features:**
    - **Mass Flow Input Only** - Enter refrigerant mass flow from compressor specs
    - **ε-NTU Method** - More accurate than LMTD for phase change applications
    - **Multi-Region Analysis** - Separate calculation for each heat transfer region
    - **Performance Comparison** - Required vs Achieved heat duty and temperatures
    - **Enhanced Correlations** - Dobson & Chato for condensation, Shah for evaporation
    
    **Methods Used:**
    - **Condenser:** Three regions (Desuperheat + Condense + Subcool) with ε-NTU
    - **Evaporator:** Two regions (Evaporation + Superheat) with ε-NTU
    - **Counterflow arrangement** assumed for maximum efficiency
    """)
    
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'inputs' not in st.session_state:
        st.session_state.inputs = None
    
    # Create layout
    col1, col2 = st.columns([3, 1])
    
    with col2:
        inputs = create_input_section()
        
        # Calculate button
        button_label = "🚀 Calculate DX Evaporator Design" if inputs["hex_type"] == "DX Evaporator" else "🚀 Calculate Condenser Design"
        
        if st.sidebar.button(button_label, type="primary", use_container_width=True):
            with st.spinner("Performing ε-NTU calculations..."):
                designer = DXHeatExchangerDesign()
                
                # Convert hex_type for internal use
                calc_inputs = inputs.copy()
                calc_inputs["hex_type"] = calc_inputs["hex_type"].lower().replace("dx ", "")
                
                if calc_inputs["hex_type"] == "evaporator":
                    results = designer.design_dx_evaporator(calc_inputs)
                else:
                    results = designer.design_condenser(calc_inputs)
                
                st.session_state.results = results
                st.session_state.inputs = inputs
                st.rerun()
        
        # Reset button
        if st.sidebar.button("🔄 Reset Design", use_container_width=True):
            st.session_state.results = None
            st.session_state.inputs = None
            st.rerun()
        
        # Quick tips
        st.sidebar.markdown("---")
        with st.sidebar.expander("💡 ε-NTU Method Benefits"):
            st.markdown("""
            **Why ε-NTU is better:**
            1. **Handles phase change** - Refrigerant capacity → ∞ during condensation/evaporation
            2. **Variable U** - Different HTC in each region
            3. **Direct calculation** - No iteration needed for LMTD
            4. **Better accuracy** - Especially for small temperature differences
            
            **For Condensers:**
            - Region 1: Desuperheating (vapor cooling)
            - Region 2: Condensing (phase change, C→∞)
            - Region 3: Subcooling (liquid cooling)
            
            **For Evaporators:**
            - Region 1: Evaporating (phase change, C→∞)
            - Region 2: Superheating (vapor heating)
            """)
    
    with col1:
        if st.session_state.results is not None:
            display_results(st.session_state.results, st.session_state.inputs)
        else:
            st.markdown("""
            ## 🔧 DX Heat Exchanger Design Tool
            
            **Advanced design tool using ε-NTU method for accurate phase-change heat exchanger design:**
            
            ### **Design Methodology:**
            
            **ε-NTU (Effectiveness - Number of Transfer Units) Method:**
            - **ε = Q_actual / Q_max** - Effectiveness (0 to 1)
            - **NTU = U·A / C_min** - Number of Transfer Units
            - **Counterflow arrangement** for maximum efficiency
            
            ### **Enhanced Features:**
            
            ✅ **Multi-Region Analysis:**
               - Condenser: 3 regions (Desuperheat + Condense + Subcool)
               - Evaporator: 2 regions (Evaporation + Superheat)
               - Separate HTC calculation for each region
               - Region-by-region performance analysis
            
            ✅ **Advanced Correlations:**
               - **Dobson & Chato** for condensation HTC
               - **Shah correlation** for evaporation HTC
               - **Gnielinski** for single-phase HTC
               - **Bell-Delaware** for shell-side HTC
            
            ✅ **Performance Comparison:**
               - Required vs Achieved heat duty
               - Required vs Achieved outlet temperatures
               - Detailed region-by-region breakdown
               - Comprehensive design adequacy assessment
            
            ### **How to Use:**
            
            1. **Select heat exchanger type** (DX Evaporator or Condenser)
            2. **Enter refrigerant mass flow** (kg/s from compressor)
            3. **{"Enter superheated inlet, condensing temp, and required subcool" if inputs["hex_type"] == "Condenser" else "Enter evaporating temp, inlet quality, and required superheat"}**
            4. **Configure water/glycol side parameters**
            5. **Set geometry parameters** (tubes, pitch, layout)
            6. **Click Calculate** and review detailed results
            7. **Optimize design** based on recommendations
            
            ### **Password Protected**
            Enter password: **Semaanju**
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🔧 <strong>DX Shell & Tube Heat Exchanger Designer</strong> | ε-NTU Method | Multi-Region Analysis</p>
        <p>🧪 Advanced Heat Transfer Correlations | 📊 Performance Comparison | 🎯 Region-by-Region Analysis</p>
        <p>⚠️ For flooded evaporators (water in tubes), use separate design tool</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
