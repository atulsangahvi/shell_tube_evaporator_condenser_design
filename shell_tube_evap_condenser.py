import streamlit as st
import numpy as np
import pandas as pd
import math
from scipy.optimize import fsolve, minimize
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
    
    # Enhanced refrigerant properties database
    REFRIGERANTS = {
        "R134a": {
            "cp_vapor": 0.852, "cp_liquid": 1.434, "h_fg_ref": 198.7, "T_ref": 5.0,
            "rho_vapor": 14.43, "rho_liquid": 1277.8, "mu_vapor": 1.11e-5, "mu_liquid": 2.04e-4,
            "k_vapor": 0.0116, "k_liquid": 0.0845, "pr_vapor": 0.815, "pr_liquid": 3.425,
            "sigma": 0.00852, "T_critical": 101.1, "P_critical": 4059.0
        },
        "R404A": {
            "cp_vapor": 0.823, "cp_liquid": 1.553, "h_fg_ref": 163.3, "T_ref": 5.0,
            "rho_vapor": 33.16, "rho_liquid": 1131.8, "mu_vapor": 1.23e-5, "mu_liquid": 1.98e-4,
            "k_vapor": 0.0108, "k_liquid": 0.0718, "pr_vapor": 0.938, "pr_liquid": 4.257,
            "sigma": 0.00682, "T_critical": 72.1, "P_critical": 3734.0
        },
        "R407C": {
            "cp_vapor": 1.246, "cp_liquid": 1.448, "h_fg_ref": 200.0, "T_ref": 5.0,
            "rho_vapor": 30.02, "rho_liquid": 1149.7, "mu_vapor": 1.25e-5, "mu_liquid": 1.90e-4,
            "k_vapor": 0.0125, "k_liquid": 0.0768, "pr_vapor": 0.789, "pr_liquid": 2.901,
            "sigma": 0.00751, "T_critical": 86.1, "P_critical": 4631.0
        },
        "R410A": {
            "cp_vapor": 1.301, "cp_liquid": 1.553, "h_fg_ref": 189.6, "T_ref": 5.0,
            "rho_vapor": 35.04, "rho_liquid": 1119.6, "mu_vapor": 1.10e-5, "mu_liquid": 1.70e-4,
            "k_vapor": 0.0130, "k_liquid": 0.0759, "pr_vapor": 0.809, "pr_liquid": 2.702,
            "sigma": 0.00653, "T_critical": 71.4, "P_critical": 4901.0
        },
        "R22": {
            "cp_vapor": 0.665, "cp_liquid": 1.256, "h_fg_ref": 183.4, "T_ref": 5.0,
            "rho_vapor": 25.52, "rho_liquid": 1208.3, "mu_vapor": 1.15e-5, "mu_liquid": 1.95e-4,
            "k_vapor": 0.0110, "k_liquid": 0.0862, "pr_vapor": 0.782, "pr_liquid": 3.101,
            "sigma": 0.00821, "T_critical": 96.2, "P_critical": 4990.0
        },
        "R32": {
            "cp_vapor": 0.816, "cp_liquid": 1.423, "h_fg_ref": 236.5, "T_ref": 5.0,
            "rho_vapor": 38.21, "rho_liquid": 949.8, "mu_vapor": 1.20e-5, "mu_liquid": 1.65e-4,
            "k_vapor": 0.0139, "k_liquid": 0.1081, "pr_vapor": 0.719, "pr_liquid": 2.297,
            "sigma": 0.00582, "T_critical": 78.1, "P_critical": 5782.0
        },
        "R1234yf": {
            "cp_vapor": 0.884, "cp_liquid": 1.352, "h_fg_ref": 148.2, "T_ref": 5.0,
            "rho_vapor": 37.82, "rho_liquid": 1084.7, "mu_vapor": 1.18e-5, "mu_liquid": 2.05e-4,
            "k_vapor": 0.0120, "k_liquid": 0.0709, "pr_vapor": 0.849, "pr_liquid": 4.102,
            "sigma": 0.00621, "T_critical": 94.7, "P_critical": 3382.0
        },
        "Ammonia (R717)": {
            "cp_vapor": 2.182, "cp_liquid": 4.685, "h_fg_ref": 1261.0, "T_ref": 5.0,
            "rho_vapor": 4.256, "rho_liquid": 625.2, "mu_vapor": 9.9e-6, "mu_liquid": 1.35e-4,
            "k_vapor": 0.0246, "k_liquid": 0.5015, "pr_vapor": 0.878, "pr_liquid": 1.261,
            "sigma": 0.02342, "T_critical": 132.3, "P_critical": 11333.0
        }
    }
    
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
    
    TUBE_MATERIALS = {
        "Copper": {"k": 386, "density": 8960, "cost_factor": 1.0},
        "Cu-Ni 90/10": {"k": 40, "density": 8940, "cost_factor": 1.8},
        "Steel": {"k": 50, "density": 7850, "cost_factor": 0.6},
        "Aluminum Brass": {"k": 100, "density": 8300, "cost_factor": 1.2},
        "Stainless Steel 304": {"k": 16, "density": 8000, "cost_factor": 2.5},
        "Stainless Steel 316": {"k": 16, "density": 8000, "cost_factor": 3.0},
        "Titanium": {"k": 22, "density": 4500, "cost_factor": 8.0}
    }
    
    TUBE_SIZES = {
        "1/4\"": 0.00635, "3/8\"": 0.009525, "1/2\"": 0.0127, "5/8\"": 0.015875,
        "3/4\"": 0.01905, "1\"": 0.0254, "1.25\"": 0.03175, "1.5\"": 0.0381
    }
    
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
        self.warnings = []
    
    def get_refrigerant_properties_at_temp(self, refrigerant: str, T_sat: float) -> Dict:
        """Calculate temperature-dependent refrigerant properties"""
        base_props = self.REFRIGERANTS[refrigerant].copy()
        
        T_crit = base_props["T_critical"]
        T_sat_K = T_sat + 273.15
        T_crit_K = T_crit + 273.15
        T_ref_K = base_props["T_ref"] + 273.15
        
        # Latent heat correction (Watson correlation)
        if T_sat < T_crit:
            h_fg_corrected = base_props["h_fg_ref"] * ((T_crit - T_sat) / (T_crit - base_props["T_ref"]))**0.38
        else:
            h_fg_corrected = 0.0
        
        # Property corrections
        rho_v_corrected = base_props["rho_vapor"] * T_ref_K / T_sat_K
        rho_l_corrected = base_props["rho_liquid"] * (1 + 0.0008 * (base_props["T_ref"] - T_sat))
        mu_v_corrected = base_props["mu_vapor"] * (T_sat_K / T_ref_K)**0.7
        mu_l_corrected = base_props["mu_liquid"] * math.exp(500 * (1/T_sat_K - 1/T_ref_K))
        k_v_corrected = base_props["k_vapor"] * (T_sat_K / T_ref_K)**0.8
        k_l_corrected = base_props["k_liquid"] * (1 - 0.001 * (T_sat - base_props["T_ref"]))
        
        base_props["h_fg"] = h_fg_corrected
        base_props["rho_vapor"] = max(rho_v_corrected, 0.1)
        base_props["rho_liquid"] = max(rho_l_corrected, 100)
        base_props["mu_vapor"] = max(mu_v_corrected, 1e-6)
        base_props["mu_liquid"] = max(mu_l_corrected, 1e-5)
        base_props["k_vapor"] = max(k_v_corrected, 0.005)
        base_props["k_liquid"] = max(k_l_corrected, 0.01)
        base_props["pr_vapor"] = mu_v_corrected * base_props["cp_vapor"] * 1000 / k_v_corrected
        base_props["pr_liquid"] = mu_l_corrected * base_props["cp_liquid"] * 1000 / k_l_corrected
        
        return base_props
    
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
            "cp": cp_corrected, "rho": rho_corrected, "mu": mu_corrected,
            "k": k_corrected, "pr": pr_corrected, "freeze_point": base_props["freeze_point"],
            "glycol_type": glycol_type, "glycol_percentage": glycol_percentage
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
    
    def gnielinski_single_phase(self, Re: float, Pr: float, f: float = None) -> float:
        """Gnielinski correlation for single-phase turbulent flow"""
        if Re < 2300:
            return 4.36
        elif Re < 3000:
            Nu_lam = 4.36
            Nu_3000 = 0.023 * 3000**0.8 * Pr**0.4
            return Nu_lam + (Re - 2300) / 700 * (Nu_3000 - Nu_lam)
        else:
            if f is None:
                f = (0.79 * math.log(Re) - 1.64)**-2
            Nu = (f/8) * (Re - 1000) * Pr / (1 + 12.7 * (f/8)**0.5 * (Pr**(2/3) - 1))
            return max(Nu, 4.36)
    
    def shah_evaporation_improved(self, Re_l: float, Pr_l: float, x: float, 
                                 rho_l: float, rho_v: float, D: float, G: float, 
                                 h_fg: float, k_l: float, cp_l: float, mu_l: float) -> float:
        """Improved Shah correlation for flow boiling"""
        if x <= 0:
            return self.gnielinski_single_phase(Re_l, Pr_l) * k_l / D
        
        if x >= 1.0:
            Re_v = G * D / mu_l
            return self.gnielinski_single_phase(Re_v, Pr_l) * k_l / D * 0.6
        
        g = 9.81
        Fr_l = G**2 / (rho_l**2 * g * D)
        Co = ((1 - x) / x)**0.8 * (rho_v / rho_l)**0.5
        q_flux_estimate = 10000
        Bo = q_flux_estimate / (G * h_fg)
        
        if Re_l < 2300:
            Nu_l = 4.36
        else:
            f_l = (0.79 * math.log(Re_l) - 1.64)**-2 if Re_l > 0 else 0.02
            Nu_l = (f_l/8) * (Re_l - 1000) * Pr_l / (1 + 12.7 * (f_l/8)**0.5 * (Pr_l**(2/3) - 1))
            Nu_l = max(Nu_l, 4.36)
        
        h_l = Nu_l * k_l / D
        
        if Fr_l >= 0.04:
            if Bo > 0.0011:
                if Co <= 1.0:
                    psi = 230 * Bo**0.5
                else:
                    psi = 1.8 / Co**0.8
                if psi < max(1.0, Bo**-0.5):
                    psi = max(1.0, Bo**-0.5)
            else:
                F_nb = 14.7 * Bo**0.5
                psi = max(1.8 / Co**0.8, F_nb)
        else:
            psi_vertical = 1.8 / Co**0.8 if Co > 0.65 else 230 * Bo**0.5
            psi = max(psi_vertical, 14.7 * Bo**0.5)
        
        enhancement = 1 + x * (rho_l / rho_v - 1)
        h_tp = h_l * psi * min(enhancement, 3.0)
        h_tp = max(h_tp, 500)
        h_tp = min(h_tp, 15000)
        
        return h_tp
    
    def dobson_chato_improved(self, G: float, D: float, T_sat: float, 
                            rho_l: float, rho_v: float, mu_l: float, mu_v: float,
                            k_l: float, cp_l: float, h_fg: float, 
                            x: float, P_sat: float = None) -> float:
        """Improved Dobson & Chato correlation for condensation"""
        g = 9.81
        delta_T = 5.0
        h_fg_prime = h_fg + 0.68 * cp_l * delta_T
        
        Re_eq = G * D / mu_l
        Re_v = G * x * D / mu_v if mu_v > 0 else 0
        
        if x > 0 and x < 1:
            X_tt = ((1 - x) / x)**0.9 * (rho_v / rho_l)**0.5 * (mu_l / mu_v)**0.1
        else:
            X_tt = 0.1
        
        Fr_so = G**2 / (rho_l**2 * g * D)
        
        if x > 0.5 and Fr_so > 1.0:
            regime = "annular"
        elif Fr_so > 7.0:
            regime = "annular"
        else:
            regime = "stratified-wavy"
        
        if regime == "annular":
            Pr_l = mu_l * cp_l / k_l
            
            if Re_eq > 2300:
                f = (0.79 * math.log(Re_eq) - 1.64)**-2
                Nu_l = (f/8) * Re_eq * Pr_l / (1 + 12.7 * (f/8)**0.5 * (Pr_l**(2/3) - 1))
            else:
                Nu_l = 4.36
            
            h_l = Nu_l * k_l / D
            
            if X_tt > 0:
                phi_l = 1 + 2.22 / X_tt**0.89
            else:
                phi_l = 1.0
            
            h_annular = h_l * phi_l
            
            if x > 0.7:
                enhancement = 1 + 0.8 * (x - 0.7) * min(Re_v / 10000, 2.0)
                h_annular *= enhancement
            
            h_tp = h_annular
            
        else:
            Nu_film = 0.555 * ((g * rho_l * (rho_l - rho_v) * k_l**3 * h_fg_prime) / 
                              (mu_l * D * delta_T))**0.25
            
            if Fr_so > 0.1:
                Pr_l = mu_l * cp_l / k_l
                Re_l = G * (1 - x) * D / mu_l if mu_l > 0 else 0
                
                if Re_l > 2300:
                    Nu_conv = 0.023 * Re_l**0.8 * Pr_l**0.4
                else:
                    Nu_conv = 4.36
                
                h_conv = Nu_conv * k_l / D
                h_tp = Nu_film * k_l / D + 0.3 * h_conv * (1 - x)
            else:
                h_tp = Nu_film * k_l / D
        
        h_tp = max(h_tp, 800)
        h_tp = min(h_tp, 8000)
        
        return h_tp
    
    def calculate_single_phase_htc(self, m_dot: float, D: float, rho: float, 
                                 mu: float, k: float, cp: float, n_passes: int = 1) -> float:
        """Calculate single-phase HTC"""
        A_flow = math.pi * D**2 / 4
        
        if n_passes > 0:
            v = m_dot / (rho * A_flow * n_passes)
        else:
            v = m_dot / (rho * A_flow)
        
        Re = rho * v * D / mu if mu > 0 else 0
        Pr = mu * cp / k if k > 0 else 0
        
        if Re > 0 and Pr > 0:
            Nu = self.gnielinski_single_phase(Re, Pr)
            h = Nu * k / D
        else:
            h = 100
        
        return h
    
    def epsilon_ntu_counterflow(self, NTU: float, C_r: float) -> float:
        """ε-NTU relationship for counterflow"""
        if C_r < 1e-6:
            epsilon = 1 - math.exp(-NTU)
        elif abs(1 - C_r) < 1e-6:
            epsilon = NTU / (1 + NTU)
        else:
            numerator = 1 - math.exp(-NTU * (1 - C_r))
            denominator = 1 - C_r * math.exp(-NTU * (1 - C_r))
            if denominator == 0:
                epsilon = 1.0
            else:
                epsilon = numerator / denominator
        
        epsilon = max(0.0, min(epsilon, 1.0))
        return epsilon
    
    def calculate_shell_side_flow_area(self, shell_diameter: float, bundle_diameter: float,
                                      tube_od: float, n_tubes: int, baffle_spacing: float,
                                      baffle_cut: float = 0.25) -> float:
        """Calculate shell-side flow area"""
        A_cross = (shell_diameter - bundle_diameter) * baffle_spacing * 0.8
        
        theta_baffle = 2 * math.acos(1 - 2 * baffle_cut)
        A_window_total = (shell_diameter**2 / 8) * (theta_baffle - math.sin(theta_baffle))
        
        n_tubes_in_window = n_tubes * baffle_cut
        A_tubes_in_window = n_tubes_in_window * math.pi * tube_od**2 / 4
        A_window = A_window_total - A_tubes_in_window
        
        A_flow = min(A_cross, A_window)
        A_flow = max(A_flow, 0.001)
        
        return A_flow
    
    def calculate_condenser_three_region(self, m_dot_ref: float, m_dot_water: float,
                                       T_ref_in: float, T_cond: float, T_subcool_req: float,
                                       T_water_in: float, ref_props: Dict, water_props: Dict,
                                       tube_id: float, shell_h: float, tube_k: float,
                                       n_tubes: int, tube_length: float, n_passes: int,
                                       R_fouling: float = 0.00035) -> Dict:
        """Calculate condenser performance using ε-NTU method"""
        
        Q_desuperheat_req = m_dot_ref * ref_props["cp_vapor"] * 1000 * (T_ref_in - T_cond)
        Q_latent_req = m_dot_ref * ref_props["h_fg"] * 1000
        Q_subcool_req = m_dot_ref * ref_props["cp_liquid"] * 1000 * T_subcool_req
        Q_total_req = Q_desuperheat_req + Q_latent_req + Q_subcool_req
        
        T_subcooled_req = T_cond - T_subcool_req
        
        if T_water_in >= T_cond - 2:
            self.warnings.append(f"WARNING: Water inlet ({T_water_in:.1f}°C) too close to condensing temp ({T_cond:.1f}°C)")
        
        C_ref_desuperheat = m_dot_ref * ref_props["cp_vapor"] * 1000
        C_ref_subcool = m_dot_ref * ref_props["cp_liquid"] * 1000
        C_water = m_dot_water * water_props["cp"]
        
        tube_od = tube_id * 1.2
        A_total = math.pi * tube_od * tube_length * n_tubes
        
        A_flow_tube = (math.pi * tube_id**2 / 4) * n_tubes / max(n_passes, 1)
        G_ref = m_dot_ref / A_flow_tube if A_flow_tube > 0 else 0
        
        h_desuperheat = self.calculate_single_phase_htc(
            m_dot_ref, tube_id, ref_props["rho_vapor"], 
            ref_props["mu_vapor"], ref_props["k_vapor"],
            ref_props["cp_vapor"] * 1000, n_passes
        )
        
        h_condense = self.dobson_chato_improved(
            G_ref, tube_id, T_cond,
            ref_props["rho_liquid"], ref_props["rho_vapor"],
            ref_props["mu_liquid"], ref_props["mu_vapor"],
            ref_props["k_liquid"], ref_props["cp_liquid"] * 1000,
            ref_props["h_fg"] * 1000, x=0.5
        )
        
        h_subcool = self.calculate_single_phase_htc(
            m_dot_ref, tube_id, ref_props["rho_liquid"], 
            ref_props["mu_liquid"], ref_props["k_liquid"],
            ref_props["cp_liquid"] * 1000, n_passes
        )
        
        h_shell = shell_h
        
        R_wall = tube_od * math.log(tube_od / tube_id) / (2 * tube_k) if tube_k > 0 else 0
        
        U_desuperheat = 1 / (1/h_desuperheat + 1/h_shell + R_wall + R_fouling)
        U_condense = 1 / (1/h_condense + 1/h_shell + R_wall + R_fouling)
        U_subcool = 1 / (1/h_subcool + 1/h_shell + R_wall + R_fouling)
        
        def objective_area_distribution(area_fracs):
            f1, f2 = area_fracs
            f3 = 1.0 - f1 - f2
            
            if f1 < 0.05 or f2 < 0.5 or f3 < 0.05:
                return 1e10
            
            A_desuperheat = A_total * f1
            A_condense = A_total * f2
            A_subcool = A_total * f3
            
            C_min1 = min(C_ref_desuperheat, C_water)
            NTU1 = U_desuperheat * A_desuperheat / C_min1 if C_min1 > 0 else 0
            
            NTU2 = U_condense * A_condense / C_water if C_water > 0 else 0
            
            C_min3 = min(C_ref_subcool, C_water)
            NTU3 = U_subcool * A_subcool / C_min3 if C_min3 > 0 else 0
            
            Q1_est = Q_desuperheat_req * min(1.0, NTU1 / 2.0)
            Q2_est = Q_latent_req * min(1.0, NTU2 / 3.0)
            Q3_est = Q_subcool_req * min(1.0, NTU3 / 2.0)
            
            error = ((Q1_est - Q_desuperheat_req)/Q_total_req)**2 + \
                   ((Q2_est - Q_latent_req)/Q_total_req)**2 + \
                   ((Q3_est - Q_subcool_req)/Q_total_req)**2
            
            return error
        
        initial_guess = [0.15, 0.7]
        bounds = [(0.05, 0.3), (0.5, 0.85)]
        
        try:
            result = minimize(objective_area_distribution, initial_guess, 
                            bounds=bounds, method='L-BFGS-B')
            f1_opt, f2_opt = result.x
            f3_opt = 1.0 - f1_opt - f2_opt
        except:
            f1_opt, f2_opt, f3_opt = 0.15, 0.7, 0.15
        
        A_desuperheat = A_total * f1_opt
        A_condense = A_total * f2_opt
        A_subcool = A_total * f3_opt
        
        T_water_1 = T_water_in
        
        C_min1 = min(C_ref_desuperheat, C_water)
        C_max1 = max(C_ref_desuperheat, C_water)
        C_r1 = C_min1 / C_max1 if C_max1 > 0 else 0
        
        NTU1 = U_desuperheat * A_desuperheat / C_min1 if C_min1 > 0 else 0
        epsilon1 = self.epsilon_ntu_counterflow(NTU1, C_r1)
        
        Q_max1 = C_min1 * (T_ref_in - T_water_1)
        Q1_achieved = epsilon1 * Q_max1
        
        T_water_2 = T_water_1 + Q1_achieved / C_water if C_water > 0 else T_water_1
        T_water_2 = min(T_water_2, T_cond - 1.0)
        
        NTU2 = U_condense * A_condense / C_water if C_water > 0 else 0
        epsilon2 = 1 - math.exp(-NTU2)
        
        Q_max2 = C_water * (T_cond - T_water_2)
        Q2_achieved = epsilon2 * Q_max2
        
        T_water_3 = T_water_2 + Q2_achieved / C_water if C_water > 0 else T_water_2
        T_water_3 = min(T_water_3, T_cond - 0.5)
        
        C_min3 = min(C_ref_subcool, C_water)
        C_max3 = max(C_ref_subcool, C_water)
        C_r3 = C_min3 / C_max3 if C_max3 > 0 else 0
        
        NTU3 = U_subcool * A_subcool / C_min3 if C_min3 > 0 else 0
        epsilon3 = self.epsilon_ntu_counterflow(NTU3, C_r3)
        
        Q_max3 = C_min3 * (T_cond - T_water_3)
        Q3_achieved = epsilon3 * Q_max3
        
        T_water_out = T_water_3 + Q3_achieved / C_water if C_water > 0 else T_water_3
        T_ref_out = T_cond - Q3_achieved / C_ref_subcool if C_ref_subcool > 0 else T_subcooled_req
        T_ref_out = max(T_ref_out, T_subcooled_req - 5)
        
        Q_total_achieved = Q1_achieved + Q2_achieved + Q3_achieved
        
        C_min_overall = C_water
        Q_max_total = C_min_overall * (T_ref_in - T_water_in)
        epsilon_overall = Q_total_achieved / Q_max_total if Q_max_total > 0 else 0
        
        U_avg = (U_desuperheat * A_desuperheat + U_condense * A_condense + U_subcool * A_subcool) / A_total
        NTU_overall = U_avg * A_total / C_water if C_water > 0 else 0
        
        dt1 = T_ref_in - T_water_out
        dt2 = T_ref_out - T_water_in
        if dt1 > 0 and dt2 > 0 and abs(dt1 - dt2) > 1e-6:
            LMTD = (dt1 - dt2) / math.log(dt1 / dt2)
        else:
            LMTD = (dt1 + dt2) / 2
        
        return {
            "Q_total_req": Q_total_req / 1000, "Q_total_achieved": Q_total_achieved / 1000,
            "Q_desuperheat_req": Q_desuperheat_req / 1000, "Q_latent_req": Q_latent_req / 1000,
            "Q_subcool_req": Q_subcool_req / 1000, "Q_desuperheat_achieved": Q1_achieved / 1000,
            "Q_latent_achieved": Q2_achieved / 1000, "Q_subcool_achieved": Q3_achieved / 1000,
            "T_water_in": T_water_in, "T_water_out": T_water_out, "T_ref_in": T_ref_in,
            "T_ref_out": T_ref_out, "T_ref_out_req": T_subcooled_req, "T_cond": T_cond,
            "h_desuperheat": h_desuperheat, "h_condense": h_condense, "h_subcool": h_subcool,
            "h_shell": h_shell, "U_desuperheat": U_desuperheat, "U_condense": U_condense,
            "U_subcool": U_subcool, "U_avg": U_avg, "A_total": A_total,
            "A_desuperheat": A_desuperheat, "A_condense": A_condense, "A_subcool": A_subcool,
            "epsilon_overall": epsilon_overall, "NTU_overall": NTU_overall, "LMTD": LMTD,
            "NTU1": NTU1, "NTU2": NTU2, "NTU3": NTU3,
            "epsilon1": epsilon1, "epsilon2": epsilon2, "epsilon3": epsilon3,
            "C_water": C_water, "C_ref_desuperheat": C_ref_desuperheat,
            "C_ref_subcool": C_ref_subcool, "G_ref": G_ref, "tube_od": tube_od
        }
    
    def calculate_evaporator_two_region(self, m_dot_ref: float, m_dot_water: float,
                                      T_evap: float, inlet_quality: float, superheat_req: float,
                                      T_water_in: float, ref_props: Dict, water_props: Dict,
                                      tube_id: float, shell_h: float, tube_k: float,
                                      n_tubes: int, tube_length: float, n_passes: int,
                                      R_fouling: float = 0.00035) -> Dict:
        """Calculate evaporator performance using ε-NTU method"""
        
        x_in = inlet_quality / 100.0
        
        Q_latent_req = m_dot_ref * (1 - x_in) * ref_props["h_fg"] * 1000
        T_superheated_req = T_evap + superheat_req
        Q_superheat_req = m_dot_ref * ref_props["cp_vapor"] * 1000 * superheat_req
        Q_total_req = Q_latent_req + Q_superheat_req
        
        if T_water_in <= T_evap + 2:
            self.warnings.append(f"WARNING: Water inlet ({T_water_in:.1f}°C) too close to evaporating temp ({T_evap:.1f}°C)")
        
        C_ref_superheat = m_dot_ref * ref_props["cp_vapor"] * 1000
        C_water = m_dot_water * water_props["cp"]
        
        tube_od = tube_id * 1.2
        A_total = math.pi * tube_od * tube_length * n_tubes
        
        A_flow_tube = (math.pi * tube_id**2 / 4) * n_tubes / max(n_passes, 1)
        G_ref = m_dot_ref / A_flow_tube if A_flow_tube > 0 else 0
        
        x_avg = (x_in + 1.0) / 2.0
        Re_l = G_ref * tube_id / ref_props["mu_liquid"] if ref_props["mu_liquid"] > 0 else 0
        Pr_l = ref_props["pr_liquid"]
        
        h_evap = self.shah_evaporation_improved(
            Re_l, Pr_l, x_avg,
            ref_props["rho_liquid"], ref_props["rho_vapor"],
            tube_id, G_ref, ref_props["h_fg"] * 1000,
            ref_props["k_liquid"], ref_props["cp_liquid"] * 1000,
            ref_props["mu_liquid"]
        )
        
        h_superheat = self.calculate_single_phase_htc(
            m_dot_ref, tube_id, ref_props["rho_vapor"], 
            ref_props["mu_vapor"], ref_props["k_vapor"],
            ref_props["cp_vapor"] * 1000, n_passes
        )
        
        h_shell = shell_h
        
        R_wall = tube_od * math.log(tube_od / tube_id) / (2 * tube_k) if tube_k > 0 else 0
        
        U_evap = 1 / (1/h_evap + 1/h_shell + R_wall + R_fouling)
        U_superheat = 1 / (1/h_superheat + 1/h_shell + R_wall + R_fouling)
        
        R_evap = 1/U_evap if U_evap > 0 else 0
        R_superheat = 1/U_superheat if U_superheat > 0 else 0
        
        total_resistance_weight = Q_latent_req * R_evap + Q_superheat_req * R_superheat
        
        if total_resistance_weight > 0:
            A_evap = A_total * (Q_latent_req * R_evap) / total_resistance_weight
            A_superheat = A_total * (Q_superheat_req * R_superheat) / total_resistance_weight
        else:
            A_evap = A_total * 0.8
            A_superheat = A_total * 0.2
        
        A_sum = A_evap + A_superheat
        if abs(A_sum - A_total) > 0.001:
            A_evap = A_evap * A_total / A_sum
            A_superheat = A_superheat * A_total / A_sum
        
        NTU_evap = U_evap * A_evap / C_water if C_water > 0 else 0
        epsilon_evap = 1 - math.exp(-NTU_evap)
        
        Q_max_evap = C_water * (T_water_in - T_evap)
        Q_evap_achieved = epsilon_evap * Q_max_evap
        
        T_water_after_evap = T_water_in - Q_evap_achieved / C_water if C_water > 0 else T_water_in
        T_water_after_evap = max(T_water_after_evap, T_evap + 1.0)
        
        C_min_superheat = min(C_ref_superheat, C_water)
        C_max_superheat = max(C_ref_superheat, C_water)
        C_r_superheat = C_min_superheat / C_max_superheat if C_max_superheat > 0 else 0
        
        NTU_superheat = U_superheat * A_superheat / C_min_superheat if C_min_superheat > 0 else 0
        epsilon_superheat = self.epsilon_ntu_counterflow(NTU_superheat, C_r_superheat)
        
        Q_max_superheat = C_min_superheat * (T_water_after_evap - T_evap)
        Q_superheat_achieved = epsilon_superheat * Q_max_superheat
        
        T_water_out = T_water_after_evap - Q_superheat_achieved / C_water if C_water > 0 else T_water_after_evap
        T_water_out = max(T_water_out, T_evap + 0.5)
        
        T_ref_out = T_evap + Q_superheat_achieved / C_ref_superheat if C_ref_superheat > 0 else T_superheated_req
        
        Q_total_achieved = Q_evap_achieved + Q_superheat_achieved
        
        C_min_overall = min(C_water, C_ref_superheat)
        Q_max_total = C_min_overall * (T_water_in - T_evap)
        epsilon_overall = Q_total_achieved / Q_max_total if Q_max_total > 0 else 0
        
        U_avg = (U_evap * A_evap + U_superheat * A_superheat) / A_total
        NTU_overall = U_avg * A_total / C_water if C_water > 0 else 0
        
        dt1 = T_water_in - T_ref_out
        dt2 = T_water_out - T_evap
        if dt1 > 0 and dt2 > 0 and abs(dt1 - dt2) > 1e-6:
            LMTD = (dt1 - dt2) / math.log(dt1 / dt2)
        else:
            LMTD = (dt1 + dt2) / 2
        
        return {
            "Q_total_req": Q_total_req / 1000, "Q_total_achieved": Q_total_achieved / 1000,
            "Q_latent_req": Q_latent_req / 1000, "Q_superheat_req": Q_superheat_req / 1000,
            "Q_latent_achieved": Q_evap_achieved / 1000, "Q_superheat_achieved": Q_superheat_achieved / 1000,
            "T_water_in": T_water_in, "T_water_out": T_water_out, "T_ref_in": T_evap,
            "T_ref_out": T_ref_out, "T_ref_out_req": T_superheated_req, "T_evap": T_evap,
            "h_evap": h_evap, "h_superheat": h_superheat, "h_shell": h_shell,
            "U_evap": U_evap, "U_superheat": U_superheat, "U_avg": U_avg,
            "A_total": A_total, "A_evap": A_evap, "A_superheat": A_superheat,
            "epsilon_overall": epsilon_overall, "NTU_overall": NTU_overall, "LMTD": LMTD,
            "NTU_evap": NTU_evap, "NTU_superheat": NTU_superheat,
            "epsilon_evap": epsilon_evap, "epsilon_superheat": epsilon_superheat,
            "C_water": C_water, "C_ref_superheat": C_ref_superheat,
            "G_ref": G_ref, "inlet_quality": inlet_quality, "tube_od": tube_od
        }
    
    def design_dx_evaporator(self, inputs: Dict) -> Dict:
        """Design DX evaporator"""
        self.warnings = []
        
        refrigerant = inputs["refrigerant"]
        m_dot_ref = inputs["m_dot_ref"]
        T_evap = inputs["T_ref"]
        superheat_req = inputs["delta_T_sh_sc"]
        inlet_quality = inputs.get("inlet_quality", 20)
        
        ref_props = self.get_refrigerant_properties_at_temp(refrigerant, T_evap)
        
        glycol_percent = inputs["glycol_percentage"]
        glycol_type = inputs.get("glycol_type", "ethylene")
        m_dot_sec_L = inputs["m_dot_sec"] / 3600
        T_sec_in = inputs["T_sec_in"]
        
        tube_size = inputs["tube_size"]
        tube_material = inputs["tube_material"]
        tube_thickness = inputs["tube_thickness"] / 1000
        tube_pitch = inputs["tube_pitch"] / 1000
        n_passes = inputs["n_passes"]
        n_baffles = inputs["n_baffles"]
        n_tubes = inputs["n_tubes"]
        tube_length = inputs["tube_length"]
        tube_layout = inputs["tube_layout"].lower()
        
        sec_props = self.calculate_water_glycol_properties(T_sec_in, glycol_percent, glycol_type)
        m_dot_sec_kg = m_dot_sec_L * sec_props["rho"] / 1000
        
        tube_od = self.TUBE_SIZES[tube_size]
        tube_id = max(tube_od - 2 * tube_thickness, tube_od * 0.8)
        
        shell_diameter = self.calculate_shell_diameter(tube_od, n_tubes, tube_pitch, tube_layout)
        
        if tube_layout == "triangular":
            D_e = 4 * (0.866 * tube_pitch**2 - 0.5 * math.pi * tube_od**2 / 4) / (math.pi * tube_od)
        else:
            D_e = 4 * (tube_pitch**2 - math.pi * tube_od**2 / 4) / (math.pi * tube_od)
        
        baffle_spacing = tube_length / (n_baffles + 1)
        bundle_diameter = self.calculate_bundle_diameter(tube_od, n_tubes, tube_pitch, tube_layout)
        
        shell_flow_area = self.calculate_shell_side_flow_area(
            shell_diameter, bundle_diameter, tube_od, n_tubes, baffle_spacing
        )
        
        v_shell = m_dot_sec_kg / (sec_props["rho"] * shell_flow_area) if shell_flow_area > 0 else 0
        G_sec = m_dot_sec_kg / shell_flow_area if shell_flow_area > 0 else 0
        Re_shell = G_sec * D_e / sec_props["mu"] if sec_props["mu"] > 0 else 0
        
        h_shell = self.calculate_shell_side_htc(Re_shell, sec_props["pr"], D_e, sec_props["k"], tube_layout)
        h_shell = max(h_shell, 500)
        h_shell = min(h_shell, 8000)
        
        tube_k = self.TUBE_MATERIALS[tube_material]["k"]
        
        results = self.calculate_evaporator_two_region(
            m_dot_ref, m_dot_sec_kg, T_evap, inlet_quality, superheat_req,
            T_sec_in, ref_props, sec_props, tube_id, h_shell, tube_k,
            n_tubes, tube_length, n_passes
        )
        
        A_flow_tube = (math.pi * tube_id**2 / 4) * n_tubes / max(n_passes, 1)
        G_ref = m_dot_ref / A_flow_tube if A_flow_tube > 0 else 0
        
        x_avg = (inlet_quality/100.0 + 1.0) / 2.0
        rho_tp = 1 / (x_avg/ref_props["rho_vapor"] + (1-x_avg)/ref_props["rho_liquid"])
        v_ref = G_ref / rho_tp
        
        Re_l = G_ref * tube_id / ref_props["mu_liquid"] if ref_props["mu_liquid"] > 0 else 0
        if Re_l > 2300:
            f_tube = (0.79 * math.log(Re_l) - 1.64)**-2 if Re_l > 0 else 0.02
        else:
            f_tube = 64 / Re_l if Re_l > 0 else 0.05
        
        phi_tp = 1 + 2.5 / x_avg if x_avg > 0 else 1
        dp_tube = f_tube * (tube_length * n_passes / tube_id) * (rho_tp * v_ref**2 / 2) * phi_tp
        
        if Re_shell < 2300:
            f_shell = 64 / Re_shell if Re_shell > 0 else 0.2
        else:
            f_shell = 0.2 * Re_shell**-0.2
        
        dp_shell = f_shell * (tube_length / D_e) * (n_baffles + 1) * (sec_props["rho"] * v_shell**2 / 2)
        
        sec_velocity_status = self.check_velocity_status(v_shell, glycol_percent, "shell")
        ref_velocity_status = self.check_velocity_status(v_ref, 0, "refrigerant_two_phase")
        
        m_dot_per_tube = m_dot_ref / n_tubes * 3600
        distribution_status = "Good" if m_dot_per_tube >= 3.6 else "Marginal" if m_dot_per_tube >= 2.0 else "Poor"
        
        freeze_point = self.calculate_freeze_point(glycol_percent, glycol_type)
        freeze_risk = "High" if results["T_water_out"] < freeze_point + 2 else "Medium" if results["T_water_out"] < freeze_point + 3 else "Low"
        
        Q_required = results["Q_total_req"] * 1000
        U_avg = results["U_avg"]
        LMTD = results["LMTD"] if results["LMTD"] > 0 else 5.0
        
        A_required = Q_required / (U_avg * LMTD) if U_avg > 0 and LMTD > 0 else 0
        area_ratio = results["A_total"] / A_required if A_required > 0 else 0
        
        design_status = self.determine_design_status(
            results["epsilon_overall"], results["A_total"], A_required,
            results["Q_total_achieved"], results["Q_total_req"]
        )
        
        validation_warnings = self.validate_design({
            "hex_type": "evaporator",
            "t_water_out": results["T_water_out"],
            "t_evap": T_evap,
            "velocity_shell_ms": v_shell,
            "velocity_tube_ms": v_ref,
            "dp_shell_kpa": dp_shell / 1000,
            "dp_tube_kpa": dp_tube / 1000,
            "inlet_quality_percent": inlet_quality
        })
        
        self.results = {
            "heat_exchanger_type": "DX Evaporator",
            "design_method": "Mass Flow Input (ε-NTU)",
            "heat_duty_required_kw": results["Q_total_req"],
            "heat_duty_achieved_kw": results["Q_total_achieved"],
            "kw_difference": results["Q_total_achieved"] - results["Q_total_req"],
            "kw_match_percentage": (results["Q_total_achieved"] / results["Q_total_req"] * 100) if results["Q_total_req"] > 0 else 0,
            "q_latent_req_kw": results["Q_latent_req"],
            "q_superheat_req_kw": results["Q_superheat_req"],
            "q_latent_achieved_kw": results["Q_latent_achieved"],
            "q_superheat_achieved_kw": results["Q_superheat_achieved"],
            "effectiveness": results["epsilon_overall"],
            "ntu": results["NTU_overall"],
            "ntu_evap": results["NTU_evap"],
            "ntu_superheat": results["NTU_superheat"],
            "epsilon_evap": results["epsilon_evap"],
            "epsilon_superheat": results["epsilon_superheat"],
            "overall_u": results["U_avg"],
            "h_tube_evap": results["h_evap"],
            "h_tube_superheat": results["h_superheat"],
            "h_shell": results["h_shell"],
            "u_evap": results["U_evap"],
            "u_superheat": results["U_superheat"],
            "lmtd": results["LMTD"],
            "t_sec_in": results["T_water_in"],
            "t_sec_out": results["T_water_out"],
            "t_ref_in": results["T_evap"],
            "t_ref_out_required": results["T_ref_out_req"],
            "t_ref_out_achieved": results["T_ref_out"],
            "superheat_difference": results["T_ref_out"] - results["T_ref_out_req"],
            "water_deltaT": abs(results["T_water_out"] - results["T_water_in"]),
            "superheat_req": superheat_req,
            "refrigerant": refrigerant,
            "refrigerant_mass_flow_kg_s": m_dot_ref,
            "refrigerant_mass_flow_kg_hr": m_dot_ref * 3600,
            "inlet_quality_percent": inlet_quality,
            "water_vol_flow_L_hr": m_dot_sec_L * 3600,
            "water_mass_flow_kg_hr": m_dot_sec_kg * 3600,
            "flow_per_tube_kg_hr": m_dot_per_tube,
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
            "n_tubes": n_tubes,
            "tube_length_m": tube_length,
            "n_passes": n_passes,
            "tube_layout": tube_layout,
            "tube_material": tube_material,
            "velocity_tube_ms": v_ref,
            "velocity_shell_ms": v_shell,
            "velocity_shell_status": sec_velocity_status["status"],
            "velocity_tube_status": ref_velocity_status["status"],
            "dp_tube_kpa": dp_tube / 1000,
            "dp_shell_kpa": dp_shell / 1000,
            "reynolds_tube": Re_l,
            "reynolds_shell": Re_shell,
            "mass_flux_tube": G_ref,
            "mass_flux_shell": G_sec,
            "c_water": results["C_water"],
            "c_ref_superheat": results["C_ref_superheat"],
            "distribution_status": distribution_status,
            "freeze_point_c": freeze_point,
            "freeze_risk": freeze_risk,
            "glycol_type": glycol_type,
            "glycol_percentage": glycol_percent,
            "design_status": design_status,
            "validation_warnings": validation_warnings
        }
        
        return self.results
    
    def design_condenser(self, inputs: Dict) -> Dict:
        """Design condenser"""
        self.warnings = []
        
        refrigerant = inputs["refrigerant"]
        m_dot_ref = inputs["m_dot_ref"]
        T_ref_in_superheated = inputs["T_ref_in_superheated"]
        T_cond = inputs["T_ref"]
        subcool_req = inputs["delta_T_sh_sc"]
        
        ref_props = self.get_refrigerant_properties_at_temp(refrigerant, T_cond)
        
        glycol_percent = inputs["glycol_percentage"]
        glycol_type = inputs.get("glycol_type", "ethylene")
        m_dot_sec_L = inputs["m_dot_sec"] / 3600
        T_sec_in = inputs["T_sec_in"]
        
        tube_size = inputs["tube_size"]
        tube_material = inputs["tube_material"]
        tube_thickness = inputs["tube_thickness"] / 1000
        tube_pitch = inputs["tube_pitch"] / 1000
        n_passes = inputs["n_passes"]
        n_baffles = inputs["n_baffles"]
        n_tubes = inputs["n_tubes"]
        tube_length = inputs["tube_length"]
        tube_layout = inputs["tube_layout"].lower()
        
        sec_props = self.calculate_water_glycol_properties(T_sec_in, glycol_percent, glycol_type)
        m_dot_sec_kg = m_dot_sec_L * sec_props["rho"] / 1000
        
        tube_od = self.TUBE_SIZES[tube_size]
        tube_id = max(tube_od - 2 * tube_thickness, tube_od * 0.8)
        
        shell_diameter = self.calculate_shell_diameter(tube_od, n_tubes, tube_pitch, tube_layout)
        
        if tube_layout == "triangular":
            D_e = 4 * (0.866 * tube_pitch**2 - 0.5 * math.pi * tube_od**2 / 4) / (math.pi * tube_od)
        else:
            D_e = 4 * (tube_pitch**2 - math.pi * tube_od**2 / 4) / (math.pi * tube_od)
        
        baffle_spacing = tube_length / (n_baffles + 1)
        bundle_diameter = self.calculate_bundle_diameter(tube_od, n_tubes, tube_pitch, tube_layout)
        
        shell_flow_area = self.calculate_shell_side_flow_area(
            shell_diameter, bundle_diameter, tube_od, n_tubes, baffle_spacing
        )
        
        v_shell = m_dot_sec_kg / (sec_props["rho"] * shell_flow_area) if shell_flow_area > 0 else 0
        G_sec = m_dot_sec_kg / shell_flow_area if shell_flow_area > 0 else 0
        Re_shell = G_sec * D_e / sec_props["mu"] if sec_props["mu"] > 0 else 0
        
        h_shell = self.calculate_shell_side_htc(Re_shell, sec_props["pr"], D_e, sec_props["k"], tube_layout)
        h_shell = max(h_shell, 500)
        h_shell = min(h_shell, 8000)
        
        tube_k = self.TUBE_MATERIALS[tube_material]["k"]
        
        results = self.calculate_condenser_three_region(
            m_dot_ref, m_dot_sec_kg, T_ref_in_superheated, T_cond, subcool_req,
            T_sec_in, ref_props, sec_props, tube_id, h_shell, tube_k,
            n_tubes, tube_length, n_passes
        )
        
        A_flow_tube = (math.pi * tube_id**2 / 4) * n_tubes / max(n_passes, 1)
        G_ref = results["G_ref"]
        
        rho_tp = 1 / (0.5/ref_props["rho_vapor"] + 0.5/ref_props["rho_liquid"])
        v_ref = G_ref / rho_tp
        
        Re_l = G_ref * tube_id / ref_props["mu_liquid"] if ref_props["mu_liquid"] > 0 else 0
        if Re_l > 2300:
            f_tube = (0.79 * math.log(Re_l) - 1.64)**-2 if Re_l > 0 else 0.02
        else:
            f_tube = 64 / Re_l if Re_l > 0 else 0.05
        
        phi_tp = 1 + 1.5 / 0.5 if 0.5 > 0 else 1
        dp_tube = f_tube * (tube_length * n_passes / tube_id) * (rho_tp * v_ref**2 / 2) * phi_tp
        
        if Re_shell < 2300:
            f_shell = 64 / Re_shell if Re_shell > 0 else 0.2
        else:
            f_shell = 0.2 * Re_shell**-0.2
        
        dp_shell = f_shell * (tube_length / D_e) * (n_baffles + 1) * (sec_props["rho"] * v_shell**2 / 2)
        
        sec_velocity_status = self.check_velocity_status(v_shell, glycol_percent, "shell")
        ref_velocity_status = self.check_velocity_status(v_ref, 0, "refrigerant_two_phase")
        
        Q_required = results["Q_total_req"] * 1000
        U_avg = results["U_avg"]
        LMTD = results["LMTD"] if results["LMTD"] > 0 else 5.0
        
        A_required = Q_required / (U_avg * LMTD) if U_avg > 0 and LMTD > 0 else 0
        area_ratio = results["A_total"] / A_required if A_required > 0 else 0
        
        design_status = self.determine_design_status(
            results["epsilon_overall"], results["A_total"], A_required,
            results["Q_total_achieved"], results["Q_total_req"]
        )
        
        validation_warnings = self.validate_design({
            "hex_type": "condenser",
            "t_water_in": T_sec_in,
            "t_ref_condensing": T_cond,
            "velocity_shell_ms": v_shell,
            "velocity_tube_ms": v_ref,
            "dp_shell_kpa": dp_shell / 1000,
            "dp_tube_kpa": dp_tube / 1000
        })
        
        self.results = {
            "heat_exchanger_type": "Condenser",
            "design_method": "Mass Flow Input (ε-NTU)",
            "heat_duty_required_kw": results["Q_total_req"],
            "heat_duty_achieved_kw": results["Q_total_achieved"],
            "kw_difference": results["Q_total_achieved"] - results["Q_total_req"],
            "kw_match_percentage": (results["Q_total_achieved"] / results["Q_total_req"] * 100) if results["Q_total_req"] > 0 else 0,
            "q_desuperheat_req_kw": results["Q_desuperheat_req"],
            "q_latent_req_kw": results["Q_latent_req"],
            "q_subcool_req_kw": results["Q_subcool_req"],
            "q_desuperheat_achieved_kw": results["Q_desuperheat_achieved"],
            "q_latent_achieved_kw": results["Q_latent_achieved"],
            "q_subcool_achieved_kw": results["Q_subcool_achieved"],
            "effectiveness": results["epsilon_overall"],
            "ntu": results["NTU_overall"],
            "ntu_desuperheat": results["NTU1"],
            "ntu_condense": results["NTU2"],
            "ntu_subcool": results["NTU3"],
            "epsilon_desuperheat": results["epsilon1"],
            "epsilon_condense": results["epsilon2"],
            "epsilon_subcool": results["epsilon3"],
            "overall_u": results["U_avg"],
            "h_tube_desuperheat": results["h_desuperheat"],
            "h_tube_condense": results["h_condense"],
            "h_tube_subcool": results["h_subcool"],
            "h_shell": results["h_shell"],
            "u_desuperheat": results["U_desuperheat"],
            "u_condense": results["U_condense"],
            "u_subcool": results["U_subcool"],
            "lmtd": results["LMTD"],
            "t_sec_in": results["T_water_in"],
            "t_sec_out": results["T_water_out"],
            "t_ref_in_superheated": results["T_ref_in"],
            "t_ref_condensing": results["T_cond"],
            "t_ref_out_required": results["T_ref_out_req"],
            "t_ref_out_achieved": results["T_ref_out"],
            "subcool_difference": results["T_ref_out"] - results["T_ref_out_req"],
            "water_deltaT": abs(results["T_water_out"] - results["T_water_in"]),
            "subcool_req": subcool_req,
            "refrigerant": refrigerant,
            "refrigerant_mass_flow_kg_s": m_dot_ref,
            "refrigerant_mass_flow_kg_hr": m_dot_ref * 3600,
            "water_vol_flow_L_hr": m_dot_sec_L * 3600,
            "water_mass_flow_kg_hr": m_dot_sec_kg * 3600,
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
            "n_tubes": n_tubes,
            "tube_length_m": tube_length,
            "n_passes": n_passes,
            "tube_layout": tube_layout,
            "tube_material": tube_material,
            "velocity_tube_ms": v_ref,
            "velocity_shell_ms": v_shell,
            "velocity_shell_status": sec_velocity_status["status"],
            "velocity_tube_status": ref_velocity_status["status"],
            "dp_tube_kpa": dp_tube / 1000,
            "dp_shell_kpa": dp_shell / 1000,
            "reynolds_tube": Re_l,
            "reynolds_shell": Re_shell,
            "mass_flux_tube": G_ref,
            "mass_flux_shell": G_sec,
            "c_water": results["C_water"],
            "c_ref_desuperheat": results["C_ref_desuperheat"],
            "c_ref_subcool": results["C_ref_subcool"],
            "glycol_type": glycol_type,
            "glycol_percentage": glycol_percent,
            "design_status": design_status,
            "validation_warnings": validation_warnings
        }
        
        return self.results
    
    def calculate_shell_diameter(self, tube_od: float, n_tubes: int, pitch: float,
                               tube_layout: str = "triangular") -> float:
        """Calculate shell diameter"""
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
    
    def calculate_bundle_diameter(self, tube_od: float, n_tubes: int, pitch: float,
                                tube_layout: str = "triangular") -> float:
        """Calculate bundle diameter"""
        if tube_layout.lower() == "triangular":
            tubes_per_row = math.sqrt(n_tubes / 0.866)
            bundle_diameter = tubes_per_row * pitch
        else:
            tubes_per_row = math.sqrt(n_tubes)
            bundle_diameter = tubes_per_row * pitch
        
        return bundle_diameter
    
    def calculate_shell_side_htc(self, Re: float, Pr: float, D_e: float,
                               k: float, tube_layout: str) -> float:
        """Calculate shell-side HTC"""
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
        """Check velocity status"""
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
            "velocity": velocity, "status": status, "color": color,
            "min": rec["min"], "opt": rec["opt"], "max": rec["max"]
        }
    
    def validate_design(self, results: Dict) -> List[str]:
        """Validate design"""
        warnings = []
        
        if results["hex_type"] == "condenser":
            if results.get("t_water_out", 0) >= results.get("t_ref_condensing", 100) - 2:
                warnings.append("⚠️ Pinch point violation: Water outlet too close to condensing temperature")
            
            if results.get("t_water_in", 0) >= results.get("t_ref_condensing", 100):
                warnings.append("🚨 CRITICAL: Water inlet above condensing temperature!")
        
        elif results["hex_type"] == "evaporator":
            if results.get("t_water_out", 100) <= results.get("t_evap", 0) + 1:
                warnings.append("⚠️ Pinch point violation: Water outlet too close to evaporating temperature")
        
        if results.get("velocity_shell_ms", 0) < 0.3:
            warnings.append("⚠️ Shell velocity too low - risk of poor distribution and fouling")
        
        if results.get("velocity_tube_ms", 0) > 30:
            warnings.append("⚠️ Tube velocity too high - risk of erosion and high pressure drop")
        
        if results["hex_type"] == "evaporator":
            if results.get("inlet_quality_percent", 0) > 50:
                warnings.append("⚠️ High inlet quality - ensure proper refrigerant distribution")
        
        if results.get("dp_shell_kpa", 0) > 50:
            warnings.append("⚠️ High shell-side pressure drop (>50 kPa)")
        
        if results.get("dp_tube_kpa", 0) > 100:
            warnings.append("⚠️ High tube-side pressure drop (>100 kPa)")
        
        return warnings
    
    def determine_design_status(self, effectiveness: float, area_total: float, 
                              area_required: float, kw_achieved: float, 
                              kw_required: float) -> str:
        """Determine design status"""
        area_ratio = area_total / area_required if area_required > 0 else 0
        kw_ratio = kw_achieved / kw_required if kw_required > 0 else 0
        
        kw_match = 0.85 <= kw_ratio <= 1.15
        area_adequate = 0.9 <= area_ratio <= 1.5
        effective_enough = effectiveness >= 0.3
        
        if kw_match and area_adequate and effective_enough:
            return "Adequate"
        elif (0.7 <= kw_ratio <= 1.3) and (0.7 <= area_ratio <= 2.0) and effectiveness >= 0.2:
            return "Marginal"
        else:
            return "Inadequate"

def number_input_with_buttons(label: str, min_value: float, max_value: float, 
                            value: float, step: float, key: str, format: str = "%.1f",
                            help_text: str = None) -> float:
    """Number input with +/- buttons"""
    
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
    """Create input section"""
    st.sidebar.header("⚙️ DX Heat Exchanger Design")
    
    inputs = {}
    
    inputs["hex_type"] = st.sidebar.radio(
        "Heat Exchanger Type",
        ["DX Evaporator", "Condenser"],
        help="DX Evaporator: Refrigerant evaporates in tubes\nCondenser: Refrigerant condenses in tubes"
    )
    
    if inputs["hex_type"] == "DX Evaporator":
        st.sidebar.markdown('<span class="dx-badge">DX Type</span>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<span class="condenser-badge">Condenser</span>', unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("📊 Design Input Method")
    st.sidebar.info("**Mass Flow Input**\nEnter refrigerant mass flow from compressor specs")
    
    designer = DXHeatExchangerDesign()
    
    st.sidebar.subheader("🔧 Refrigerant Parameters")
    
    inputs["refrigerant"] = st.sidebar.selectbox(
        "Refrigerant Type",
        list(designer.REFRIGERANTS.keys())
    )
    
    inputs["m_dot_ref"] = number_input_with_buttons(
        label="Refrigerant Mass Flow (kg/s)",
        min_value=0.01, max_value=10.0, value=0.221, step=0.001,
        key="m_dot_ref", format="%.3f",
        help_text="From compressor specification sheet"
    )
    
    if inputs["hex_type"] == "Condenser":
        st.sidebar.subheader("🌡️ Condenser Parameters")
        
        inputs["T_ref_in_superheated"] = number_input_with_buttons(
            label="Superheated Refrigerant Inlet (°C)",
            min_value=50.0, max_value=150.0, value=95.0, step=1.0,
            key="T_ref_superheated", format="%.1f",
            help_text="Temperature from compressor"
        )
        
        inputs["T_ref"] = number_input_with_buttons(
            label="Condensing Temperature (°C)",
            min_value=20.0, max_value=80.0, value=45.0, step=1.0,
            key="T_cond", format="%.1f"
        )
        
        inputs["delta_T_sh_sc"] = number_input_with_buttons(
            label="Required Subcool at Exit (K)",
            min_value=0.0, max_value=20.0, value=5.0, step=0.5,
            key="subcool", format="%.1f"
        )
    
    else:
        st.sidebar.subheader("🌡️ Evaporator Parameters")
        
        inputs["T_ref"] = number_input_with_buttons(
            label="Evaporating Temperature (°C)",
            min_value=-50.0, max_value=20.0, value=5.0, step=1.0,
            key="T_evap", format="%.1f"
        )
        
        inputs["inlet_quality"] = number_input_with_buttons(
            label="Inlet Quality from TXV (%)",
            min_value=0.0, max_value=100.0, value=20.0, step=1.0,
            key="inlet_quality", format="%.1f",
            help_text="Quality entering evaporator (0-100%)"
        )
        
        inputs["delta_T_sh_sc"] = number_input_with_buttons(
            label="Required Superheat at Exit (K)",
            min_value=3.0, max_value=15.0, value=5.0, step=0.5,
            key="superheat", format
