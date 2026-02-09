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
</style>
""", unsafe_allow_html=True)

class DXHeatExchangerDesign:
    """DX (Direct Expansion) Shell & Tube Heat Exchanger Design"""
    
    # Refrigerant properties database
    REFRIGERANTS = {
        "R134a": {
            "cp_vapor": 0.852,  # kJ/kg·K at 5°C
            "cp_liquid": 1.434,
            "h_fg": 198.7,  # kJ/kg
            "rho_vapor": 14.43,  # kg/m³
            "rho_liquid": 1277.8,
            "mu_vapor": 1.11e-5,  # Pa·s
            "mu_liquid": 2.04e-4,
            "k_vapor": 0.0116,  # W/m·K
            "k_liquid": 0.0845,
            "pr_vapor": 0.815,
            "pr_liquid": 3.425,
            "sigma": 0.00852  # Surface tension N/m
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
            "sigma": 0.00682
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
            "sigma": 0.00751
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
            "sigma": 0.00653
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
            "sigma": 0.00821
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
            "sigma": 0.00582
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
            "sigma": 0.00621
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
            "sigma": 0.02342
        }
    }
    
    # Enhanced water/glycol properties database
    # Ethylene Glycol (EG) properties
    ETHYLENE_GLYCOL_PROPERTIES = {
        0: {"cp": 4.186, "rho": 998.2, "mu": 0.00100, "k": 0.598, "pr": 7.01, "freeze_point": 0.0},
        10: {"cp": 4.080, "rho": 1022.0, "mu": 0.00132, "k": 0.570, "pr": 9.45, "freeze_point": -3.5},
        20: {"cp": 3.950, "rho": 1040.0, "mu": 0.00180, "k": 0.540, "pr": 13.15, "freeze_point": -7.5},
        30: {"cp": 3.780, "rho": 1057.0, "mu": 0.00258, "k": 0.510, "pr": 19.10, "freeze_point": -14.0},
        40: {"cp": 3.600, "rho": 1069.0, "mu": 0.00400, "k": 0.470, "pr": 30.60, "freeze_point": -23.0},
        50: {"cp": 3.420, "rho": 1077.0, "mu": 0.00680, "k": 0.430, "pr": 54.10, "freeze_point": -36.0},
        60: {"cp": 3.200, "rho": 1082.0, "mu": 0.01200, "k": 0.390, "pr": 98.50, "freeze_point": -52.0}
    }
    
    # Propylene Glycol (PG) properties - Food grade
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
    
    # Tube sizes (inches to meters)
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
    
    # Recommended velocities (m/s)
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
        self.glycol_type = "ethylene"  # Default
    
    def calculate_water_glycol_properties(self, temperature: float, glycol_percentage: int, 
                                        glycol_type: str = "ethylene") -> Dict:
        """Calculate water/glycol mixture properties with temperature correction"""
        if glycol_type.lower() == "propylene":
            base_props = self.PROPYLENE_GLYCOL_PROPERTIES.get(glycol_percentage, 
                                                           self.PROPYLENE_GLYCOL_PROPERTIES[0])
        else:
            base_props = self.ETHYLENE_GLYCOL_PROPERTIES.get(glycol_percentage, 
                                                           self.ETHYLENE_GLYCOL_PROPERTIES[0])
        
        # Store glycol type
        self.glycol_type = glycol_type.lower()
        
        # Temperature correction factors (more accurate)
        T_ref = 20.0  # Reference temperature
        
        # Temperature dependent corrections
        T_K = temperature + 273.15
        T_ref_K = T_ref + 273.15
        
        # Specific heat correction
        cp_factor = 1.0 - 0.0005 * (temperature - T_ref)
        
        # Density correction (water/glycol expands with temperature)
        rho_factor = 1.0 - 0.0002 * (temperature - T_ref)
        
        # Viscosity correction (Arrhenius type)
        if temperature > T_ref:
            mu_factor = math.exp(-0.025 * (temperature - T_ref))
        else:
            mu_factor = math.exp(0.035 * (T_ref - temperature))
        
        # Thermal conductivity correction
        k_factor = 1.0 + 0.0018 * (temperature - T_ref)
        
        # Calculate corrected properties
        cp_corrected = base_props["cp"] * cp_factor * 1000  # Convert to J/kg·K
        rho_corrected = base_props["rho"] * rho_factor
        mu_corrected = base_props["mu"] * mu_factor
        k_corrected = base_props["k"] * k_factor
        
        # Calculate Prandtl number
        pr_corrected = mu_corrected * cp_corrected / k_corrected
        
        return {
            "cp": cp_corrected,  # J/kg·K
            "rho": rho_corrected,  # kg/m³
            "mu": mu_corrected,  # Pa·s
            "k": k_corrected,  # W/m·K
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
    
    def gnielinski_single_phase(self, Re: float, Pr: float) -> float:
        """Gnielinski correlation for single-phase turbulent flow"""
        if Re < 2300:
            # Laminar flow - constant heat flux
            return 4.36
        elif Re < 3000:
            # Transition region - interpolate
            Nu_lam = 4.36
            # Use Dittus-Boelter at Re=3000 for interpolation
            Nu_3000 = 0.023 * 3000**0.8 * Pr**0.4
            return Nu_lam + (Re - 2300) / 700 * (Nu_3000 - Nu_lam)
        else:
            # Turbulent flow - Gnielinski correlation
            f = (0.79 * math.log(Re) - 1.64)**-2
            Nu = (f/8) * (Re - 1000) * Pr / (1 + 12.7 * (f/8)**0.5 * (Pr**(2/3) - 1))
            return Nu
    
    def shah_evaporation(self, Re_l: float, Pr_l: float, x: float, 
                        rho_l: float, rho_v: float, D: float, G: float, 
                        h_fg: float, k_l: float) -> float:
        """Shah correlation for flow boiling in tubes (DX evaporator)"""
        if x <= 0:
            # Subcooled region
            return self.gnielinski_single_phase(Re_l, Pr_l) * k_l / D
        
        # Convective boiling number
        Co = ((1 - x) / x)**0.8 * (rho_v / rho_l)**0.5 if x > 0 else 1e6
        
        # Boiling number (simplified)
        Bo = G * h_fg / (k_l * 273)  # Approximate
        
        if Co <= 0.65:
            # Nucleate boiling dominant
            N = Co
        else:
            # Convective boiling dominant
            N = 0.38 * Co**-0.3
        
        # Enhancement factor
        if Bo > 0.0011:
            F = 14.7 * Bo**0.56 * N
        else:
            F = 15.43 * Bo**0.56 * N
        
        # Single-phase liquid Nusselt
        if Re_l < 2300:
            Nu_l = 4.36
        else:
            f_l = (0.79 * math.log(Re_l) - 1.64)**-2
            Nu_l = (f_l/8) * (Re_l - 1000) * Pr_l / (1 + 12.7 * (f_l/8)**0.5 * (Pr_l**(2/3) - 1))
        
        # Two-phase Nusselt
        Nu_tp = Nu_l * (1 + 2.4e4 * Bo**1.16 + 1.37 * Co**-0.86) if x > 0 else Nu_l
        
        return Nu_tp * k_l / D
    
    def cavallini_condensation(self, Re_l: float, Re_v: float, Pr_l: float, x: float,
                             rho_l: float, rho_v: float, mu_l: float, mu_v: float,
                             D: float, k_l: float) -> float:
        """Cavallini-Zecchin correlation for condensation in tubes"""
        # Martinelli parameter
        X_tt = ((1 - x) / x)**0.9 * (rho_v / rho_l)**0.5 * (mu_l / mu_v)**0.1 if x > 0 else 1e6
        
        # All liquid Reynolds number
        Re_lo = Re_l * (1 - x) if x < 1 else Re_l
        
        # Single-phase liquid Nusselt
        Nu_lo = self.gnielinski_single_phase(Re_lo, Pr_l)
        
        # Two-phase multiplier
        if X_tt <= 10:
            phi_l = 1 + 1.8 / X_tt**0.87
        else:
            phi_l = 1.0
        
        Nu_tp = Nu_lo * phi_l
        
        return Nu_tp * k_l / D
    
    def friedel_two_phase_pressure_drop(self, Re_l: float, Re_v: float, x: float,
                                      rho_l: float, rho_v: float, mu_l: float,
                                      mu_v: float, D: float, G: float, L: float) -> float:
        """Friedel correlation for two-phase frictional pressure drop"""
        # Liquid-only friction factor
        f_lo = (0.79 * math.log(Re_l) - 1.64)**-2 if Re_l > 2300 else 64/Re_l
        
        # Two-phase multiplier
        Fr = G**2 / (9.81 * D * rho_l**2)
        We = G**2 * D / (rho_l * 0.001)  # Using 0.001 as sigma approximation
        
        A = (1 - x)**2 + x**2 * (rho_l * f_lo) / (rho_v * f_lo)
        B = x**0.78 * (1 - x)**0.224
        C = (rho_l / rho_v)**0.91 * (mu_v / mu_l)**0.19 * (1 - mu_v / mu_l)**0.7
        
        phi_lo2 = A + 3.24 * B * C / (Fr**0.045 * We**0.035)
        
        # Liquid-only pressure gradient
        dp_dz_lo = 2 * f_lo * G**2 / (D * rho_l)
        
        # Two-phase pressure gradient
        dp_dz_tp = dp_dz_lo * phi_lo2
        
        return dp_dz_tp * L
    
    def calculate_shell_diameter(self, tube_od: float, n_tubes: int, pitch: float,
                               tube_layout: str = "triangular") -> float:
        """Calculate shell diameter based on tube count and pitch"""
        if tube_layout.lower() == "triangular":
            # Triangular pitch
            tubes_per_row = math.sqrt(n_tubes / 0.866)
            bundle_width = tubes_per_row * pitch
        else:
            # Square pitch
            tubes_per_row = math.sqrt(n_tubes)
            bundle_width = tubes_per_row * pitch
        
        # Add clearances (TEMA standards)
        if bundle_width < 0.3:
            clearance = 0.010  # 10mm
        elif bundle_width < 0.6:
            clearance = 0.015  # 15mm
        else:
            clearance = 0.020  # 20mm
        
        shell_diameter = bundle_width + 2 * clearance
        
        return max(shell_diameter, 0.1)
    
    def calculate_shell_side_htc(self, Re: float, Pr: float, D_e: float,
                               k: float, tube_layout: str) -> float:
        """Calculate shell-side HTC using Bell-Delaware method (simplified)"""
        if Re < 100:
            # Laminar flow
            if tube_layout == "triangular":
                Nu = 1.0
            else:
                Nu = 0.9
        elif Re < 1000:
            # Transition region
            if tube_layout == "triangular":
                Nu = 0.6 * Re**0.5 * Pr**0.33
            else:
                Nu = 0.5 * Re**0.5 * Pr**0.33
        else:
            # Turbulent flow
            if tube_layout == "triangular":
                Nu = 0.36 * Re**0.55 * Pr**0.33
            else:
                Nu = 0.31 * Re**0.6 * Pr**0.33
        
        return Nu * k / D_e
    
    def design_from_kw(self, inputs: Dict) -> Dict:
        """Design heat exchanger from kW requirement"""
        
        # Extract inputs
        hex_type = inputs["hex_type"].lower()
        refrigerant = inputs["refrigerant"]
        Q_total_kw = inputs["heat_duty_kw"]  # kW from user input
        
        if hex_type == "evaporator":
            T_ref = inputs["T_ref"]
            delta_T_superheat = inputs["delta_T_sh_sc"]
            
            # Get properties
            ref_props = self.REFRIGERANTS[refrigerant]
            
            # Calculate mass flow from kW: Q = m_dot * (h_fg + cp_vapor * ΔT_superheat)
            total_enthalpy_change = ref_props["h_fg"] + ref_props["cp_vapor"] * delta_T_superheat
            m_dot_ref = Q_total_kw / total_enthalpy_change  # kg/s
            
            T_ref_out = T_ref + delta_T_superheat
            
            # Calculate sensible and latent portions
            Q_latent = m_dot_ref * ref_props["h_fg"]
            Q_sensible = m_dot_ref * ref_props["cp_vapor"] * delta_T_superheat
            
        else:  # condenser
            T_ref = inputs["T_ref"]
            delta_T_subcool = inputs["delta_T_sh_sc"]
            
            # Get properties
            ref_props = self.REFRIGERANTS[refrigerant]
            
            # Calculate mass flow from kW: Q = m_dot * (h_fg + cp_liquid * ΔT_subcool)
            total_enthalpy_change = ref_props["h_fg"] + ref_props["cp_liquid"] * delta_T_subcool
            m_dot_ref = Q_total_kw / total_enthalpy_change  # kg/s
            
            T_ref_out = T_ref - delta_T_subcool
            
            # Calculate sensible and latent portions
            Q_latent = m_dot_ref * ref_props["h_fg"]
            Q_sensible = m_dot_ref * ref_props["cp_liquid"] * delta_T_subcool
        
        # Return calculated parameters
        return {
            "m_dot_ref": m_dot_ref,
            "T_ref_out": T_ref_out,
            "Q_latent": Q_latent,
            "Q_sensible": Q_sensible,
            "Q_total": Q_total_kw,
            "ref_props": ref_props
        }
    
    def design_dx_evaporator(self, inputs: Dict, design_from_kw: bool = False) -> Dict:
        """Design DX evaporator - refrigerant in tubes, water/glycol in shell"""
        
        # Extract inputs
        refrigerant = inputs["refrigerant"]
        
        if design_from_kw:
            # Calculate from kW requirement
            kw_results = self.design_from_kw(inputs)
            m_dot_ref = kw_results["m_dot_ref"]  # kg/s
            T_ref_out = kw_results["T_ref_out"]
            Q_latent = kw_results["Q_latent"]
            Q_sensible = kw_results["Q_sensible"]
            Q_total = kw_results["Q_total"]
            ref_props = kw_results["ref_props"]
        else:
            # Use direct mass flow input
            m_dot_ref = inputs["m_dot_ref"] / 3600  # kg/s
            T_evap = inputs["T_ref"]
            delta_T_superheat = inputs["delta_T_sh_sc"]
            ref_props = self.REFRIGERANTS[refrigerant]
            
            # Calculate heat duty (evaporator)
            # Q = m_dot_ref * (h_fg + cp_vapor * ΔT_superheat)
            Q_latent = m_dot_ref * ref_props["h_fg"]  # kW
            Q_sensible = m_dot_ref * ref_props["cp_vapor"] * delta_T_superheat  # kW
            Q_total = Q_latent + Q_sensible  # kW
            T_ref_out = T_evap + delta_T_superheat
            T_evap = inputs["T_ref"]
        
        T_evap = inputs["T_ref"]
        delta_T_superheat = inputs["delta_T_sh_sc"]
        
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
        
        # Get water/glycol properties
        sec_props = self.calculate_water_glycol_properties(T_sec_in, glycol_percent, glycol_type)
        
        # Convert secondary flow to kg/s
        m_dot_sec_kg = m_dot_sec_L * sec_props["rho"] / 1000
        
        # Tube dimensions
        tube_od = self.TUBE_SIZES[tube_size]
        tube_id = max(tube_od - 2 * tube_thickness, tube_od * 0.8)
        
        # Calculate shell diameter
        shell_diameter = self.calculate_shell_diameter(
            tube_od, n_tubes, tube_pitch, tube_layout
        )
        
        # Baffle spacing
        baffle_spacing = tube_length / (n_baffles + 1)
        
        # Calculate equivalent diameter for shell side
        if tube_layout == "triangular":
            D_e = 4 * (0.866 * tube_pitch**2 - 0.5 * math.pi * tube_od**2) / (math.pi * tube_od)
        else:
            D_e = 4 * (tube_pitch**2 - 0.25 * math.pi * tube_od**2) / (math.pi * tube_od)
        
        # Flow areas
        tube_flow_area = (math.pi * tube_id**2 / 4) * n_tubes / n_passes
        shell_cross_area = math.pi * shell_diameter**2 / 4
        tube_bundle_area = n_tubes * math.pi * tube_od**2 / 4
        shell_flow_area = (shell_cross_area - tube_bundle_area) * 0.4  # 40% flow area
        
        # Mass fluxes
        G_ref = m_dot_ref / tube_flow_area if tube_flow_area > 0 else 0
        G_sec = m_dot_sec_kg / shell_flow_area if shell_flow_area > 0 else 0
        
        # Heat transfer coefficients - TUBE SIDE (Refrigerant evaporation)
        # For DX evaporator, quality changes from ~0.2 to 1.0
        # Use average quality of 0.6 for design
        quality_avg = 0.6
        
        Re_l = G_ref * tube_id / ref_props["mu_liquid"]
        Pr_l = ref_props["pr_liquid"]
        
        h_ref = self.shah_evaporation(
            Re_l, Pr_l, quality_avg,
            ref_props["rho_liquid"], ref_props["rho_vapor"],
            tube_id, G_ref, ref_props["h_fg"] * 1000,
            ref_props["k_liquid"]
        )
        
        # Heat transfer coefficients - SHELL SIDE (Water/Glycol)
        Re_shell = G_sec * D_e / sec_props["mu"]
        h_shell = self.calculate_shell_side_htc(
            Re_shell, sec_props["pr"], D_e, sec_props["k"], tube_layout
        )
        
        # Overall U
        tube_k = self.TUBE_MATERIALS[tube_material]["k"]
        R_i = 1 / h_ref
        R_o = 1 / h_shell
        R_w = tube_od * math.log(tube_od / tube_id) / (2 * tube_k)
        R_f = 0.00035  # Combined fouling factor (m²K/W)
        
        U = 1 / (R_i + R_o + R_w + R_f)
        
        # Total heat transfer area
        A_total = math.pi * tube_od * tube_length * n_tubes
        
        # Calculate water outlet temperature
        if m_dot_sec_kg > 0:
            T_sec_out = T_sec_in - (Q_total * 1000) / (m_dot_sec_kg * sec_props["cp"])
        else:
            T_sec_out = T_sec_in
        
        # Calculate LMTD for evaporator
        dt1 = T_sec_in - T_evap
        dt2 = T_sec_out - T_ref_out
        
        if inputs["flow_arrangement"] != "counter":
            dt1 = abs(T_sec_in - T_evap)
            dt2 = abs(T_sec_out - T_ref_out)
        
        if dt1 <= 0 or dt2 <= 0 or abs(dt1 - dt2) < 1e-6:
            LMTD = min(dt1, dt2) if min(dt1, dt2) > 0 else 0
        else:
            LMTD = (dt1 - dt2) / math.log(dt1 / dt2)
        
        # Calculate required area
        A_required = (Q_total * 1000) / (U * LMTD) if U > 0 and LMTD > 0 else 0
        
        # Calculate effectiveness
        C_sec = m_dot_sec_kg * sec_props["cp"]  # Water capacity rate (W/K)
        NTU = U * A_total / C_sec if C_sec > 0 else 0
        effectiveness = 1 - math.exp(-NTU)
        
        # Pressure drops
        # Tube side (two-phase evaporation)
        Re_v = G_ref * tube_id / ref_props["mu_vapor"]
        dp_tube = self.friedel_two_phase_pressure_drop(
            Re_l, Re_v, quality_avg,
            ref_props["rho_liquid"], ref_props["rho_vapor"],
            ref_props["mu_liquid"], ref_props["mu_vapor"],
            tube_id, G_ref, tube_length * n_passes
        )
        
        # Shell side pressure drop
        if Re_shell < 2300:
            f_shell = 64 / Re_shell if Re_shell > 0 else 0.2
        else:
            f_shell = 0.2 * Re_shell**-0.2
        
        dp_shell = f_shell * (tube_length / D_e) * n_baffles * (sec_props["rho"] * (G_sec/sec_props["rho"])**2 / 2)
        
        # Calculate velocities
        # For two-phase refrigerant, use homogeneous density
        rho_tp = 1 / (quality_avg/ref_props["rho_vapor"] + (1-quality_avg)/ref_props["rho_liquid"])
        v_ref = G_ref / rho_tp
        v_sec = G_sec / sec_props["rho"]
        
        # Check velocities against recommendations
        sec_velocity_status = self.check_velocity_status(v_sec, glycol_percent, "shell")
        ref_velocity_status = self.check_velocity_status(v_ref, 0, "refrigerant_two_phase")
        
        # Refrigerant distribution check (DX-specific)
        m_dot_per_tube = m_dot_ref / n_tubes * 3600  # kg/hr
        distribution_status = "Good" if m_dot_per_tube >= 3.6 else "Marginal" if m_dot_per_tube >= 2.0 else "Poor"
        
        # Freeze protection check
        freeze_point = self.calculate_freeze_point(glycol_percent, glycol_type)
        freeze_risk = "High" if T_sec_out < freeze_point + 2 else "Medium" if T_sec_out < freeze_point + 3 else "Low"
        
        # Store results
        self.results = {
            # Basic info
            "heat_exchanger_type": "DX Evaporator",
            "refrigerant_side": "Tube side",
            "water_side": "Shell side",
            "design_method": "kW Input" if design_from_kw else "Mass Flow Input",
            
            # Thermal performance
            "heat_duty_kw": Q_total,
            "q_latent_kw": Q_latent,
            "q_sensible_kw": Q_sensible,
            "effectiveness": effectiveness,
            "ntu": NTU,
            "overall_u": U,
            "h_tube": h_ref,
            "h_shell": h_shell,
            "lmtd": LMTD,
            
            # Temperatures
            "t_sec_in": T_sec_in,
            "t_sec_out": T_sec_out,
            "t_ref_in": T_evap,
            "t_ref_out": T_ref_out,
            "water_deltaT": abs(T_sec_out - T_sec_in),
            "superheat": delta_T_superheat,
            
            # Flow rates
            "refrigerant_mass_flow_kg_hr": m_dot_ref * 3600,
            "water_vol_flow_L_hr": m_dot_sec_L * 3600,
            "water_mass_flow_kg_hr": m_dot_sec_kg * 3600,
            "flow_per_tube_kg_hr": m_dot_per_tube,
            
            # Geometry
            "shell_diameter_m": shell_diameter,
            "tube_pitch_mm": tube_pitch * 1000,
            "pitch_ratio": tube_pitch / tube_od if tube_od > 0 else 0,
            "tube_od_mm": tube_od * 1000,
            "tube_id_mm": tube_id * 1000,
            "area_total_m2": A_total,
            "area_required_m2": A_required,
            "area_ratio": A_total / A_required if A_required > 0 else 0,
            "baffle_spacing_m": baffle_spacing,
            
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
            
            # DX-specific
            "distribution_status": distribution_status,
            "freeze_point_c": freeze_point,
            "freeze_risk": freeze_risk,
            "glycol_type": glycol_type,
            "glycol_percentage": glycol_percent,
            
            # Design status
            "design_status": self.determine_design_status(effectiveness, A_total, A_required),
        }
        
        return self.results
    
    def design_condenser(self, inputs: Dict, design_from_kw: bool = False) -> Dict:
        """Design condenser - refrigerant in tubes, water/glycol in shell"""
        
        # Extract inputs
        refrigerant = inputs["refrigerant"]
        
        if design_from_kw:
            # Calculate from kW requirement
            kw_results = self.design_from_kw(inputs)
            m_dot_ref = kw_results["m_dot_ref"]  # kg/s
            T_ref_out = kw_results["T_ref_out"]
            Q_latent = kw_results["Q_latent"]
            Q_sensible = kw_results["Q_sensible"]
            Q_total = kw_results["Q_total"]
            ref_props = kw_results["ref_props"]
        else:
            # Use direct mass flow input
            m_dot_ref = inputs["m_dot_ref"] / 3600  # kg/s
            T_cond = inputs["T_ref"]
            delta_T_subcool = inputs["delta_T_sh_sc"]
            ref_props = self.REFRIGERANTS[refrigerant]
            
            # Calculate heat duty (condenser)
            # Q = m_dot_ref * (h_fg + cp_liquid * ΔT_subcool)
            Q_latent = m_dot_ref * ref_props["h_fg"]  # kW
            Q_sensible = m_dot_ref * ref_props["cp_liquid"] * delta_T_subcool  # kW
            Q_total = Q_latent + Q_sensible  # kW
            T_ref_out = T_cond - delta_T_subcool
            T_cond = inputs["T_ref"]
        
        T_cond = inputs["T_ref"]
        delta_T_subcool = inputs["delta_T_sh_sc"]
        
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
        
        # Get water/glycol properties
        sec_props = self.calculate_water_glycol_properties(T_sec_in, glycol_percent, glycol_type)
        
        # Convert secondary flow to kg/s
        m_dot_sec_kg = m_dot_sec_L * sec_props["rho"] / 1000
        
        # Tube dimensions
        tube_od = self.TUBE_SIZES[tube_size]
        tube_id = max(tube_od - 2 * tube_thickness, tube_od * 0.8)
        
        # Calculate shell diameter
        shell_diameter = self.calculate_shell_diameter(
            tube_od, n_tubes, tube_pitch, tube_layout
        )
        
        # Baffle spacing
        baffle_spacing = tube_length / (n_baffles + 1)
        
        # Calculate equivalent diameter for shell side
        if tube_layout == "triangular":
            D_e = 4 * (0.866 * tube_pitch**2 - 0.5 * math.pi * tube_od**2) / (math.pi * tube_od)
        else:
            D_e = 4 * (tube_pitch**2 - 0.25 * math.pi * tube_od**2) / (math.pi * tube_od)
        
        # Flow areas
        tube_flow_area = (math.pi * tube_id**2 / 4) * n_tubes / n_passes
        shell_cross_area = math.pi * shell_diameter**2 / 4
        tube_bundle_area = n_tubes * math.pi * tube_od**2 / 4
        shell_flow_area = (shell_cross_area - tube_bundle_area) * 0.4  # 40% flow area
        
        # Mass fluxes
        G_ref = m_dot_ref / tube_flow_area if tube_flow_area > 0 else 0
        G_sec = m_dot_sec_kg / shell_flow_area if shell_flow_area > 0 else 0
        
        # Heat transfer coefficients - TUBE SIDE (Refrigerant condensation)
        # For condenser, quality changes from 1.0 to ~0.0
        # Use average quality of 0.5 for design
        quality_avg = 0.5
        
        Re_l = G_ref * tube_id / ref_props["mu_liquid"]
        Re_v = G_ref * tube_id / ref_props["mu_vapor"]
        Pr_l = ref_props["pr_liquid"]
        
        h_ref = self.cavallini_condensation(
            Re_l, Re_v, Pr_l, quality_avg,
            ref_props["rho_liquid"], ref_props["rho_vapor"],
            ref_props["mu_liquid"], ref_props["mu_vapor"],
            tube_id, ref_props["k_liquid"]
        )
        
        # Heat transfer coefficients - SHELL SIDE (Water/Glycol)
        Re_shell = G_sec * D_e / sec_props["mu"]
        h_shell = self.calculate_shell_side_htc(
            Re_shell, sec_props["pr"], D_e, sec_props["k"], tube_layout
        )
        
        # Overall U
        tube_k = self.TUBE_MATERIALS[tube_material]["k"]
        R_i = 1 / h_ref
        R_o = 1 / h_shell
        R_w = tube_od * math.log(tube_od / tube_id) / (2 * tube_k)
        R_f = 0.00035  # Combined fouling factor (m²K/W)
        
        U = 1 / (R_i + R_o + R_w + R_f)
        
        # Total heat transfer area
        A_total = math.pi * tube_od * tube_length * n_tubes
        
        # Calculate water outlet temperature
        if m_dot_sec_kg > 0:
            T_sec_out = T_sec_in + (Q_total * 1000) / (m_dot_sec_kg * sec_props["cp"])
        else:
            T_sec_out = T_sec_in
        
        # Calculate LMTD for condenser
        dt1 = T_cond - T_sec_in
        dt2 = T_ref_out - T_sec_out
        
        if inputs["flow_arrangement"] != "counter":
            dt1 = abs(T_cond - T_sec_in)
            dt2 = abs(T_ref_out - T_sec_out)
        
        if dt1 <= 0 or dt2 <= 0 or abs(dt1 - dt2) < 1e-6:
            LMTD = min(dt1, dt2) if min(dt1, dt2) > 0 else 0
        else:
            LMTD = (dt1 - dt2) / math.log(dt1 / dt2)
        
        # Calculate required area
        A_required = (Q_total * 1000) / (U * LMTD) if U > 0 and LMTD > 0 else 0
        
        # Calculate effectiveness
        C_sec = m_dot_sec_kg * sec_props["cp"]  # Water capacity rate (W/K)
        NTU = U * A_total / C_sec if C_sec > 0 else 0
        effectiveness = 1 - math.exp(-NTU)
        
        # Pressure drops
        # Tube side (two-phase condensation)
        dp_tube = self.friedel_two_phase_pressure_drop(
            Re_l, Re_v, quality_avg,
            ref_props["rho_liquid"], ref_props["rho_vapor"],
            ref_props["mu_liquid"], ref_props["mu_vapor"],
            tube_id, G_ref, tube_length * n_passes
        )
        
        # Shell side pressure drop
        if Re_shell < 2300:
            f_shell = 64 / Re_shell if Re_shell > 0 else 0.2
        else:
            f_shell = 0.2 * Re_shell**-0.2
        
        dp_shell = f_shell * (tube_length / D_e) * n_baffles * (sec_props["rho"] * (G_sec/sec_props["rho"])**2 / 2)
        
        # Calculate velocities
        # For two-phase refrigerant, use homogeneous density
        rho_tp = 1 / (quality_avg/ref_props["rho_vapor"] + (1-quality_avg)/ref_props["rho_liquid"])
        v_ref = G_ref / rho_tp
        v_sec = G_sec / sec_props["rho"]
        
        # Check velocities against recommendations
        sec_velocity_status = self.check_velocity_status(v_sec, glycol_percent, "shell")
        ref_velocity_status = self.check_velocity_status(v_ref, 0, "refrigerant_two_phase")
        
        # Store results
        self.results = {
            # Basic info
            "heat_exchanger_type": "Condenser",
            "refrigerant_side": "Tube side",
            "water_side": "Shell side",
            "design_method": "kW Input" if design_from_kw else "Mass Flow Input",
            
            # Thermal performance
            "heat_duty_kw": Q_total,
            "q_latent_kw": Q_latent,
            "q_sensible_kw": Q_sensible,
            "effectiveness": effectiveness,
            "ntu": NTU,
            "overall_u": U,
            "h_tube": h_ref,
            "h_shell": h_shell,
            "lmtd": LMTD,
            
            # Temperatures
            "t_sec_in": T_sec_in,
            "t_sec_out": T_sec_out,
            "t_ref_in": T_cond,
            "t_ref_out": T_ref_out,
            "water_deltaT": abs(T_sec_out - T_sec_in),
            "subcool": delta_T_subcool,
            
            # Flow rates
            "refrigerant_mass_flow_kg_hr": m_dot_ref * 3600,
            "water_vol_flow_L_hr": m_dot_sec_L * 3600,
            "water_mass_flow_kg_hr": m_dot_sec_kg * 3600,
            
            # Geometry
            "shell_diameter_m": shell_diameter,
            "tube_pitch_mm": tube_pitch * 1000,
            "pitch_ratio": tube_pitch / tube_od if tube_od > 0 else 0,
            "tube_od_mm": tube_od * 1000,
            "tube_id_mm": tube_id * 1000,
            "area_total_m2": A_total,
            "area_required_m2": A_required,
            "area_ratio": A_total / A_required if A_required > 0 else 0,
            "baffle_spacing_m": baffle_spacing,
            
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
            
            # Glycol info
            "glycol_type": glycol_type,
            "glycol_percentage": glycol_percent,
            
            # Design status
            "design_status": self.determine_design_status(effectiveness, A_total, A_required),
        }
        
        return self.results
    
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
    
    def determine_design_status(self, effectiveness: float, area_total: float, area_required: float) -> str:
        """Determine overall design status"""
        area_ratio = area_total / area_required if area_required > 0 else 0
        
        if effectiveness >= 0.7 and area_ratio >= 0.95:
            return "Adequate"
        elif effectiveness >= 0.6 and area_ratio >= 0.9:
            return "Marginal"
        else:
            return "Inadequate"

# Helper function for number input with +/- buttons
def number_input_with_buttons(label: str, min_value: float, max_value: float, 
                            value: float, step: float, key: str, format: str = "%.1f",
                            help_text: str = None) -> float:
    """Create a number input with +/- buttons"""
    
    # Initialize session state for this input
    if key not in st.session_state:
        st.session_state[key] = value
    
    # Create columns for buttons and input
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("−", key=f"{key}_minus"):
            st.session_state[key] = max(min_value, st.session_state[key] - step)
    
    with col2:
        # Display current value
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
    
    # Design method selection
    st.sidebar.subheader("📊 Design Input Method")
    design_method = st.sidebar.radio(
        "Select Input Method",
        ["Heat Duty (kW)", "Refrigerant Mass Flow"],
        help="Design from total heat load or from compressor mass flow"
    )
    
    # Initialize designer for refrigerant list
    designer = DXHeatExchangerDesign()
    
    # Refrigerant parameters
    st.sidebar.subheader("🔧 Refrigerant Parameters")
    
    inputs["refrigerant"] = st.sidebar.selectbox(
        "Refrigerant Type",
        list(designer.REFRIGERANTS.keys())
    )
    
    if design_method == "Heat Duty (kW)":
        # kW input method
        inputs["heat_duty_kw"] = number_input_with_buttons(
            label="Heat Duty (kW)",
            min_value=1.0,
            max_value=5000.0,
            value=100.0,
            step=10.0,
            key="heat_duty",
            format="%.0f",
            help_text="Total heat load to be transferred"
        )
    else:
        # Mass flow input method
        inputs["m_dot_ref"] = number_input_with_buttons(
            label="Refrigerant Mass Flow (kg/hr)",
            min_value=10.0,
            max_value=10000.0,
            value=500.0,
            step=10.0,
            key="m_dot_ref",
            format="%.0f",
            help_text="From compressor specification sheet"
        )
    
    # Temperature parameters
    if inputs["hex_type"] == "DX Evaporator":
        inputs["T_ref"] = number_input_with_buttons(
            label="Evaporating Temperature (°C)",
            min_value=-50.0,
            max_value=20.0,
            value=5.0,
            step=1.0,
            key="T_evap",
            format="%.1f"
        )
        
        inputs["delta_T_sh_sc"] = number_input_with_buttons(
            label="Superheat at Exit (K)",
            min_value=3.0,
            max_value=15.0,
            value=5.0,
            step=0.5,
            key="superheat",
            format="%.1f",
            help_text="DX evaporators require 3-8K superheat for proper TXV operation"
        )
    else:
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
            label="Subcool at Exit (K)",
            min_value=0.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
            key="subcool",
            format="%.1f"
        )
    
    st.sidebar.markdown("---")
    
    # Water/Glycol parameters
    st.sidebar.subheader("💧 Water/Glycol Side")
    
    # Glycol type selection
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
    
    # Display glycol badge
    st.sidebar.markdown(f"""
    <div style="background-color: {bg_color}; color: {text_color}; padding: 0.5rem; 
                border-radius: 0.5rem; text-align: center; font-weight: bold; margin: 0.5rem 0;">
        {glycol_label}
    </div>
    """, unsafe_allow_html=True)
    
    # Glycol percentage
    if "Glycol" in glycol_choice:
        # Use number input instead of slider
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
        
        # Show freeze point
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
    
    # Flow arrangement
    inputs["flow_arrangement"] = st.sidebar.radio(
        "Flow Arrangement",
        ["Counter", "Parallel"]
    ).lower()
    
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
    
    # Tube thickness
    inputs["tube_thickness"] = number_input_with_buttons(
        label="Tube Thickness (mm)",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
        key="tube_thickness",
        format="%.1f"
    )
    
    # Tube pitch
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
    
    # Number of passes
    inputs["n_passes"] = st.sidebar.selectbox(
        "Tube Passes",
        [1, 2, 4, 6],
        help="Number of times refrigerant passes through tubes"
    )
    
    # Number of baffles
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
    
    # Number of tubes
    inputs["n_tubes"] = int(number_input_with_buttons(
        label="Number of Tubes",
        min_value=1,
        max_value=500,
        value=100,
        step=1,
        key="n_tubes",
        format="%.0f"
    ))
    
    # Tube length
    inputs["tube_length"] = number_input_with_buttons(
        label="Tube Length (m)",
        min_value=0.5,
        max_value=10.0,
        value=3.0,
        step=0.1,
        key="tube_length",
        format="%.1f"
    )
    
    # Tube layout
    inputs["tube_layout"] = st.sidebar.radio(
        "Tube Layout",
        ["Triangular", "Square"],
        help="Triangular: Higher heat transfer, more compact\nSquare: Easier cleaning, lower pressure drop"
    )
    
    # Glycol properties info
    with st.sidebar.expander("🧪 Glycol Properties"):
        st.markdown("""
        **Ethylene Glycol (EG):**
        - Better heat transfer
        - Lower viscosity
        - Toxic - not for food applications
        - Lower cost
        
        **Propylene Glycol (PG):**
        - Food grade, non-toxic
        - Higher viscosity
        - Slightly lower heat transfer
        - Higher cost
        
        **Freeze Protection:**
        - 20% EG: -7.5°C
        - 30% EG: -14°C
        - 40% EG: -23°C
        - 50% EG: -36°C
        """)
    
    return inputs, design_method

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

def display_results(results: Dict, inputs: Dict, design_method: str):
    """Display calculation results"""
    
    # Header with badges
    if results["heat_exchanger_type"] == "DX Evaporator":
        header_html = f"""
        <div style='display: flex; align-items: center;'>
            <h2>📊 DX Evaporator Design Results</h2>
            <span class="dx-badge">DX Type</span>
            {display_glycol_badge(results['glycol_type'], results['glycol_percentage'])}
            <span style="margin-left: 10px; background-color: #FEF3C7; color: #92400E; padding: 0.25rem 0.5rem; border-radius: 0.5rem; font-size: 0.85rem;">
                {results['design_method']}
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
                {results['design_method']}
            </span>
        </div>
        """
    
    st.markdown(header_html, unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Heat Duty", f"{results['heat_duty_kw']:.1f} kW")
        st.caption(f"Latent: {results['q_latent_kw']:.1f} kW | Sensible: {results['q_sensible_kw']:.1f} kW")
    
    with col2:
        status_color = "normal" if results['design_status'] == "Adequate" else "off" if results['design_status'] == "Marginal" else "inverse"
        st.metric("Design Status", results['design_status'], delta_color=status_color)
    
    with col3:
        st.metric("Effectiveness", f"{results['effectiveness']:.3f}")
        st.caption(f"NTU: {results['ntu']:.2f}")
    
    with col4:
        st.metric("Area Ratio", f"{results['area_ratio']:.2f}")
        st.caption(f"{results['area_total_m2']:.1f} m² / {results['area_required_m2']:.1f} m²")
    
    st.markdown("---")
    
    # Flow parameters
    st.markdown("### 💧 Flow Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Refrigerant Side (Tubes)")
        st.write(f"**Mass Flow:** {results['refrigerant_mass_flow_kg_hr']:.1f} kg/hr")
        st.write(f"**Inlet Temp:** {results['t_ref_in']:.1f} °C")
        st.write(f"**Outlet Temp:** {results['t_ref_out']:.1f} °C")
        
        if results["heat_exchanger_type"] == "DX Evaporator":
            st.write(f"**Superheat:** {results['superheat']:.1f} K")
        else:
            st.write(f"**Subcool:** {results['subcool']:.1f} K")
        
        st.markdown(f"**Velocity:** {display_velocity_indicator(results['velocity_tube_ms'], results['velocity_tube_status'])}", unsafe_allow_html=True)
        st.write(f"**Pressure Drop:** {results['dp_tube_kpa']:.2f} kPa")
        
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
        
        if results["heat_exchanger_type"] == "DX Evaporator" and results['glycol_percentage'] > 0:
            st.write(f"**Freeze Point:** {results['freeze_point_c']:.1f}°C")
            st.write(f"**Freeze Risk:** {results['freeze_risk']}")
    
    st.markdown("---")
    
    # Thermal performance
    st.markdown("### 🌡️ Thermal Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Heat Transfer")
        st.write(f"**Overall U:** {results['overall_u']:.1f} W/m²K")
        st.write(f"**Tube HTC:** {results['h_tube']:.0f} W/m²K")
        st.write(f"**Shell HTC:** {results['h_shell']:.0f} W/m²K")
        st.write(f"**LMTD:** {results['lmtd']:.1f} K")
    
    with col2:
        st.markdown("#### Geometry")
        st.write(f"**Shell Diameter:** {results['shell_diameter_m']*1000:.0f} mm")
        st.write(f"**Tube OD:** {results['tube_od_mm']:.1f} mm")
        st.write(f"**Tube ID:** {results['tube_id_mm']:.1f} mm")
        st.write(f"**Tube Pitch:** {results['tube_pitch_mm']:.1f} mm")
        st.write(f"**Pitch/OD Ratio:** {results['pitch_ratio']:.2f}")
        st.write(f"**Tube Layout:** {inputs['tube_layout']}")
    
    with col3:
        st.markdown("#### Flow Characteristics")
        st.write(f"**Tube Re:** {results['reynolds_tube']:,.0f}")
        st.write(f"**Shell Re:** {results['reynolds_shell']:,.0f}")
        st.write(f"**Tube Mass Flux:** {results['mass_flux_tube']:.1f} kg/m²s")
        st.write(f"**Shell Mass Flux:** {results['mass_flux_shell']:.1f} kg/m²s")
        st.write(f"**Baffle Spacing:** {results['baffle_spacing_m']:.2f} m")
    
    st.markdown("---")
    
    # Design recommendations
    st.markdown("### 💡 Design Recommendations")
    
    if results['design_status'] == "Inadequate":
        st.error(f"""
        **DESIGN INADEQUATE**
        
        **Issues:**
        1. Effectiveness too low ({results['effectiveness']:.3f}, target ≥0.7)
        2. Area ratio too low ({results['area_ratio']:.2f}, target ≥0.95)
        
        **Solutions:**
        1. Increase water flow rate to improve velocity
        2. Add more tubes or increase tube length
        3. Reduce tube pitch to fit more tubes
        4. Consider different tube layout
        """)
    elif results['design_status'] == "Marginal":
        st.warning(f"""
        **DESIGN MARGINAL**
        
        **Considerations:**
        1. Effectiveness: {results['effectiveness']:.3f} (target ≥0.7)
        2. Area ratio: {results['area_ratio']:.2f} (target ≥0.95)
        
        **Recommendations:**
        1. Fine-tune water flow rate
        2. Consider minor geometry adjustments
        3. Monitor performance in operation
        """)
    else:
        st.success(f"""
        **DESIGN ADEQUATE** ✅
        
        **Performance Summary:**
        1. Effectiveness: {results['effectiveness']:.3f} (good)
        2. Area ratio: {results['area_ratio']:.2f} (adequate)
        3. Water velocity: {results['velocity_shell_ms']:.2f} m/s ({results['velocity_shell_status']['status']})
        4. Pressure drop: {results['dp_shell_kpa']:.2f} kPa
        
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
        
        if results['distribution_status'] == "Poor":
            st.error(f"""
            **POOR REFRIGERANT DISTRIBUTION**
            
            Flow per tube ({results['flow_per_tube_kg_hr']:.1f} kg/hr) is too low for good distribution.
            
            **Solutions:**
            1. Reduce number of tubes
            2. Increase refrigerant flow (if compressor allows)
            3. Use enhanced distributor design
            4. Consider individual TXVs per circuit
            """)
        
        if results['freeze_risk'] in ["High", "Medium"]:
            st.warning(f"""
            **FREEZE RISK DETECTED**
            
            Water outlet temperature ({results['t_sec_out']:.1f}°C) is close to freeze point ({results['freeze_point_c']:.1f}°C).
            
            **Recommendations:**
            1. Increase glycol percentage
            2. Increase water flow rate
            3. Add freeze protection controls
            4. Monitor temperature closely
            """)
        
        if results['superheat'] < 3.0:
            st.warning(f"""
            **LOW SUPERHEAT** ({results['superheat']:.1f} K)
            
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
        report_data = {
            "Parameter": [
                "Heat Exchanger Type", "Design Method", "Refrigerant",
                "Glycol Type", "Glycol Percentage", "Freeze Point (°C)",
                "Heat Duty (kW)", "Latent Heat (kW)", "Sensible Heat (kW)",
                "Effectiveness", "NTU", "Overall U (W/m²K)",
                "Tube HTC (W/m²K)", "Shell HTC (W/m²K)", "LMTD (K)",
                "Refrigerant Mass Flow (kg/hr)", "Water Flow Rate (L/hr)",
                "Water Mass Flow (kg/hr)", "Water Inlet Temp (°C)",
                "Water Outlet Temp (°C)", "Water ΔT (K)",
                "Refrigerant Inlet Temp (°C)", "Refrigerant Outlet Temp (°C)",
                "Superheat/Subcool (K)", "Shell Diameter (mm)",
                "Tube OD (mm)", "Tube ID (mm)", "Tube Pitch (mm)",
                "Pitch/OD Ratio", "Tube Layout", "Number of Tubes",
                "Tube Length (m)", "Tube Passes", "Number of Baffles",
                "Baffle Spacing (m)", "Total Area (m²)", "Required Area (m²)",
                "Area Ratio", "Water Velocity (m/s)", "Velocity Status",
                "Refrigerant Velocity (m/s)", "Shell ΔP (kPa)", "Tube ΔP (kPa)",
                "Shell Reynolds", "Tube Reynolds", "Design Status"
            ],
            "Value": [
                results["heat_exchanger_type"], results["design_method"], inputs["refrigerant"],
                results["glycol_type"].title(), f"{results['glycol_percentage']}%", f"{results.get('freeze_point_c', 0):.1f}",
                f"{results['heat_duty_kw']:.1f}", f"{results['q_latent_kw']:.1f}", f"{results['q_sensible_kw']:.1f}",
                f"{results['effectiveness']:.3f}", f"{results['ntu']:.2f}", f"{results['overall_u']:.1f}",
                f"{results['h_tube']:.0f}", f"{results['h_shell']:.0f}", f"{results['lmtd']:.1f}",
                f"{results['refrigerant_mass_flow_kg_hr']:.1f}", f"{results['water_vol_flow_L_hr']:,.0f}",
                f"{results['water_mass_flow_kg_hr']:,.0f}", f"{results['t_sec_in']:.1f}",
                f"{results['t_sec_out']:.1f}", f"{results['water_deltaT']:.1f}",
                f"{results['t_ref_in']:.1f}", f"{results['t_ref_out']:.1f}",
                f"{results.get('superheat', results.get('subcool', 0)):.1f}",
                f"{results['shell_diameter_m']*1000:.0f}", f"{results['tube_od_mm']:.1f}",
                f"{results['tube_id_mm']:.1f}", f"{results['tube_pitch_mm']:.1f}",
                f"{results['pitch_ratio']:.2f}", inputs['tube_layout'], str(inputs['n_tubes']),
                f"{inputs['tube_length']}", str(inputs['n_passes']), str(inputs['n_baffles']),
                f"{results['baffle_spacing_m']:.3f}", f"{results['area_total_m2']:.2f}",
                f"{results['area_required_m2']:.2f}", f"{results['area_ratio']:.2f}",
                f"{results['velocity_shell_ms']:.2f}", results['velocity_shell_status']['status'],
                f"{results['velocity_tube_ms']:.2f}", f"{results['dp_shell_kpa']:.2f}",
                f"{results['dp_tube_kpa']:.2f}", f"{results['reynolds_shell']:,.0f}",
                f"{results['reynolds_tube']:,.0f}", results['design_status']
            ],
            "Unit": [
                "", "", "",
                "", "", "°C",
                "kW", "kW", "kW",
                "-", "-", "W/m²K",
                "W/m²K", "W/m²K", "K",
                "kg/hr", "L/hr",
                "kg/hr", "°C",
                "°C", "K",
                "°C", "°C",
                "K", "mm",
                "mm", "mm", "mm",
                "-", "", "",
                "m", "", "",
                "m", "m²", "m²",
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
    st.markdown("### Direct Expansion (DX) Evaporator & Condenser | kW or Mass Flow Input | Water/Glycol in Shell")
    
    # Important note
    st.info("""
    **🔧 This tool designs DX (Direct Expansion) type shell & tube heat exchangers:**
    - **Two input methods:** Heat Duty (kW) OR Refrigerant Mass Flow
    - **Refrigerant flows inside tubes** (evaporates or condenses)
    - **Water/Glycol flows in shell side**
    - **Supports both Ethylene and Propylene (Food Grade) glycols**
    - **Manual number inputs with +/- buttons** for precise control
    """)
    
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'inputs' not in st.session_state:
        st.session_state.inputs = None
    if 'design_method' not in st.session_state:
        st.session_state.design_method = None
    
    # Create layout
    col1, col2 = st.columns([3, 1])
    
    with col2:
        inputs, design_method = create_input_section()
        
        # Calculate button
        button_label = "🚀 Calculate DX Design" if inputs["hex_type"] == "DX Evaporator" else "🚀 Calculate Condenser Design"
        
        if st.sidebar.button(button_label, type="primary", use_container_width=True):
            with st.spinner("Performing engineering calculations..."):
                designer = DXHeatExchangerDesign()
                
                # Convert hex_type for internal use
                calc_inputs = inputs.copy()
                calc_inputs["hex_type"] = calc_inputs["hex_type"].lower().replace("dx ", "")
                
                # Determine if using kW or mass flow
                design_from_kw = (design_method == "Heat Duty (kW)")
                
                if calc_inputs["hex_type"] == "evaporator":
                    results = designer.design_dx_evaporator(calc_inputs, design_from_kw)
                else:
                    results = designer.design_condenser(calc_inputs, design_from_kw)
                
                st.session_state.results = results
                st.session_state.inputs = inputs
                st.session_state.design_method = design_method
                st.rerun()
        
        # Reset button
        if st.sidebar.button("🔄 Reset Design", use_container_width=True):
            st.session_state.results = None
            st.session_state.inputs = None
            st.session_state.design_method = None
            st.rerun()
        
        # Quick tips
        st.sidebar.markdown("---")
        with st.sidebar.expander("💡 Design Tips"):
            st.markdown("""
            **Input Methods:**
            - **Heat Duty (kW):** Use when you know total cooling/heating load
            - **Mass Flow:** Use when you have compressor specifications
            
            **For DX Evaporators:**
            1. Ensure minimum 3K superheat for TXV
            2. Check refrigerant distribution
            3. Monitor freeze risk with glycol
            
            **For Condensers:**
            1. 5-10K subcool typical
            2. Higher water temperatures reduce LMTD
            3. Consider fouling factors
            
            **Using +/- buttons:**
            - Click buttons for quick adjustments
            - Or type directly in number boxes
            - All inputs support decimal values
            """)
    
    with col1:
        if st.session_state.results is not None:
            display_results(st.session_state.results, st.session_state.inputs, st.session_state.design_method)
        else:
            st.markdown("""
            ## 🔧 DX Heat Exchanger Design Tool
            
            **Complete design tool for DX (Direct Expansion) shell & tube heat exchangers with two input methods:**
            
            ### **Two Input Methods:**
            
            1. **Heat Duty (kW) Input:**
               - Enter total cooling/heating load
               - Program calculates required refrigerant flow
               - Ideal for system design from load requirements
            
            2. **Refrigerant Mass Flow Input:**
               - Enter compressor mass flow rate
               - Program calculates heat duty
               - Ideal for matching existing compressor
            
            ### **Enhanced Features:**
            
            ✅ **Manual Number Inputs:**
               - **No sliders** - type any value directly
               - **+/- buttons** for quick adjustments
               - **Precise control** over all parameters
               - **Decimal values** supported everywhere
            
            ✅ **Comprehensive Glycol Support:**
               - Ethylene Glycol (EG) - industrial use
               - Propylene Glycol (PG) - food grade, non-toxic
               - Freeze point calculations
               - Temperature-dependent properties
            
            ✅ **Advanced Engineering:**
               - Shah correlation for evaporation
               - Cavallini correlation for condensation
               - Gnielinski correlation for single-phase
               - Friedel correlation for two-phase ΔP
            
            ### **How to Use:**
            
            1. **Select heat exchanger type** (DX Evaporator or Condenser)
            2. **Choose input method** (kW or Mass Flow)
            3. **Enter parameters** using +/- buttons or direct typing
            4. **Choose glycol type** and percentage (if needed)
            5. **Configure geometry** (tubes, pitch, layout)
            6. **Click Calculate** and review results
            7. **Optimize** based on recommendations
            
            ### **Password Protected**
            Enter password: **Semaanju**
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🔧 <strong>DX Shell & Tube Heat Exchanger Designer</strong> | kW & Mass Flow Input | Manual Number Inputs</p>
        <p>🧪 Ethylene & Propylene Glycol Support | 🎯 +/- Button Controls | 📊 Advanced Engineering Correlations</p>
        <p>⚠️ For flooded evaporators (water in tubes), use separate design tool</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
