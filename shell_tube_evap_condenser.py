import streamlit as st
import numpy as np
import pandas as pd
import math
from scipy.optimize import fsolve
import plotly.graph_objects as go
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Shell & Tube Heat Exchanger Designer",
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
</style>
""", unsafe_allow_html=True)

class HeatExchangerDesign:
    """Engineering-grade heat exchanger design with ε-NTU method"""
    
    # Physical properties database
    REFRIGERANTS = {
        "R134a": {
            "cp_vapor": 0.85,  # kJ/kg·K at 5°C
            "cp_liquid": 1.43,  # kJ/kg·K
            "h_fg": 198.7,  # kJ/kg at 5°C
            "rho_vapor": 14.4,  # kg/m³ at 5°C
            "rho_liquid": 1278,  # kg/m³
            "mu_vapor": 1.11e-5,  # Pa·s
            "mu_liquid": 2.04e-4,  # Pa·s
            "k_vapor": 0.0116,  # W/m·K
            "k_liquid": 0.085,  # W/m·K
            "pr_vapor": 0.82,
            "pr_liquid": 3.43
        },
        "R404A": {
            "cp_vapor": 0.82,
            "cp_liquid": 1.55,
            "h_fg": 163.3,
            "rho_vapor": 33.2,
            "rho_liquid": 1132,
            "mu_vapor": 1.23e-5,
            "mu_liquid": 1.98e-4,
            "k_vapor": 0.0108,
            "k_liquid": 0.072,
            "pr_vapor": 0.94,
            "pr_liquid": 4.26
        },
        "R407C": {
            "cp_vapor": 1.25,
            "cp_liquid": 1.45,
            "h_fg": 200.0,
            "rho_vapor": 30.0,
            "rho_liquid": 1150.0,
            "mu_vapor": 1.25e-5,
            "mu_liquid": 1.9e-4,
            "k_vapor": 0.0125,
            "k_liquid": 0.077,
            "pr_vapor": 0.79,
            "pr_liquid": 2.9
        },
        "R410A": {
            "cp_vapor": 1.30,
            "cp_liquid": 1.55,
            "h_fg": 190.0,
            "rho_vapor": 35.0,
            "rho_liquid": 1120.0,
            "mu_vapor": 1.1e-5,
            "mu_liquid": 1.7e-4,
            "k_vapor": 0.013,
            "k_liquid": 0.076,
            "pr_vapor": 0.81,
            "pr_liquid": 2.7
        },
        "Ammonia (R717)": {
            "cp_vapor": 2.18,
            "cp_liquid": 4.69,
            "h_fg": 1261.0,
            "rho_vapor": 4.26,
            "rho_liquid": 625.2,
            "mu_vapor": 9.9e-6,
            "mu_liquid": 1.35e-4,
            "k_vapor": 0.0246,
            "k_liquid": 0.502,
            "pr_vapor": 0.88,
            "pr_liquid": 1.26
        }
    }
    
    # Tube materials properties
    TUBE_MATERIALS = {
        "Copper": {"k": 386, "density": 8960, "cost_factor": 1.0},
        "Cu-Ni 90/10": {"k": 40, "density": 8940, "cost_factor": 1.8},
        "Steel": {"k": 50, "density": 7850, "cost_factor": 0.6},
        "Aluminum Brass": {"k": 100, "density": 8300, "cost_factor": 1.2}
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
    
    # Glycol properties
    GLYCOL_PROPERTIES = {
        0: {"cp": 4.186, "rho": 998.2, "mu": 0.001, "k": 0.598, "pr": 7.01},
        10: {"cp": 4.08, "rho": 1022, "mu": 0.0013, "k": 0.57, "pr": 9.3},
        20: {"cp": 3.95, "rho": 1040, "mu": 0.0018, "k": 0.54, "pr": 13.2},
        30: {"cp": 3.78, "rho": 1057, "mu": 0.0026, "k": 0.51, "pr": 19.3},
        40: {"cp": 3.60, "rho": 1069, "mu": 0.0040, "k": 0.47, "pr": 30.6},
        50: {"cp": 3.42, "rho": 1077, "mu": 0.0068, "k": 0.43, "pr": 54.1}
    }
    
    def __init__(self):
        self.results = {}
    
    def calculate_water_glycol_properties(self, temperature: float, glycol_percentage: int) -> Dict:
        """Calculate water/glycol mixture properties"""
        base_props = self.GLYCOL_PROPERTIES.get(glycol_percentage, self.GLYCOL_PROPERTIES[0])
        temp_factor = 1 + 0.02 * (temperature - 20) / 50
        return {
            "cp": base_props["cp"] * temp_factor * 1000,  # J/kg·K
            "rho": base_props["rho"] / temp_factor,
            "mu": base_props["mu"] / temp_factor,
            "k": base_props["k"] * temp_factor,
            "pr": base_props["pr"] / temp_factor
        }
    
    def calculate_ntu_effectiveness(self, C_h: float, C_c: float, U: float, A: float, 
                                  flow_arrangement: str, hex_type: str) -> Tuple[float, float]:
        """Calculate NTU and effectiveness using ε-NTU method"""
        if hex_type in ["evaporator", "condenser"]:
            C_min = C_c  # Secondary fluid capacity
            C_r = 0  # Phase change
        else:
            C_min = min(C_h, C_c)
            C_max = max(C_h, C_c)
            C_r = C_min / C_max if C_max > 0 else 0
        
        NTU = U * A / C_min if C_min > 0 else 0
        
        if C_r == 0:
            effectiveness = 1 - math.exp(-NTU)
        elif flow_arrangement == "counter":
            effectiveness = (1 - math.exp(-NTU * (1 - C_r))) / (1 - C_r * math.exp(-NTU * (1 - C_r)))
        else:
            effectiveness = (1 - math.exp(-NTU * (1 + C_r))) / (1 + C_r)
        
        return NTU, effectiveness
    
    def calculate_two_phase_htc(self, refrigerant: str, quality: float, G: float, 
                              D: float, hex_type: str = "evaporator") -> float:
        """Calculate two-phase heat transfer coefficient"""
        props = self.REFRIGERANTS[refrigerant]
        
        # Single-phase liquid HTC
        Re_l = G * D / props["mu_liquid"]
        Pr_l = props["pr_liquid"]
        Nu_l = 0.023 * Re_l**0.8 * Pr_l**0.4 if Re_l > 2300 else 4.36
        h_l = Nu_l * props["k_liquid"] / D
        
        # Two-phase multiplier
        if hex_type == "evaporator":
            if quality <= 0:
                return h_l
            elif quality >= 1:
                Re_v = G * D / props["mu_vapor"]
                Pr_v = props["pr_vapor"]
                Nu_v = 0.023 * Re_v**0.8 * Pr_v**0.4 if Re_v > 2300 else 4.36
                return Nu_v * props["k_vapor"] / D
            else:
                return h_l * (1 + 10 * quality**0.8)
        else:
            if quality >= 1:
                Re_v = G * D / props["mu_vapor"]
                Pr_v = props["pr_vapor"]
                Nu_v = 0.023 * Re_v**0.8 * Pr_v**0.4 if Re_v > 2300 else 4.36
                return Nu_v * props["k_vapor"] / D
            elif quality <= 0:
                return h_l
            else:
                return h_l * (1 + 8 * (1 - quality)**0.8)
    
    def calculate_shell_diameter(self, tube_od: float, n_tubes: int, pitch: float,
                               tube_layout: str = "triangular") -> float:
        """
        Calculate shell diameter based on tube count, pitch, and layout
        Using TEMA standards
        """
        # Calculate bundle diameter
        if tube_layout == "triangular":
            # TEMA constants for triangular pitch
            pitch_ratio = pitch / tube_od
            if 1.2 <= pitch_ratio <= 1.3:
                K1 = 0.319
                n1 = 2.142
            elif 1.3 < pitch_ratio <= 1.5:
                K1 = 0.249
                n1 = 2.207
            else:
                K1 = 0.175
                n1 = 2.285
        else:  # square layout
            pitch_ratio = pitch / tube_od
            if 1.2 <= pitch_ratio <= 1.3:
                K1 = 0.215
                n1 = 2.207
            elif 1.3 < pitch_ratio <= 1.5:
                K1 = 0.156
                n1 = 2.291
            else:
                K1 = 0.158
                n1 = 2.263
        
        # Bundle diameter using TEMA formula
        bundle_diameter = tube_od * (n_tubes / K1) ** (1 / n1)
        
        # Alternative calculation based on pitch
        if tube_layout == "triangular":
            # For triangular pitch, tubes per row = sqrt(n_tubes/0.866)
            tubes_per_row = math.sqrt(n_tubes / 0.866)
        else:
            tubes_per_row = math.sqrt(n_tubes)
        
        diameter_by_pitch = pitch * tubes_per_row
        
        # Use the larger value
        bundle_diameter = max(bundle_diameter, diameter_by_pitch)
        
        # Add clearances: tube to shell = 10-15mm, tube to tube = pitch - tube_od
        clearance = 0.015  # 15mm minimum clearance
        shell_diameter = bundle_diameter + 2 * clearance
        
        return max(shell_diameter, 0.1)  # Minimum 100mm
    
    def calculate_shell_flow_area(self, shell_diameter: float, tube_od: float, 
                                pitch: float, tube_layout: str, baffle_spacing: float,
                                n_baffles: int) -> float:
        """
        Calculate shell-side flow area considering tube layout and pitch
        """
        if tube_layout == "triangular":
            # For triangular pitch, effective flow area
            # Cross-flow area between tubes
            pitch_effective = pitch * math.cos(math.radians(30))
            free_area = (pitch_effective - tube_od) * baffle_spacing
        else:  # square
            free_area = (pitch - tube_od) * baffle_spacing
        
        # Number of flow lanes
        flow_lanes = shell_diameter / pitch
        
        # Total flow area
        flow_area = free_area * flow_lanes
        
        # Baffle window area (simplified)
        baffle_cut = 0.25  # 25% baffle cut typical
        window_area = (math.pi * shell_diameter**2 / 4) * baffle_cut
        
        # Use the smaller of cross-flow and window area
        return min(flow_area, window_area)
    
    def calculate_baffle_spacing(self, tube_length: float, n_baffles: int) -> float:
        """Calculate baffle spacing"""
        return tube_length / (n_baffles + 1)
    
    def calculate_overall_u(self, h_i: float, h_o: float, tube_k: float,
                          tube_id: float, tube_od: float) -> float:
        """Calculate overall heat transfer coefficient"""
        R_i = 1 / (h_i * (tube_id / tube_od))
        R_o = 1 / h_o
        R_w = math.log(tube_od / tube_id) / (2 * math.pi * tube_k)
        R_f = 0.0002 * (1 + tube_od / tube_id)  # Fouling
        U = 1 / (R_i + R_o + R_w + R_f)
        return U
    
    def design_heat_exchanger(self, inputs: Dict) -> Dict:
        """Main design calculation using ε-NTU method"""
        
        # Extract inputs
        hex_type = inputs["hex_type"].lower()
        refrigerant = inputs["refrigerant"]
        m_dot_ref = inputs["m_dot_ref"] / 3600  # kg/s
        T_ref = inputs["T_ref"]
        delta_T = inputs["delta_T_sh_sc"]
        
        # Secondary fluid
        glycol_percent = inputs["glycol_percentage"]
        m_dot_sec_L = inputs["m_dot_sec"] / 3600  # L/s
        T_sec_in = inputs["T_sec_in"]
        
        # Geometry - NOW WITH PITCH INPUT
        tube_size = inputs["tube_size"]
        tube_material = inputs["tube_material"]
        tube_thickness = inputs["tube_thickness"] / 1000  # m
        tube_pitch = inputs["tube_pitch"] / 1000  # mm to m - NEW
        n_passes = inputs["n_passes"]
        n_baffles = inputs["n_baffles"]
        n_tubes = inputs["n_tubes"]
        tube_length = inputs["tube_length"]
        tube_layout = inputs["tube_layout"].lower()
        
        # Get properties
        ref_props = self.REFRIGERANTS[refrigerant]
        sec_props = self.calculate_water_glycol_properties(T_sec_in, glycol_percent)
        
        # Convert secondary flow to kg/s
        m_dot_sec_kg = m_dot_sec_L * sec_props["rho"] / 1000
        
        # Calculate heat duty
        if hex_type == "evaporator":
            Q_total = m_dot_ref * (ref_props["h_fg"] + ref_props["cp_vapor"] * delta_T)
            T_ref_out = T_ref + delta_T
        else:
            Q_total = m_dot_ref * (ref_props["h_fg"] + ref_props["cp_liquid"] * delta_T)
            T_ref_out = T_ref - delta_T
        
        # Tube dimensions
        tube_od = self.TUBE_SIZES[tube_size]
        tube_id = tube_od - 2 * tube_thickness
        if tube_id <= 0:
            tube_id = tube_od * 0.8
        
        # Calculate shell diameter WITH PITCH
        shell_diameter = self.calculate_shell_diameter(
            tube_od, n_tubes, tube_pitch, tube_layout
        )
        
        # Calculate baffle spacing
        baffle_spacing = self.calculate_baffle_spacing(tube_length, n_baffles)
        
        # Calculate shell-side flow area WITH PITCH
        shell_flow_area = self.calculate_shell_flow_area(
            shell_diameter, tube_od, tube_pitch, tube_layout, 
            baffle_spacing, n_baffles
        )
        
        # Tube-side flow area
        tube_flow_area = (math.pi * tube_id**2 / 4) * n_tubes / n_passes
        
        # Mass fluxes
        G_ref = m_dot_ref / tube_flow_area if tube_flow_area > 0 else 0
        v_sec = m_dot_sec_kg / (sec_props["rho"] * shell_flow_area) if shell_flow_area > 0 else 0
        
        # Heat transfer coefficients
        if hex_type == "evaporator":
            h_ref = self.calculate_two_phase_htc(refrigerant, 0.5, G_ref, tube_id, "evaporator")
        else:
            h_ref = self.calculate_two_phase_htc(refrigerant, 0.5, G_ref, tube_id, "condenser")
        
        # Shell-side HTC
        if tube_layout == "triangular":
            D_e = 4 * (tube_pitch**2 * math.sqrt(3)/4 - math.pi * tube_od**2/8) / (math.pi * tube_od/2)
        else:
            D_e = 4 * (tube_pitch**2 - math.pi * tube_od**2/4) / (math.pi * tube_od)
        
        Re_shell = sec_props["rho"] * v_sec * D_e / sec_props["mu"]
        Nu_shell = 0.36 * Re_shell**0.55 * sec_props["pr"]**(1/3) if Re_shell > 100 else 3.66
        h_shell = Nu_shell * sec_props["k"] / D_e
        
        # Overall U
        tube_k = self.TUBE_MATERIALS[tube_material]["k"]
        U = self.calculate_overall_u(h_ref, h_shell, tube_k, tube_id, tube_od)
        
        # Capacity rates
        C_sec = m_dot_sec_kg * sec_props["cp"]  # W/K
        
        # Total area
        A_total = math.pi * tube_od * tube_length * n_tubes
        
        # Calculate NTU and effectiveness
        if hex_type == "evaporator":
            NTU, effectiveness = self.calculate_ntu_effectiveness(
                1e10, C_sec, U, A_total, inputs["flow_arrangement"], hex_type
            )
        else:
            NTU, effectiveness = self.calculate_ntu_effectiveness(
                C_sec, 1e10, U, A_total, inputs["flow_arrangement"], hex_type
            )
        
        # Calculate outlet temperatures
        if hex_type == "evaporator":
            T_sec_out = T_sec_in - effectiveness * (T_sec_in - T_ref)
            Q_actual = effectiveness * C_sec * (T_sec_in - T_ref)
        else:
            T_sec_out = T_sec_in + effectiveness * (T_ref - T_sec_in)
            Q_actual = effectiveness * C_sec * (T_ref - T_sec_in)
        
        # Pressure drops
        if hex_type == "evaporator":
            rho_ref = (ref_props["rho_liquid"] + ref_props["rho_vapor"]) / 2
        else:
            rho_ref = ref_props["rho_liquid"]
        
        v_ref = G_ref / rho_ref
        mu_ref = ref_props["mu_liquid"] if hex_type == "condenser" else ref_props["mu_vapor"]
        Re_ref = rho_ref * v_ref * tube_id / mu_ref if mu_ref > 0 else 0
        
        f_ref = 0.046 * Re_ref**-0.2 if Re_ref > 2300 else (64/Re_ref if Re_ref > 0 else 0.05)
        dp_tube = f_ref * (tube_length * n_passes / tube_id) * (rho_ref * v_ref**2 / 2)
        
        f_shell = 0.2 * Re_shell**-0.2 if Re_shell > 0 else 0.2
        dp_shell = f_shell * (tube_length / D_e) * n_baffles * (sec_props["rho"] * v_sec**2 / 2)
        
        # Calculate LMTD and required area
        if hex_type == "evaporator":
            dt1 = T_sec_in - T_ref
            dt2 = T_sec_out - T_ref_out
        else:
            dt1 = T_ref - T_sec_in
            dt2 = T_ref_out - T_sec_out
        
        if inputs["flow_arrangement"] == "counter":
            dt1, dt2 = dt1, dt2
        
        if dt1 <= 0 or dt2 <= 0 or abs(dt1 - dt2) < 1e-6:
            LMTD = min(dt1, dt2) if min(dt1, dt2) > 0 else 0
        else:
            LMTD = (dt1 - dt2) / math.log(dt1 / dt2)
        
        A_required = (Q_total * 1000) / (U * LMTD) if U > 0 and LMTD > 0 else 0
        
        # Calculate pitch ratio
        pitch_ratio = tube_pitch / tube_od if tube_od > 0 else 0
        
        # Store results
        self.results = {
            "heat_duty_kw": Q_total,
            "effectiveness": effectiveness,
            "ntu": NTU,
            "overall_u": U,
            "h_tube": h_ref,
            "h_shell": h_shell,
            "t_sec_out": T_sec_out,
            "t_ref_out": T_ref_out,
            "dp_tube_kpa": dp_tube / 1000,
            "dp_shell_kpa": dp_shell / 1000,
            "shell_diameter_m": shell_diameter,
            "tube_pitch_mm": tube_pitch * 1000,
            "pitch_ratio": pitch_ratio,
            "velocity_tube_ms": v_ref,
            "velocity_shell_ms": v_sec,
            "reynolds_tube": Re_ref,
            "reynolds_shell": Re_shell,
            "area_total_m2": A_total,
            "area_required_m2": A_required,
            "area_ratio": A_total / A_required if A_required > 0 else 0,
            "mass_flux": G_ref,
            "baffle_spacing_m": baffle_spacing,
            "design_status": "Adequate" if effectiveness >= 0.7 and A_total >= A_required else "Inadequate",
            "design_method": "ε-NTU Method",
            "tube_layout": tube_layout,
            "n_baffles": n_baffles
        }
        
        return self.results

def create_input_section():
    """Create input section in sidebar WITH TUBE PITCH INPUT"""
    st.sidebar.header("⚙️ Design Inputs")
    
    # Initialize session state
    if 'tube_thickness' not in st.session_state:
        st.session_state.tube_thickness = 1.0
    if 'tube_pitch' not in st.session_state:
        st.session_state.tube_pitch = 25.0
    
    inputs = {}
    
    # Heat exchanger type
    inputs["hex_type"] = st.sidebar.radio(
        "Heat Exchanger Type",
        ["Evaporator", "Condenser"]
    )
    
    # Fluid arrangement
    col1, col2 = st.sidebar.columns(2)
    with col1:
        inputs["tube_side"] = st.selectbox(
            "Tube Side Fluid",
            ["Refrigerant", "Water/Glycol"]
        )
    with col2:
        inputs["shell_side"] = st.selectbox(
            "Shell Side Fluid",
            ["Water/Glycol", "Refrigerant"]
        )
    
    st.sidebar.markdown("---")
    
    # Refrigerant parameters
    st.sidebar.subheader("Refrigerant Parameters")
    
    designer_temp = HeatExchangerDesign()
    inputs["refrigerant"] = st.sidebar.selectbox(
        "Refrigerant",
        list(designer_temp.REFRIGERANTS.keys())
    )
    
    inputs["m_dot_ref"] = st.sidebar.number_input(
        "Refrigerant Mass Flow (kg/hr)",
        min_value=10.0,
        max_value=10000.0,
        value=500.0,
        step=50.0
    )
    
    if inputs["hex_type"] == "Evaporator":
        inputs["T_ref"] = st.sidebar.number_input(
            "Evaporating Temperature (°C)",
            min_value=-50.0,
            max_value=20.0,
            value=5.0,
            step=1.0
        )
        inputs["delta_T_sh_sc"] = st.sidebar.number_input(
            "Superheating (ΔT in K)",
            min_value=0.0,
            max_value=20.0,
            value=5.0,
            step=0.5
        )
    else:
        inputs["T_ref"] = st.sidebar.number_input(
            "Condensing Temperature (°C)",
            min_value=20.0,
            max_value=80.0,
            value=45.0,
            step=1.0
        )
        inputs["delta_T_sh_sc"] = st.sidebar.number_input(
            "Subcooling (ΔT in K)",
            min_value=0.0,
            max_value=20.0,
            value=5.0,
            step=0.5
        )
    
    st.sidebar.markdown("---")
    
    # Secondary fluid parameters
    st.sidebar.subheader("Water/Glycol Parameters")
    
    secondary_type = st.sidebar.radio(
        "Secondary Fluid Type",
        ["Water", "Water + PE Glycol"]
    )
    
    if secondary_type == "Water + PE Glycol":
        inputs["glycol_percentage"] = st.sidebar.select_slider(
            "Glycol Percentage",
            options=[0, 10, 20, 30, 40, 50],
            value=20
        )
    else:
        inputs["glycol_percentage"] = 0
    
    inputs["secondary_fluid"] = secondary_type
    
    inputs["m_dot_sec"] = st.sidebar.number_input(
        "Flow Rate (L/hr)",
        min_value=100.0,
        max_value=100000.0,
        value=5000.0,
        step=500.0
    )
    
    inputs["T_sec_in"] = st.sidebar.number_input(
        "Inlet Temperature (°C)",
        min_value=0.0,
        max_value=80.0,
        value=25.0 if inputs["hex_type"] == "Condenser" else 12.0,
        step=1.0
    )
    
    inputs["flow_arrangement"] = st.sidebar.radio(
        "Flow Arrangement",
        ["Counter", "Parallel"]
    ).lower()
    
    st.sidebar.markdown("---")
    
    # Geometry parameters
    st.sidebar.subheader("Geometry Parameters")
    
    inputs["tube_size"] = st.sidebar.selectbox(
        "Tube Size",
        list(designer_temp.TUBE_SIZES.keys())
    )
    
    inputs["tube_material"] = st.sidebar.selectbox(
        "Tube Material",
        list(designer_temp.TUBE_MATERIALS.keys())
    )
    
    # Tube thickness with +/- buttons
    st.sidebar.markdown("**Tube Thickness**")
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        if st.button("−", key="thickness_minus"):
            st.session_state.tube_thickness = max(0.1, st.session_state.tube_thickness - 0.1)
    with col2:
        inputs["tube_thickness"] = st.number_input(
            "Thickness (mm)",
            min_value=0.1,
            max_value=5.0,
            value=st.session_state.tube_thickness,
            step=0.1,
            key="thickness_input",
            label_visibility="collapsed"
        )
    with col3:
        if st.button("＋", key="thickness_plus"):
            st.session_state.tube_thickness = min(5.0, st.session_state.tube_thickness + 0.1)
    
    # TUBE PITCH INPUT - NEW SECTION
    st.sidebar.markdown("**Tube Pitch**")
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        if st.button("−", key="pitch_minus"):
            st.session_state.tube_pitch = max(15.0, st.session_state.tube_pitch - 0.5)
    with col2:
        inputs["tube_pitch"] = st.number_input(
            "Pitch (mm)",
            min_value=15.0,
            max_value=100.0,
            value=st.session_state.tube_pitch,
            step=0.5,
            key="pitch_input",
            label_visibility="collapsed",
            help="Center-to-center distance between tubes"
        )
    with col3:
        if st.button("＋", key="pitch_plus"):
            st.session_state.tube_pitch = min(100.0, st.session_state.tube_pitch + 0.5)
    
    # Calculate and display pitch ratio
    tube_od = designer_temp.TUBE_SIZES[inputs["tube_size"]] * 1000  # mm
    pitch_ratio = inputs["tube_pitch"] / tube_od if tube_od > 0 else 0
    st.sidebar.caption(f"Pitch/OD ratio: {pitch_ratio:.2f}")
    
    if pitch_ratio < 1.25:
        st.sidebar.warning("⚠️ Pitch ratio < 1.25 may be too tight")
    elif pitch_ratio > 1.5:
        st.sidebar.info("ℹ️ Pitch ratio > 1.5 is good for cleaning")
    
    inputs["n_passes"] = st.sidebar.selectbox(
        "Tube Passes",
        [1, 2, 4, 6]
    )
    
    inputs["n_baffles"] = st.sidebar.slider(
        "Number of Baffles",
        min_value=1,
        max_value=20,
        value=5,
        step=1
    )
    
    inputs["n_tubes"] = st.sidebar.slider(
        "Number of Tubes",
        min_value=1,
        max_value=500,
        value=100,
        step=1
    )
    
    inputs["tube_length"] = st.sidebar.slider(
        "Tube Length (m)",
        min_value=0.5,
        max_value=10.0,
        value=3.0,
        step=0.5
    )
    
    inputs["tube_layout"] = st.sidebar.radio(
        "Tube Layout",
        ["Triangular", "Square"]
    )
    
    # Display typical pitch guidelines
    with st.sidebar.expander("💡 Pitch Guidelines"):
        st.markdown("""
        **Typical Tube Pitch Values:**
        
        | Layout | Minimum | Typical | Maximum |
        |--------|---------|---------|---------|
        | Triangular | 1.25×OD | 1.25-1.33×OD | 1.5×OD |
        | Square | 1.25×OD | 1.25-1.5×OD | 2.0×OD |
        
        Where OD = Tube Outer Diameter
        
        **Considerations:**
        - **Tight pitch** (<1.25×OD): More tubes, higher pressure drop
        - **Normal pitch** (1.25-1.33×OD): Standard design
        - **Wide pitch** (>1.5×OD): Easier cleaning, fewer tubes
        """)
    
    return inputs

def display_results(results: Dict, inputs: Dict):
    """Display calculation results"""
    
    st.markdown("## 📊 Design Results")
    st.info(f"**Design Method:** {results.get('design_method', 'ε-NTU Method')}")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Heat Duty", f"{results['heat_duty_kw']:.2f} kW")
    
    with col2:
        status_color = "normal" if results['design_status'] == "Adequate" else "inverse"
        st.metric("Design Status", results['design_status'], delta_color=status_color)
    
    with col3:
        st.metric("Effectiveness (ε)", f"{results['effectiveness']:.3f}")
    
    with col4:
        st.metric("NTU", f"{results['ntu']:.2f}")
    
    st.markdown("---")
    
    # Geometry details
    st.markdown("### 📐 Geometry Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Tube Bundle")
        st.write(f"**Tube OD:** {designer_temp.TUBE_SIZES[inputs['tube_size']]*1000:.1f} mm")
        st.write(f"**Tube ID:** {results.get('tube_id_mm', 0):.1f} mm")
        st.write(f"**Tube Pitch:** {results['tube_pitch_mm']:.1f} mm")
        st.write(f"**Pitch/OD Ratio:** {results['pitch_ratio']:.2f}")
        st.write(f"**Number of Tubes:** {inputs['n_tubes']}")
        st.write(f"**Tube Length:** {inputs['tube_length']} m")
        st.write(f"**Tube Layout:** {inputs['tube_layout']}")
    
    with col2:
        st.markdown("#### Shell & Baffles")
        st.write(f"**Shell Diameter:** {results['shell_diameter_m']*1000:.1f} mm")
        st.write(f"**Number of Baffles:** {inputs['n_baffles']}")
        st.write(f"**Baffle Spacing:** {results['baffle_spacing_m']:.2f} m")
        st.write(f"**Tube Passes:** {inputs['n_passes']}")
        st.write(f"**Tube Material:** {inputs['tube_material']}")
        st.write(f"**Flow Arrangement:** {inputs['flow_arrangement'].title()}")
    
    with col3:
        st.markdown("#### Thermal Performance")
        st.write(f"**Overall U:** {results['overall_u']:.1f} W/m²K")
        st.write(f"**Tube HTC:** {results['h_tube']:.1f} W/m²K")
        st.write(f"**Shell HTC:** {results['h_shell']:.1f} W/m²K")
        st.write(f"**Area Total:** {results['area_total_m2']:.2f} m²")
        st.write(f"**Area Required:** {results['area_required_m2']:.2f} m²")
        st.write(f"**Area Ratio:** {results['area_ratio']:.2f}")
    
    st.markdown("---")
    
    # Temperature results
    st.markdown("### 🌡️ Temperature Profile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Refrigerant")
        if inputs["hex_type"] == "Evaporator":
            st.write(f"**Inlet (Sat. Liquid):** {inputs['T_ref']:.1f} °C")
            st.write(f"**Outlet (Superheated):** {results['t_ref_out']:.1f} °C")
            st.write(f"**Superheat:** {inputs['delta_T_sh_sc']:.1f} K")
        else:
            st.write(f"**Inlet (Sat. Vapor):** {inputs['T_ref']:.1f} °C")
            st.write(f"**Outlet (Subcooled):** {results['t_ref_out']:.1f} °C")
            st.write(f"**Subcool:** {inputs['delta_T_sh_sc']:.1f} K")
    
    with col2:
        st.markdown("#### Secondary Fluid")
        st.write(f"**Inlet:** {inputs['T_sec_in']:.1f} °C")
        st.write(f"**Outlet:** {results['t_sec_out']:.1f} °C")
        st.write(f"**ΔT:** {abs(results['t_sec_out'] - inputs['T_sec_in']):.1f} K")
    
    st.markdown("---")
    
    # Flow parameters
    st.markdown("### 💧 Flow Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Tube Side")
        st.write(f"**Velocity:** {results['velocity_tube_ms']:.2f} m/s")
        st.write(f"**Mass Flux:** {results['mass_flux']:.1f} kg/m²s")
        st.write(f"**Reynolds:** {results['reynolds_tube']:.0f}")
        st.write(f"**Pressure Drop:** {results['dp_tube_kpa']:.2f} kPa")
    
    with col2:
        st.markdown("#### Shell Side")
        st.write(f"**Velocity:** {results['velocity_shell_ms']:.2f} m/s")
        st.write(f"**Reynolds:** {results['reynolds_shell']:.0f}")
        st.write(f"**Pressure Drop:** {results['dp_shell_kpa']:.2f} kPa")
    
    st.markdown("---")
    
    # ε-NTU Analysis
    st.markdown("### 📈 ε-NTU Analysis")
    
    fig = go.Figure()
    ntu_range = np.linspace(0, 5, 100)
    epsilon_cr0 = 1 - np.exp(-ntu_range)
    epsilon_cr05 = (1 - np.exp(-ntu_range * 0.5)) / (1 - 0.5 * np.exp(-ntu_range * 0.5))
    epsilon_cr1 = ntu_range / (1 + ntu_range)
    
    fig.add_trace(go.Scatter(x=ntu_range, y=epsilon_cr0, mode='lines',
                            name='C_r = 0 (Phase Change)', line=dict(color='blue', width=3)))
    fig.add_trace(go.Scatter(x=ntu_range, y=epsilon_cr05, mode='lines',
                            name='C_r = 0.5', line=dict(color='green', dash='dash')))
    fig.add_trace(go.Scatter(x=ntu_range, y=epsilon_cr1, mode='lines',
                            name='C_r = 1.0', line=dict(color='red', dash='dot')))
    
    if results['ntu'] <= 5:
        fig.add_trace(go.Scatter(
            x=[results['ntu']], y=[results['effectiveness']],
            mode='markers+text', name='Design Point',
            marker=dict(size=15, color='gold', symbol='star'),
            text=[f"NTU={results['ntu']:.2f}, ε={results['effectiveness']:.3f}"],
            textposition="top right"
        ))
    
    fig.update_layout(
        title='NTU-Effectiveness Diagram',
        xaxis_title='NTU',
        yaxis_title='Effectiveness (ε)',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Tube layout visualization
    st.markdown("### 🎯 Tube Layout Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create tube arrangement visualization
        fig = go.Figure()
        
        tube_od_mm = designer_temp.TUBE_SIZES[inputs['tube_size']] * 1000
        pitch_mm = results['tube_pitch_mm']
        n_tubes = min(inputs['n_tubes'], 50)  # Limit for visualization
        
        if inputs['tube_layout'] == "Triangular":
            # Triangular arrangement
            rows = int(math.sqrt(n_tubes / 0.866))
            for i in range(rows):
                for j in range(rows):
                    if i * rows + j < n_tubes:
                        x = j * pitch_mm
                        y = i * pitch_mm * 0.866  # sin(60°)
                        if i % 2 == 1:
                            x += pitch_mm / 2
                        fig.add_shape(
                            type="circle",
                            xref="x", yref="y",
                            x0=x - tube_od_mm/2, y0=y - tube_od_mm/2,
                            x1=x + tube_od_mm/2, y1=y + tube_od_mm/2,
                            line_color="blue",
                            fillcolor="lightblue",
                            opacity=0.7
                        )
        else:
            # Square arrangement
            rows = int(math.sqrt(n_tubes))
            for i in range(rows):
                for j in range(rows):
                    if i * rows + j < n_tubes:
                        x = j * pitch_mm
                        y = i * pitch_mm
                        fig.add_shape(
                            type="circle",
                            xref="x", yref="y",
                            x0=x - tube_od_mm/2, y0=y - tube_od_mm/2,
                            x1=x + tube_od_mm/2, y1=y + tube_od_mm/2,
                            line_color="blue",
                            fillcolor="lightblue",
                            opacity=0.7
                        )
        
        fig.update_layout(
            title=f"{inputs['tube_layout']} Layout",
            xaxis_title="Width (mm)",
            yaxis_title="Height (mm)",
            showlegend=False,
            width=400,
            height=400,
            plot_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Layout Parameters")
        st.write(f"**Layout Type:** {inputs['tube_layout']}")
        st.write(f"**Tube OD:** {tube_od_mm:.1f} mm")
        st.write(f"**Tube Pitch:** {pitch_mm:.1f} mm")
        st.write(f"**Clearance:** {pitch_mm - tube_od_mm:.1f} mm")
        st.write(f"**Pitch/OD Ratio:** {results['pitch_ratio']:.2f}")
        
        # Pitch recommendation
        if results['pitch_ratio'] < 1.25:
            st.error("⚠️ **Pitch too tight!** Minimum recommended is 1.25×OD")
        elif results['pitch_ratio'] < 1.33:
            st.success("✅ **Good pitch** - Standard design")
        elif results['pitch_ratio'] < 1.5:
            st.info("ℹ️ **Wide pitch** - Good for cleaning")
        else:
            st.warning("⚠️ **Very wide pitch** - May reduce tube count significantly")
        
        st.markdown("""
        **Tube Count Check:**
        - With current pitch, maximum tubes in shell: ~{:.0f}
        - You have specified: {} tubes
        """.format(
            (results['shell_diameter_m'] * 1000 / pitch_mm)**2 * 
            (0.866 if inputs['tube_layout'] == "Triangular" else 1.0),
            inputs['n_tubes']
        ))
    
    st.markdown("---")
    
    # Design recommendations
    st.markdown("### 💡 Design Recommendations")
    
    if results['effectiveness'] < 0.7:
        st.error(f"""
        **Low Effectiveness!** (ε = {results['effectiveness']:.3f})
        
        **To improve:**
        1. **Increase heat transfer area:**
           - Add more tubes (currently {inputs['n_tubes']})
           - Increase tube length (currently {inputs['tube_length']} m)
           - Reduce tube pitch to fit more tubes
        2. **Check tube pitch:** Current {results['pitch_ratio']:.2f}×OD
           - Consider tighter pitch (1.25×OD) for more tubes
        3. **Review velocities:** Tube: {results['velocity_tube_ms']:.2f} m/s, Shell: {results['velocity_shell_ms']:.2f} m/s
        """)
    elif results['effectiveness'] > 0.95:
        st.warning(f"""
        **Very High Effectiveness** (ε = {results['effectiveness']:.3f})
        - May be overdesigned
        - Consider cost optimization
        """)
    else:
        st.success(f"""
        **Good Design Effectiveness** (ε = {results['effectiveness']:.3f})
        - In optimal range (0.7-0.95)
        """)
    
    # Area check
    if results['area_ratio'] < 0.9:
        st.warning(f"""
        **Undersized Area!**
        - Available: {results['area_total_m2']:.2f} m²
        - Required: {results['area_required_m2']:.2f} m²
        - Ratio: {results['area_ratio']:.2f} (should be ≥ 1.0)
        """)
    elif results['area_ratio'] > 1.1:
        st.info(f"""
        **Oversized Area**
        - Ratio: {results['area_ratio']:.2f}
        - Consider cost optimization
        """)
    
    # Pressure drop checks
    if results['dp_tube_kpa'] > 100:
        st.warning(f"**High tube-side ΔP:** {results['dp_tube_kpa']:.1f} kPa")
    if results['dp_shell_kpa'] > 50:
        st.warning(f"**High shell-side ΔP:** {results['dp_shell_kpa']:.1f} kPa")
    
    # Pitch-specific recommendations
    if results['pitch_ratio'] < 1.25:
        st.error("""
        **Critical: Tube pitch is too tight!**
        - Minimum recommended pitch is 1.25×OD for manufacturing and cleaning
        - Current pitch: {:.2f}×OD
        - Increase pitch to at least 1.25×OD
        """.format(results['pitch_ratio']))
    elif results['pitch_ratio'] > 1.5:
        st.info("""
        **Wide pitch design**
        - Good for cleaning and maintenance
        - May require larger shell for same tube count
        - Consider if cleaning access is critical
        """)
    
    st.markdown("---")
    
    # Export results
    st.markdown("### 💾 Export Results")
    
    if st.button("📥 Download Engineering Report", key="download_report"):
        report_data = {
            "Parameter": [
                "Heat Exchanger Type", "Refrigerant", "Design Method",
                "Heat Duty (kW)", "Effectiveness (ε)", "NTU",
                "Overall U (W/m²K)", "Tube HTC (W/m²K)", "Shell HTC (W/m²K)",
                "Secondary Outlet Temp (°C)", "Refrigerant Outlet Temp (°C)",
                "Shell Diameter (mm)", "Tube Pitch (mm)", "Pitch/OD Ratio",
                "Tube Layout", "Number of Tubes", "Tube Length (m)",
                "Tube Passes", "Number of Baffles", "Baffle Spacing (m)",
                "Tube Side ΔP (kPa)", "Shell Side ΔP (kPa)",
                "Total Area (m²)", "Required Area (m²)", "Area Ratio",
                "Tube Velocity (m/s)", "Shell Velocity (m/s)",
                "Design Status"
            ],
            "Value": [
                inputs["hex_type"], inputs["refrigerant"], results.get('design_method', 'ε-NTU'),
                f"{results['heat_duty_kw']:.2f}", f"{results['effectiveness']:.3f}", f"{results['ntu']:.2f}",
                f"{results['overall_u']:.1f}", f"{results['h_tube']:.1f}", f"{results['h_shell']:.1f}",
                f"{results['t_sec_out']:.1f}", f"{results['t_ref_out']:.1f}",
                f"{results['shell_diameter_m']*1000:.1f}", f"{results['tube_pitch_mm']:.1f}", f"{results['pitch_ratio']:.2f}",
                inputs['tube_layout'], str(inputs['n_tubes']), f"{inputs['tube_length']}",
                str(inputs['n_passes']), str(inputs['n_baffles']), f"{results['baffle_spacing_m']:.3f}",
                f"{results['dp_tube_kpa']:.2f}", f"{results['dp_shell_kpa']:.2f}",
                f"{results['area_total_m2']:.2f}", f"{results['area_required_m2']:.2f}", f"{results['area_ratio']:.2f}",
                f"{results['velocity_tube_ms']:.2f}", f"{results['velocity_shell_ms']:.2f}",
                results['design_status']
            ]
        }
        
        df_report = pd.DataFrame(report_data)
        csv = df_report.to_csv(index=False)
        
        st.download_button(
            label="Download CSV Report",
            data=csv,
            file_name="heat_exchanger_design_report.csv",
            mime="text/csv",
            key="download_csv"
        )

# Main application
st.markdown("<h1 class='main-header'>🌡️ Shell & Tube Heat Exchanger Designer</h1>", unsafe_allow_html=True)
st.markdown("### Complete Design with Tube Pitch Input & ε-NTU Method")

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'inputs' not in st.session_state:
    st.session_state.inputs = None

# Initialize designer for tube sizes
designer_temp = HeatExchangerDesign()

# Create layout
col1, col2 = st.columns([3, 1])

with col2:
    inputs = create_input_section()
    
    if st.sidebar.button("🚀 Calculate Design", type="primary", use_container_width=True):
        with st.spinner("Performing engineering calculations..."):
            designer = HeatExchangerDesign()
            calc_inputs = inputs.copy()
            calc_inputs["hex_type"] = calc_inputs["hex_type"].lower()
            results = designer.design_heat_exchanger(calc_inputs)
            st.session_state.results = results
            st.session_state.inputs = inputs
            st.rerun()

with col1:
    if st.session_state.results is not None:
        display_results(st.session_state.results, st.session_state.inputs)
    else:
        st.markdown("""
        ## 🎯 Complete Heat Exchanger Design Tool
        
        ### **Now with Tube Pitch Input!**
        
        This tool now includes **tube pitch** (center-to-center distance) as a critical design parameter.
        
        ### **Key Features:**
        
        1. **Tube Pitch Configuration**
           - Input pitch in mm with +/- controls
           - Automatic pitch/OD ratio calculation
           - Visual tube layout display
           - Pitch recommendations based on TEMA standards
        
        2. **Engineering Design Methods**
           - ε-NTU method for phase-change applications
           - Two-phase heat transfer correlations
           - Proper shell diameter calculation using pitch
           - Pressure drop calculations
        
        3. **Complete Geometry**
           - Tube size, thickness, material
           - Shell diameter based on actual pitch
           - Baffle spacing and count
           - Flow arrangement (counter/parallel)
        
        ### **How to Use:**
        
        1. Configure all parameters in sidebar
        2. **Pay attention to tube pitch** - critical for shell diameter
        3. Click "Calculate Design"
        4. Review results and recommendations
        5. Optimize design based on feedback
        
        ### **Tube Pitch Guidelines:**
        
        | Application | Recommended Pitch/OD |
        |-------------|----------------------|
        | Standard triangular | 1.25-1.33 |
        | Cleanable triangular | 1.33-1.5 |
        | Square layout | 1.25-1.5 |
        | Heavy fouling | 1.5+ |
        
        ⚠️ **Minimum pitch:** 1.25×OD for manufacturing
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🔬 <strong>Complete Heat Exchanger Design Tool</strong> | With Tube Pitch Input & ε-NTU Method</p>
    <p>⚠️ Tube pitch is critical for accurate shell diameter calculation</p>
</div>
""", unsafe_allow_html=True)