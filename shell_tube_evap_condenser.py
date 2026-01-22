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

# Custom CSS for better styling
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
        "Copper": {
            "k": 386,  # W/m·K
            "density": 8960,  # kg/m³
            "cost_factor": 1.0
        },
        "Cu-Ni 90/10": {
            "k": 40,  # W/m·K
            "density": 8940,
            "cost_factor": 1.8
        },
        "Steel": {
            "k": 50,  # W/m·K
            "density": 7850,
            "cost_factor": 0.6
        },
        "Aluminum Brass": {
            "k": 100,  # W/m·K
            "density": 8300,
            "cost_factor": 1.2
        }
    }
    
    # Tube sizes (inches to meters)
    TUBE_SIZES = {
        "1/4\"": 0.00635,  # m
        "3/8\"": 0.009525,  # m
        "1/2\"": 0.0127,  # m
        "5/8\"": 0.015875,  # m
        "3/4\"": 0.01905,  # m
        "1\"": 0.0254,  # m
        "1.25\"": 0.03175,  # m
        "1.5\"": 0.0381  # m
    }
    
    # Glycol properties by percentage
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
        
        # Temperature correction factors
        temp_factor = 1 + 0.02 * (temperature - 20) / 50
        
        return {
            "cp": base_props["cp"] * temp_factor * 1000,  # J/kg·K
            "rho": base_props["rho"] / temp_factor,  # kg/m³
            "mu": base_props["mu"] / temp_factor,  # Pa·s
            "k": base_props["k"] * temp_factor,  # W/m·K
            "pr": base_props["pr"] / temp_factor
        }
    
    def calculate_ntu_effectiveness(self, C_h: float, C_c: float, U: float, A: float, 
                                  flow_arrangement: str, hex_type: str) -> Tuple[float, float]:
        """
        Calculate NTU and effectiveness using ε-NTU method
        
        For evaporators/condensers: C_r = 0 (phase change)
        ε = 1 - exp(-NTU)
        """
        # Determine which fluid is changing phase
        if hex_type == "evaporator":
            # Refrigerant evaporating: C_ref → ∞
            C_min = C_c  # Secondary fluid capacity
            C_r = 0  # Phase change
        elif hex_type == "condenser":
            # Refrigerant condensing: C_ref → ∞
            C_min = C_c  # Secondary fluid capacity
            C_r = 0  # Phase change
        else:
            # Single-phase both sides
            C_min = min(C_h, C_c)
            C_max = max(C_h, C_c)
            C_r = C_min / C_max if C_max > 0 else 0
        
        # Calculate NTU
        NTU = U * A / C_min if C_min > 0 else 0
        
        # Calculate effectiveness
        if C_r == 0:
            # Phase change (evaporator/condenser)
            effectiveness = 1 - math.exp(-NTU)
        else:
            if flow_arrangement == "counter":
                effectiveness = (1 - math.exp(-NTU * (1 - C_r))) / (1 - C_r * math.exp(-NTU * (1 - C_r)))
            else:  # parallel
                effectiveness = (1 - math.exp(-NTU * (1 + C_r))) / (1 + C_r)
        
        return NTU, effectiveness
    
    def calculate_two_phase_htc(self, refrigerant: str, quality: float, G: float, 
                              D: float, hex_type: str = "evaporator") -> float:
        """
        Calculate two-phase heat transfer coefficient
        Simplified Shah correlation for boiling/condensation
        """
        props = self.REFRIGERANTS[refrigerant]
        
        # Calculate single-phase liquid HTC as baseline
        Re_l = G * D / props["mu_liquid"]
        Pr_l = props["pr_liquid"]
        
        if Re_l > 2300:
            Nu_l = 0.023 * Re_l**0.8 * Pr_l**0.4
        else:
            Nu_l = 4.36
        
        h_l = Nu_l * props["k_liquid"] / D
        
        # Simplified two-phase multiplier
        if hex_type == "evaporator":
            # Boiling
            if quality <= 0:
                return h_l
            elif quality >= 1:
                # All vapor
                Re_v = G * D / props["mu_vapor"]
                Pr_v = props["pr_vapor"]
                if Re_v > 2300:
                    Nu_v = 0.023 * Re_v**0.8 * Pr_v**0.4
                else:
                    Nu_v = 4.36
                return Nu_v * props["k_vapor"] / D
            else:
                # Two-phase boiling (simplified)
                return h_l * (1 + 10 * quality**0.8)
        else:
            # Condensation
            if quality >= 1:
                # All vapor
                Re_v = G * D / props["mu_vapor"]
                Pr_v = props["pr_vapor"]
                if Re_v > 2300:
                    Nu_v = 0.023 * Re_v**0.8 * Pr_v**0.4
                else:
                    Nu_v = 4.36
                return Nu_v * props["k_vapor"] / D
            elif quality <= 0:
                return h_l
            else:
                # Two-phase condensation (simplified)
                return h_l * (1 + 8 * (1 - quality)**0.8)
    
    def calculate_shell_diameter(self, tube_od: float, n_tubes: int, pitch_ratio: float = 1.25,
                               tube_layout: str = "triangular") -> float:
        """Calculate shell diameter based on tube count and layout"""
        # TEMA standards constants
        if tube_layout.lower() == "triangular":
            if pitch_ratio == 1.25:
                K1 = 0.319
                n1 = 2.142
            else:
                K1 = 0.249
                n1 = 2.207
        else:  # square
            if pitch_ratio == 1.25:
                K1 = 0.215
                n1 = 2.207
            else:
                K1 = 0.156
                n1 = 2.291
        
        # Bundle diameter
        bundle_diameter = tube_od * (n_tubes / K1) ** (1 / n1)
        
        # Add clearance
        shell_diameter = bundle_diameter + 0.025  # 25mm clearance
        
        return max(shell_diameter, 0.1)  # Minimum 100mm
    
    def calculate_overall_u(self, h_i: float, h_o: float, tube_k: float,
                          tube_id: float, tube_od: float) -> float:
        """Calculate overall heat transfer coefficient"""
        # Thermal resistances (based on outer area)
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
        
        # Geometry
        tube_size = inputs["tube_size"]
        tube_material = inputs["tube_material"]
        tube_thickness = inputs["tube_thickness"] / 1000  # m
        n_passes = inputs["n_passes"]
        n_baffles = inputs["n_baffles"]
        n_tubes = inputs["n_tubes"]
        tube_length = inputs["tube_length"]
        tube_layout = inputs["tube_layout"]
        
        # Get properties
        ref_props = self.REFRIGERANTS[refrigerant]
        sec_props = self.calculate_water_glycol_properties(T_sec_in, glycol_percent)
        
        # Convert secondary flow to kg/s
        m_dot_sec_kg = m_dot_sec_L * sec_props["rho"] / 1000
        
        # Calculate heat duty
        if hex_type == "evaporator":
            # Latent heat + superheat
            Q_total = m_dot_ref * (ref_props["h_fg"] + ref_props["cp_vapor"] * delta_T)
            T_ref_out = T_ref + delta_T
        else:  # condenser
            # Latent heat + subcool
            Q_total = m_dot_ref * (ref_props["h_fg"] + ref_props["cp_liquid"] * delta_T)
            T_ref_out = T_ref - delta_T
        
        # Tube dimensions
        tube_od = self.TUBE_SIZES[tube_size]
        tube_id = tube_od - 2 * tube_thickness
        if tube_id <= 0:
            tube_id = tube_od * 0.8  # Fallback
        
        # Calculate shell diameter
        shell_diameter = self.calculate_shell_diameter(tube_od, n_tubes, 1.25, tube_layout)
        
        # Flow areas
        tube_flow_area = (math.pi * tube_id**2 / 4) * n_tubes / n_passes
        shell_flow_area = (shell_diameter * tube_length / (n_baffles + 1)) * 0.3  # Simplified
        
        # Mass fluxes
        G_ref = m_dot_ref / tube_flow_area if tube_flow_area > 0 else 0
        v_sec = m_dot_sec_kg / (sec_props["rho"] * shell_flow_area) if shell_flow_area > 0 else 0
        
        # Heat transfer coefficients
        if hex_type == "evaporator":
            # Two-phase evaporation
            h_ref = self.calculate_two_phase_htc(refrigerant, 0.5, G_ref, tube_id, "evaporator")
        else:
            # Two-phase condensation
            h_ref = self.calculate_two_phase_htc(refrigerant, 0.5, G_ref, tube_id, "condenser")
        
        # Shell-side HTC
        D_e = 4 * shell_flow_area / (math.pi * tube_od * n_tubes) if n_tubes > 0 else tube_od
        Re_shell = sec_props["rho"] * v_sec * D_e / sec_props["mu"]
        
        if Re_shell > 100:
            Nu_shell = 0.36 * Re_shell**0.55 * sec_props["pr"]**(1/3)
        else:
            Nu_shell = 3.66
        
        h_shell = Nu_shell * sec_props["k"] / D_e
        
        # Overall U
        tube_k = self.TUBE_MATERIALS[tube_material]["k"]
        U = self.calculate_overall_u(h_ref, h_shell, tube_k, tube_id, tube_od)
        
        # Capacity rates
        C_sec = m_dot_sec_kg * sec_props["cp"]  # W/K
        C_ref_inf = 1e10  # Approximate infinite for phase change
        
        # Total area
        A_total = math.pi * tube_od * tube_length * n_tubes
        
        # Calculate NTU and effectiveness using ε-NTU method
        if hex_type == "evaporator":
            NTU, effectiveness = self.calculate_ntu_effectiveness(
                C_ref_inf, C_sec, U, A_total, inputs["flow_arrangement"], hex_type
            )
        else:
            NTU, effectiveness = self.calculate_ntu_effectiveness(
                C_sec, C_ref_inf, U, A_total, inputs["flow_arrangement"], hex_type
            )
        
        # Calculate outlet temperatures
        if hex_type == "evaporator":
            T_sec_out = T_sec_in - effectiveness * (T_sec_in - T_ref)
            Q_actual = effectiveness * C_sec * (T_sec_in - T_ref)
        else:
            T_sec_out = T_sec_in + effectiveness * (T_ref - T_sec_in)
            Q_actual = effectiveness * C_sec * (T_ref - T_sec_in)
        
        # Pressure drops (simplified)
        if hex_type == "evaporator":
            rho_ref = (ref_props["rho_liquid"] + ref_props["rho_vapor"]) / 2
        else:
            rho_ref = ref_props["rho_liquid"]
        
        v_ref = G_ref / rho_ref
        Re_ref = rho_ref * v_ref * tube_id / (ref_props["mu_liquid"] if hex_type == "condenser" else ref_props["mu_vapor"])
        
        if Re_ref > 2300:
            f_ref = 0.046 * Re_ref**-0.2
        else:
            f_ref = 64 / Re_ref if Re_ref > 0 else 0.05
        
        dp_tube = f_ref * (tube_length * n_passes / tube_id) * (rho_ref * v_ref**2 / 2)
        
        if Re_shell > 0:
            f_shell = 0.2 * Re_shell**-0.2
        else:
            f_shell = 0.2
        
        dp_shell = f_shell * (tube_length / D_e) * n_baffles * (sec_props["rho"] * v_sec**2 / 2)
        
        # Calculate required area based on heat duty
        if hex_type == "evaporator":
            dt1 = T_sec_in - T_ref
            dt2 = T_sec_out - T_ref_out
        else:
            dt1 = T_ref - T_sec_in
            dt2 = T_ref_out - T_sec_out
        
        if inputs["flow_arrangement"] == "counter":
            dt1, dt2 = dt1, dt2
        else:
            dt1 = T_sec_in - T_ref if hex_type == "evaporator" else T_ref - T_sec_in
            dt2 = T_sec_out - T_ref_out if hex_type == "evaporator" else T_ref_out - T_sec_out
        
        if dt1 <= 0 or dt2 <= 0 or abs(dt1 - dt2) < 1e-6:
            LMTD = min(dt1, dt2) if min(dt1, dt2) > 0 else 0
        else:
            LMTD = (dt1 - dt2) / math.log(dt1 / dt2)
        
        A_required = (Q_total * 1000) / (U * LMTD) if U > 0 and LMTD > 0 else 0
        
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
            "velocity_tube_ms": v_ref,
            "velocity_shell_ms": v_sec,
            "reynolds_tube": Re_ref,
            "reynolds_shell": Re_shell,
            "area_total_m2": A_total,
            "area_required_m2": A_required,
            "area_ratio": A_total / A_required if A_required > 0 else 0,
            "mass_flux": G_ref,
            "design_status": "Adequate" if effectiveness >= 0.7 and A_total >= A_required else "Inadequate",
            "design_method": "ε-NTU Method"
        }
        
        return self.results

def create_input_section():
    """Create input section in sidebar"""
    st.sidebar.header("⚙️ Design Inputs")
    
    # Initialize session state for thickness
    if 'tube_thickness' not in st.session_state:
        st.session_state.tube_thickness = 1.0
    
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
    
    # Initialize designer for refrigerant list
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
    
    # Initialize for tube sizes
    inputs["tube_size"] = st.sidebar.selectbox(
        "Tube Size",
        list(designer_temp.TUBE_SIZES.keys())
    )
    
    inputs["tube_material"] = st.sidebar.selectbox(
        "Tube Material",
        list(designer_temp.TUBE_MATERIALS.keys())
    )
    
    # Tube thickness with +/- buttons
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        if st.button("−", key="thickness_minus"):
            st.session_state.tube_thickness = max(0.1, st.session_state.tube_thickness - 0.1)
    with col2:
        inputs["tube_thickness"] = st.number_input(
            "Tube Thickness (mm)",
            min_value=0.1,
            max_value=5.0,
            value=st.session_state.tube_thickness,
            step=0.1,
            key="thickness_input"
        )
    with col3:
        if st.button("＋", key="thickness_plus"):
            st.session_state.tube_thickness = min(5.0, st.session_state.tube_thickness + 0.1)
    
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
    ).lower()
    
    return inputs

def display_results(results: Dict, inputs: Dict):
    """Display calculation results"""
    
    st.markdown("## 📊 Design Results")
    st.info(f"**Design Method:** {results.get('design_method', 'ε-NTU Method')}")
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Heat Duty",
            f"{results['heat_duty_kw']:.2f} kW",
            delta_color="normal"
        )
    
    with col2:
        status_color = "normal" if results['design_status'] == "Adequate" else "inverse"
        st.metric(
            "Design Status",
            results['design_status'],
            delta_color=status_color
        )
    
    with col3:
        st.metric(
            "Effectiveness (ε)",
            f"{results['effectiveness']:.3f}",
            help="ε = Q_actual / Q_max"
        )
    
    with col4:
        st.metric(
            "NTU",
            f"{results['ntu']:.2f}",
            help="NTU = UA / C_min"
        )
    
    st.markdown("---")
    
    # Temperature results
    st.markdown("### 🌡️ Temperature Profile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Refrigerant")
        if inputs["hex_type"] == "Evaporator":
            st.write(f"**Inlet (Saturated Liquid):** {inputs['T_ref']:.1f} °C")
            st.write(f"**Outlet (Superheated Vapor):** {results['t_ref_out']:.1f} °C")
            st.write(f"**Superheating:** {inputs['delta_T_sh_sc']:.1f} K")
        else:
            st.write(f"**Inlet (Saturated Vapor):** {inputs['T_ref']:.1f} °C")
            st.write(f"**Outlet (Subcooled Liquid):** {results['t_ref_out']:.1f} °C")
            st.write(f"**Subcooling:** {inputs['delta_T_sh_sc']:.1f} K")
    
    with col2:
        st.markdown("#### Secondary Fluid")
        st.write(f"**Inlet:** {inputs['T_sec_in']:.1f} °C")
        st.write(f"**Outlet:** {results['t_sec_out']:.1f} °C")
        st.write(f"**Temperature Change:** {abs(results['t_sec_out'] - inputs['T_sec_in']):.1f} K")
    
    # Heat transfer details
    st.markdown("### 🔥 Heat Transfer Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Tube Side")
        st.write(f"**HTC:** {results['h_tube']:.1f} W/m²K")
        st.write(f"**Velocity:** {results['velocity_tube_ms']:.2f} m/s")
        st.write(f"**Reynolds:** {results['reynolds_tube']:.0f}")
        st.write(f"**Mass Flux:** {results['mass_flux']:.1f} kg/m²s")
    
    with col2:
        st.markdown("#### Shell Side")
        st.write(f"**HTC:** {results['h_shell']:.1f} W/m²K")
        st.write(f"**Velocity:** {results['velocity_shell_ms']:.2f} m/s")
        st.write(f"**Reynolds:** {results['reynolds_shell']:.0f}")
        st.write(f"**Shell Diameter:** {results['shell_diameter_m']*1000:.1f} mm")
    
    with col3:
        st.markdown("#### Overall")
        st.write(f"**Overall U:** {results['overall_u']:.1f} W/m²K")
        st.write(f"**Tube ΔP:** {results['dp_tube_kpa']:.2f} kPa")
        st.write(f"**Shell ΔP:** {results['dp_shell_kpa']:.2f} kPa")
        st.write(f"**Area Total:** {results['area_total_m2']:.2f} m²")
        st.write(f"**Area Required:** {results['area_required_m2']:.2f} m²")
    
    st.markdown("---")
    
    # ε-NTU Analysis
    st.markdown("### 📈 ε-NTU Analysis")
    
    # Create NTU-effectiveness chart
    fig = go.Figure()
    
    # Generate curves for different C_r values
    ntu_range = np.linspace(0, 5, 100)
    
    # C_r = 0 (phase change) - all flow arrangements
    epsilon_cr0 = 1 - np.exp(-ntu_range)
    
    # C_r = 0.5
    epsilon_cr05_counter = (1 - np.exp(-ntu_range * 0.5)) / (1 - 0.5 * np.exp(-ntu_range * 0.5))
    
    # C_r = 1.0
    epsilon_cr1_counter = ntu_range / (1 + ntu_range)
    epsilon_cr1_parallel = (1 - np.exp(-2 * ntu_range)) / 2
    
    fig.add_trace(go.Scatter(x=ntu_range, y=epsilon_cr0, mode='lines',
                            name='C_r = 0 (Phase Change)', line=dict(color='blue', width=3)))
    fig.add_trace(go.Scatter(x=ntu_range, y=epsilon_cr05_counter, mode='lines',
                            name='C_r = 0.5, Counter', line=dict(color='green', dash='dash')))
    fig.add_trace(go.Scatter(x=ntu_range, y=epsilon_cr1_counter, mode='lines',
                            name='C_r = 1.0, Counter', line=dict(color='red', dash='dot')))
    fig.add_trace(go.Scatter(x=ntu_range, y=epsilon_cr1_parallel, mode='lines',
                            name='C_r = 1.0, Parallel', line=dict(color='orange', dash='dot')))
    
    # Add design point
    if results['ntu'] <= 5:
        fig.add_trace(go.Scatter(
            x=[results['ntu']],
            y=[results['effectiveness']],
            mode='markers+text',
            name='Design Point',
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
    
    # Design recommendations
    st.markdown("### 💡 Design Recommendations")
    
    if results['effectiveness'] < 0.7:
        st.error(f"""
        **Low Effectiveness Design!** (ε = {results['effectiveness']:.3f})
        
        Effectiveness below 0.7 indicates poor heat exchanger performance.
        
        **Recommended Actions:**
        1. **Increase heat transfer area:**
           - Add more tubes (currently {inputs['n_tubes']})
           - Increase tube length (currently {inputs['tube_length']} m)
           - Use smaller tube pitch to fit more tubes
        2. **Improve heat transfer coefficients:**
           - Increase fluid velocities
           - Consider different tube material
           - Optimize baffle spacing
        3. **Review operating conditions:**
           - Check temperature approach
           - Verify flow rates
           - Consider different refrigerant
        """)
    elif results['effectiveness'] > 0.95:
        st.warning(f"""
        **Very High Effectiveness** (ε = {results['effectiveness']:.3f})
        
        While high effectiveness is good, values above 0.95 often indicate overdesign.
        
        **Consider:**
        - Reducing heat transfer area for cost savings
        - Using smaller tubes or fewer passes
        - Checking if such high effectiveness is truly required
        """)
    else:
        st.success(f"""
        **Good Design Effectiveness** (ε = {results['effectiveness']:.3f})
        
        Effectiveness in optimal range (0.7-0.95).
        
        **Design is balanced between performance and cost.**
        """)
    
    # Area analysis
    area_ratio = results['area_total_m2'] / results['area_required_m2'] if results['area_required_m2'] > 0 else 0
    if area_ratio < 0.9:
        st.warning(f"""
        **Undersized Heat Transfer Area!**
        - Available: {results['area_total_m2']:.2f} m²
        - Required: {results['area_required_m2']:.2f} m²
        - Ratio: {area_ratio:.2f} (should be ≥ 1.0)
        """)
    elif area_ratio > 1.1:
        st.info(f"""
        **Oversized Heat Transfer Area**
        - Available: {results['area_total_m2']:.2f} m²
        - Required: {results['area_required_m2']:.2f} m²
        - Ratio: {area_ratio:.2f}
        - Consider reducing area for cost savings
        """)
    
    # Pressure drop checks
    if results['dp_tube_kpa'] > 100:
        st.warning(f"**High tube-side pressure drop:** {results['dp_tube_kpa']:.1f} kPa. Consider larger tubes or fewer passes.")
    
    if results['dp_shell_kpa'] > 50:
        st.warning(f"**High shell-side pressure drop:** {results['dp_shell_kpa']:.1f} kPa. Consider larger shell diameter or reduce baffles.")
    
    st.markdown("---")
    
    # Export results
    st.markdown("### 💾 Export Results")
    
    if st.button("📥 Download Engineering Report", key="download_report"):
        report_data = {
            "Parameter": [
                "Heat Exchanger Type",
                "Refrigerant",
                "Design Method",
                "Heat Duty (kW)",
                "Effectiveness (ε)",
                "NTU",
                "Overall U (W/m²K)",
                "Tube Side HTC (W/m²K)",
                "Shell Side HTC (W/m²K)",
                "Secondary Outlet Temp (°C)",
                "Refrigerant Outlet Temp (°C)",
                "Shell Diameter (mm)",
                "Tube Side ΔP (kPa)",
                "Shell Side ΔP (kPa)",
                "Total Area (m²)",
                "Required Area (m²)",
                "Area Ratio",
                "Design Status"
            ],
            "Value": [
                inputs["hex_type"],
                inputs["refrigerant"],
                results.get('design_method', 'ε-NTU Method'),
                f"{results['heat_duty_kw']:.2f}",
                f"{results['effectiveness']:.3f}",
                f"{results['ntu']:.2f}",
                f"{results['overall_u']:.1f}",
                f"{results['h_tube']:.1f}",
                f"{results['h_shell']:.1f}",
                f"{results['t_sec_out']:.1f}",
                f"{results['t_ref_out']:.1f}",
                f"{results['shell_diameter_m']*1000:.1f}",
                f"{results['dp_tube_kpa']:.2f}",
                f"{results['dp_shell_kpa']:.2f}",
                f"{results['area_total_m2']:.2f}",
                f"{results['area_required_m2']:.2f}",
                f"{area_ratio:.2f}",
                results['design_status']
            ]
        }
        
        df_report = pd.DataFrame(report_data)
        csv = df_report.to_csv(index=False)
        
        st.download_button(
            label="Download CSV Report",
            data=csv,
            file_name="heat_exchanger_engineering_report.csv",
            mime="text/csv",
            key="download_csv"
        )

# Main application
st.markdown("<h1 class='main-header'>🌡️ Shell & Tube Heat Exchanger Designer</h1>", unsafe_allow_html=True)
st.markdown("### Using ε-NTU Method for Evaporators and Condensers")

# Initialize session state for results
if 'results' not in st.session_state:
    st.session_state.results = None
if 'inputs' not in st.session_state:
    st.session_state.inputs = None

# Create layout
col1, col2 = st.columns([3, 1])

with col2:
    # Input section in sidebar
    inputs = create_input_section()
    
    # Calculate button
    if st.sidebar.button("🚀 Calculate Design", type="primary", use_container_width=True):
        with st.spinner("Performing engineering calculations..."):
            designer = HeatExchangerDesign()
            
            # Convert inputs for calculation
            calc_inputs = inputs.copy()
            calc_inputs["hex_type"] = calc_inputs["hex_type"].lower()
            
            # Perform calculation
            results = designer.design_heat_exchanger(calc_inputs)
            
            # Store results in session state
            st.session_state.results = results
            st.session_state.inputs = inputs
            
            st.rerun()

with col1:
    # Display results if available
    if st.session_state.results is not None:
        display_results(st.session_state.results, st.session_state.inputs)
    else:
        # Show instructions and methodology
        st.markdown("""
        ## 🎯 Engineering Design Methodology
        
        This tool uses the **ε-NTU (Effectiveness - Number of Transfer Units) method** 
        for accurate heat exchanger design, especially for phase-change applications.
        
        ### **Key Engineering Methods:**
        
        #### **1. ε-NTU Method**
        ```
        NTU = UA / C_min
        ε = Q_actual / Q_max
        ```
        
        **For evaporators/condensers (C_r = 0):**
        ```
        ε = 1 - exp(-NTU)
        ```
        
        #### **2. Two-Phase Heat Transfer**
        - **Evaporators**: Simplified Shah correlation for boiling
        - **Condensers**: Simplified correlation for condensation
        - Accounts for vapor quality changes
        
        #### **3. Thermal Resistance Network**
        ```
        1/U = 1/h_i × (A_o/A_i) + 1/h_o + R_wall + R_fouling
        ```
        
        ### **Design Process:**
        
        1. **Energy Balance**: Calculate heat duty from refrigerant properties
        2. **Heat Transfer Coefficients**: Calculate h for both sides
        3. **Overall U**: Include all thermal resistances
        4. **ε-NTU Analysis**: Determine effectiveness and NTU
        5. **Outlet Temperatures**: Calculate from effectiveness
        6. **Pressure Drop**: Estimate using fluid mechanics
        7. **Design Assessment**: Check constraints
        
        ### **To Get Started:**
        
        1. Configure all parameters in the sidebar
        2. Click "Calculate Design"
        3. Review detailed engineering analysis
        4. Use recommendations to optimize design
        
        ### **Design Guidelines:**
        
        - **Effectiveness (ε)**: 0.7-0.95 is optimal
        - **NTU**: Typically 1-3 for good designs
        - **Pressure Drop**: < 100 kPa tube side, < 50 kPa shell side
        - **Area Ratio**: Available/Required ≥ 1.0
        
        ⚠️ **Note**: This is for preliminary design. Final design requires:
        - Detailed property data (REFPROP)
        - Full Bell-Delaware method for shell side
        - Mechanical design calculations
        - Code compliance verification
        """)
        
        # Quick example button
        if st.button("📋 Show Example Calculation", key="example_button"):
            st.info("Configure parameters in sidebar and click 'Calculate Design' to see example results.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🔬 <strong>Engineering Design Tool</strong> | ε-NTU Method with Two-Phase Correlations</p>
    <p>⚠️ For preliminary design only | Consult engineering standards for final design</p>
</div>
""", unsafe_allow_html=True)