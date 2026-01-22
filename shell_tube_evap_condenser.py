import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from typing import Tuple, Dict, List
import math

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
    """Class for heat exchanger design calculations"""
    
    # Physical properties database
    REFRIGERANTS = {
        "R134a": {
            "cp_vapor": 1.26,  # kJ/kg·K
            "cp_liquid": 1.43,  # kJ/kg·K
            "h_fg": 216.0,  # kJ/kg (latent heat)
            "rho_vapor": 25.0,  # kg/m³
            "rho_liquid": 1200.0,  # kg/m³
            "mu_vapor": 0.000013,  # Pa·s
            "mu_liquid": 0.0002,  # Pa·s
            "k_vapor": 0.013,  # W/m·K
            "k_liquid": 0.08,  # W/m·K
            "pr_vapor": 0.8,
            "pr_liquid": 3.0
        },
        "R404A": {
            "cp_vapor": 1.20,
            "cp_liquid": 1.50,
            "h_fg": 160.0,
            "rho_vapor": 40.0,
            "rho_liquid": 1100.0,
            "mu_vapor": 0.000012,
            "mu_liquid": 0.00018,
            "k_vapor": 0.012,
            "k_liquid": 0.075,
            "pr_vapor": 0.78,
            "pr_liquid": 2.8
        },
        "R407C": {
            "cp_vapor": 1.25,
            "cp_liquid": 1.45,
            "h_fg": 200.0,
            "rho_vapor": 30.0,
            "rho_liquid": 1150.0,
            "mu_vapor": 0.0000125,
            "mu_liquid": 0.00019,
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
            "mu_vapor": 0.000011,
            "mu_liquid": 0.00017,
            "k_vapor": 0.013,
            "k_liquid": 0.076,
            "pr_vapor": 0.81,
            "pr_liquid": 2.7
        },
        "Ammonia (R717)": {
            "cp_vapor": 2.20,
            "cp_liquid": 4.70,
            "h_fg": 1368.0,
            "rho_vapor": 5.0,
            "rho_liquid": 600.0,
            "mu_vapor": 0.000011,
            "mu_liquid": 0.00015,
            "k_vapor": 0.025,
            "k_liquid": 0.50,
            "pr_vapor": 0.90,
            "pr_liquid": 1.5
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
        
        # Temperature correction factors (simplified)
        temp_factor = 1 + 0.02 * (temperature - 20) / 50
        
        return {
            "cp": base_props["cp"] * temp_factor,  # kJ/kg·K
            "rho": base_props["rho"] / temp_factor,  # kg/m³
            "mu": base_props["mu"] / temp_factor,  # Pa·s
            "k": base_props["k"] * temp_factor,  # W/m·K
            "pr": base_props["pr"] / temp_factor
        }
    
    def calculate_reynolds_number(self, velocity: float, diameter: float, rho: float, mu: float) -> float:
        """Calculate Reynolds number"""
        return (rho * velocity * diameter) / mu
    
    def calculate_nusselt_tube(self, re: float, pr: float, flow_type: str = "turbulent") -> float:
        """Calculate Nusselt number for tube side"""
        if flow_type == "turbulent" and re > 2300:
            # Dittus-Boelter equation
            return 0.023 * (re ** 0.8) * (pr ** 0.4)
        else:
            # Laminar flow
            return 4.36  # Constant heat flux
    
    def calculate_nusselt_shell(self, re: float, pr: float) -> float:
        """Calculate Nusselt number for shell side (Bell-Delaware method simplified)"""
        if re > 100:
            return 0.36 * (re ** 0.55) * (pr ** (1/3))
        else:
            return 3.66  # Laminar flow
    
    def calculate_heat_transfer_coefficient(self, nu: float, k: float, diameter: float) -> float:
        """Calculate heat transfer coefficient"""
        return (nu * k) / diameter
    
    def calculate_pressure_drop_tube(self, rho: float, velocity: float, length: float, 
                                   diameter: float, friction_factor: float, passes: int) -> float:
        """Calculate pressure drop in tube side"""
        # Major losses
        dp_major = friction_factor * (length / diameter) * (rho * velocity ** 2) / 2
        
        # Minor losses (simplified)
        dp_minor = 2.5 * passes * (rho * velocity ** 2) / 2
        
        return dp_major + dp_minor  # Pa
    
    def calculate_pressure_drop_shell(self, rho: float, velocity: float, length: float, 
                                    shell_diameter: float, tube_diameter: float, 
                                    baffle_spacing: float, n_baffles: int) -> float:
        """Calculate pressure drop in shell side"""
        # Simplified shell side pressure drop
        dp = 0.5 * rho * velocity ** 2 * n_baffles
        return dp  # Pa
    
    def calculate_log_mean_temperature_difference(self, th_in: float, th_out: float, 
                                                 tc_in: float, tc_out: float, 
                                                 flow_arrangement: str) -> float:
        """Calculate LMTD with correction factor"""
        if flow_arrangement == "counter":
            delta_t1 = th_in - tc_out
            delta_t2 = th_out - tc_in
        else:  # parallel
            delta_t1 = th_in - tc_in
            delta_t2 = th_out - tc_out
        
        if delta_t1 == delta_t2:
            return delta_t1
        
        if delta_t1 <= 0 or delta_t2 <= 0:
            return 0
        
        return (delta_t1 - delta_t2) / math.log(delta_t1 / delta_t2)
    
    def calculate_overall_heat_transfer_coefficient(self, hi: float, ho: float, 
                                                  tube_k: float, tube_id: float, 
                                                  tube_od: float, fouling_factor: float = 0.0002) -> float:
        """Calculate overall heat transfer coefficient"""
        # U based on outer diameter
        r_i = 1 / (hi * (tube_id / tube_od))
        r_o = 1 / ho
        r_w = math.log(tube_od / tube_id) / (2 * math.pi * tube_k)
        r_f = fouling_factor * (tube_od / tube_id + 1)
        
        u_total = 1 / (r_i + r_o + r_w + r_f)
        return u_total
    
    def calculate_shell_diameter(self, tube_od: float, n_tubes: int, pitch_ratio: float = 1.25, 
                               tube_layout: str = "triangular") -> float:
        """Estimate shell diameter based on tube count and layout"""
        # Bundle diameter estimation (TEMA standards)
        if tube_layout == "triangular":
            k1 = 0.319
            n1 = 2.142
        else:  # square
            k1 = 0.215
            n1 = 2.207
        
        bundle_diameter = tube_od * (n_tubes / k1) ** (1 / n1)
        
        # Add clearance for shell
        shell_diameter = bundle_diameter * 1.1
        
        return shell_diameter
    
    def design_heat_exchanger(self, inputs: Dict) -> Dict:
        """Main design calculation function"""
        
        # Extract inputs
        hex_type = inputs["hex_type"]  # "evaporator" or "condenser"
        refrigerant = inputs["refrigerant"]
        m_dot_ref = inputs["m_dot_ref"] / 3600  # kg/hr to kg/s
        T_ref = inputs["T_ref"]  # Evap or cond temp
        delta_T_sh_sc = inputs["delta_T_sh_sc"]
        flow_arrangement = inputs["flow_arrangement"]
        
        # Secondary fluid properties
        secondary_fluid = inputs["secondary_fluid"]
        glycol_percentage = inputs["glycol_percentage"]
        m_dot_sec = inputs["m_dot_sec"] / 3600  # L/hr to L/s
        T_sec_in = inputs["T_sec_in"]
        
        # Tube parameters
        tube_size = inputs["tube_size"]
        tube_material = inputs["tube_material"]
        tube_thickness = inputs["tube_thickness"] / 1000  # mm to m
        n_passes = inputs["n_passes"]
        n_baffles = inputs["n_baffles"]
        n_tubes = inputs["n_tubes"]
        tube_length = inputs["tube_length"]
        tube_layout = inputs["tube_layout"]
        
        # Get refrigerant properties
        ref_props = self.REFRIGERANTS[refrigerant]
        
        # Calculate refrigerant temperatures
        if hex_type == "evaporator":
            T_ref_in = T_ref  # Evaporating temperature
            T_ref_out = T_ref + delta_T_sh_sc  # Superheated temperature
            # Use latent heat for evaporation plus sensible for superheat
            Q_total = m_dot_ref * (ref_props["h_fg"] + 
                                  ref_props["cp_vapor"] * delta_T_sh_sc)  # kW
        else:  # condenser
            T_ref_in = T_ref  # Condensing temperature
            T_ref_out = T_ref - delta_T_sh_sc  # Subcooled temperature
            Q_total = m_dot_ref * (ref_props["h_fg"] + 
                                  ref_props["cp_liquid"] * delta_T_sh_sc)  # kW
        
        # Calculate secondary fluid properties
        sec_props = self.calculate_water_glycol_properties(T_sec_in, glycol_percentage)
        
        # Convert volumetric flow to mass flow for secondary fluid
        m_dot_sec_kg = m_dot_sec * sec_props["rho"] / 1000  # kg/s (L/s to m³/s)
        
        # Calculate secondary fluid outlet temperature
        T_sec_out = T_sec_in + (Q_total * 1000) / (m_dot_sec_kg * sec_props["cp"] * 1000)
        
        # Check for temperature cross
        if hex_type == "evaporator":
            if T_sec_out < T_ref_out:
                st.warning("⚠️ Temperature cross detected! Secondary fluid outlet temperature is below refrigerant outlet temperature.")
        else:
            if T_sec_out > T_ref_out:
                st.warning("⚠️ Temperature cross detected! Secondary fluid outlet temperature is above refrigerant outlet temperature.")
        
        # Tube dimensions
        tube_od = self.TUBE_SIZES[tube_size]
        tube_id = tube_od - 2 * tube_thickness
        
        # Calculate shell diameter
        shell_diameter = self.calculate_shell_diameter(tube_od, n_tubes, 1.25, tube_layout)
        
        # Calculate flow areas
        tube_flow_area = (math.pi * tube_id ** 2 / 4) * n_tubes / n_passes
        shell_flow_area = (shell_diameter * tube_length / n_baffles) * (tube_od / (1.25 * tube_od))
        
        # Calculate velocities
        if hex_type == "evaporator":
            # Refrigerant in two-phase flow (use average density)
            rho_ref_avg = (ref_props["rho_vapor"] + ref_props["rho_liquid"]) / 2
            v_ref = m_dot_ref / (rho_ref_avg * tube_flow_area)
        else:
            # Refrigerant as liquid (condenser outlet)
            v_ref = m_dot_ref / (ref_props["rho_liquid"] * tube_flow_area)
        
        v_sec = m_dot_sec_kg / (sec_props["rho"] * shell_flow_area)
        
        # Calculate Reynolds numbers
        if hex_type == "evaporator":
            mu_ref = (ref_props["mu_vapor"] + ref_props["mu_liquid"]) / 2
        else:
            mu_ref = ref_props["mu_liquid"]
        
        re_ref = self.calculate_reynolds_number(v_ref, tube_id, 
                                               ref_props["rho_liquid"] if hex_type == "condenser" else rho_ref_avg, 
                                               mu_ref)
        re_sec = self.calculate_reynolds_number(v_sec, tube_od, sec_props["rho"], sec_props["mu"])
        
        # Calculate Nusselt numbers
        if hex_type == "evaporator":
            pr_ref = (ref_props["pr_vapor"] + ref_props["pr_liquid"]) / 2
        else:
            pr_ref = ref_props["pr_liquid"]
        
        nu_ref = self.calculate_nusselt_tube(re_ref, pr_ref, "turbulent" if re_ref > 2300 else "laminar")
        nu_sec = self.calculate_nusselt_shell(re_sec, sec_props["pr"])
        
        # Calculate heat transfer coefficients
        if hex_type == "evaporator":
            k_ref = (ref_props["k_vapor"] + ref_props["k_liquid"]) / 2
        else:
            k_ref = ref_props["k_liquid"]
        
        hi = self.calculate_heat_transfer_coefficient(nu_ref, k_ref, tube_id)
        ho = self.calculate_heat_transfer_coefficient(nu_sec, sec_props["k"], tube_od)
        
        # Get tube material conductivity
        tube_k = self.TUBE_MATERIALS[tube_material]["k"]
        
        # Calculate overall heat transfer coefficient
        u_overall = self.calculate_overall_heat_transfer_coefficient(hi, ho, tube_k, tube_id, tube_od)
        
        # Calculate LMTD
        if hex_type == "evaporator":
            lmtd = self.calculate_log_mean_temperature_difference(
                T_sec_in, T_sec_out, T_ref, T_ref_out, flow_arrangement
            )
        else:
            lmtd = self.calculate_log_mean_temperature_difference(
                T_ref_in, T_ref_out, T_sec_in, T_sec_out, flow_arrangement
            )
        
        # Calculate required heat transfer area
        a_required = (Q_total * 1000) / (u_overall * lmtd)  # m²
        
        # Calculate available heat transfer area
        a_available = math.pi * tube_od * tube_length * n_tubes
        
        # Calculate effectiveness
        effectiveness = min(a_available / a_required, 1.0) if a_required > 0 else 0
        
        # Calculate pressure drops
        # Friction factor (simplified)
        f_ref = 0.046 * (re_ref ** -0.2) if re_ref > 2300 else 64 / re_ref
        f_sec = 0.2 * (re_sec ** -0.2)
        
        dp_ref = self.calculate_pressure_drop_tube(
            ref_props["rho_liquid"] if hex_type == "condenser" else rho_ref_avg,
            v_ref, tube_length * n_passes, tube_id, f_ref, n_passes
        ) / 1000  # kPa
        
        dp_sec = self.calculate_pressure_drop_shell(
            sec_props["rho"], v_sec, tube_length, shell_diameter, tube_od,
            tube_length / (n_baffles + 1), n_baffles
        ) / 1000  # kPa
        
        # Store results
        self.results = {
            "heat_duty_kw": Q_total,
            "secondary_fluid_outlet_temp": T_sec_out,
            "refrigerant_outlet_temp": T_ref_out,
            "shell_diameter_m": shell_diameter,
            "tube_velocity_mps": v_ref,
            "shell_velocity_mps": v_sec,
            "tube_reynolds": re_ref,
            "shell_reynolds": re_sec,
            "tube_h_coeff": hi,
            "shell_h_coeff": ho,
            "overall_u": u_overall,
            "lmtd": lmtd,
            "required_area": a_required,
            "available_area": a_available,
            "area_effectiveness": effectiveness * 100,
            "tube_pressure_drop_kpa": dp_ref,
            "shell_pressure_drop_kpa": dp_sec,
            "design_status": "Adequate" if effectiveness >= 0.9 else "Inadequate"
        }
        
        return self.results

def create_input_section():
    """Create input section in sidebar"""
    st.sidebar.header("⚙️ Design Inputs")
    
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
    inputs["refrigerant"] = st.sidebar.selectbox(
        "Refrigerant",
        list(designer.REFRIGERANTS.keys())
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
        ["Counter Flow", "Parallel Flow"]
    )
    
    st.sidebar.markdown("---")
    
    # Geometry parameters
    st.sidebar.subheader("Geometry Parameters")
    
    inputs["tube_size"] = st.sidebar.selectbox(
        "Tube Size",
        list(designer.TUBE_SIZES.keys())
    )
    
    inputs["tube_material"] = st.sidebar.selectbox(
        "Tube Material",
        list(designer.TUBE_MATERIALS.keys())
    )
    
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        if st.button("−"):
            st.session_state.tube_thickness = max(0.5, st.session_state.get("tube_thickness", 1.0) - 0.1)
    with col2:
        inputs["tube_thickness"] = st.number_input(
            "Tube Thickness (mm)",
            min_value=0.5,
            max_value=3.0,
            value=st.session_state.get("tube_thickness", 1.0),
            step=0.1,
            key="thickness_input"
        )
    with col3:
        if st.button("＋"):
            st.session_state.tube_thickness = min(3.0, st.session_state.get("tube_thickness", 1.0) + 0.1)
    
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
    
    return inputs

def display_results(results: Dict, inputs: Dict):
    """Display calculation results"""
    
    st.markdown("## 📊 Design Results")
    
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
            "Area Effectiveness",
            f"{results['area_effectiveness']:.1f}%",
            delta_color="normal"
        )
    
    with col4:
        st.metric(
            "Overall U",
            f"{results['overall_u']:.1f} W/m²·K"
        )
    
    st.markdown("---")
    
    # Temperature results
    st.markdown("### 🌡️ Temperature Profile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Refrigerant")
        if inputs["hex_type"] == "Evaporator":
            st.write(f"**Inlet (Evaporating):** {inputs['T_ref']:.1f} °C")
            st.write(f"**Outlet (Superheated):** {results['refrigerant_outlet_temp']:.1f} °C")
            st.write(f"**Superheating:** {inputs['delta_T_sh_sc']:.1f} K")
        else:
            st.write(f"**Inlet (Condensing):** {inputs['T_ref']:.1f} °C")
            st.write(f"**Outlet (Subcooled):** {results['refrigerant_outlet_temp']:.1f} °C")
            st.write(f"**Subcooling:** {inputs['delta_T_sh_sc']:.1f} K")
    
    with col2:
        st.markdown("#### Secondary Fluid")
        st.write(f"**Inlet:** {inputs['T_sec_in']:.1f} °C")
        st.write(f"**Outlet:** {results['secondary_fluid_outlet_temp']:.1f} °C")
        st.write(f"**ΔT:** {results['secondary_fluid_outlet_temp'] - inputs['T_sec_in']:.1f} K")
    
    # Geometry results
    st.markdown("### 📐 Geometry Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Shell Side")
        st.write(f"**Diameter:** {results['shell_diameter_m']*1000:.1f} mm")
        st.write(f"**Velocity:** {results['shell_velocity_mps']:.2f} m/s")
        st.write(f"**Re:** {results['shell_reynolds']:.0f}")
        st.write(f"**h:** {results['shell_h_coeff']:.1f} W/m²·K")
        st.write(f"**ΔP:** {results['shell_pressure_drop_kpa']:.1f} kPa")
    
    with col2:
        st.markdown("#### Tube Side")
        st.write(f"**Velocity:** {results['tube_velocity_mps']:.2f} m/s")
        st.write(f"**Re:** {results['tube_reynolds']:.0f}")
        st.write(f"**h:** {results['tube_h_coeff']:.1f} W/m²·K")
        st.write(f"**ΔP:** {results['tube_pressure_drop_kpa']:.1f} kPa")
    
    with col3:
        st.markdown("#### Heat Transfer")
        st.write(f"**LMTD:** {results['lmtd']:.1f} K")
        st.write(f"**Required Area:** {results['required_area']:.2f} m²")
        st.write(f"**Available Area:** {results['available_area']:.2f} m²")
        st.write(f"**Area Ratio:** {results['available_area']/results['required_area']:.2f}" if results['required_area'] > 0 else "N/A")
    
    # Visualizations
    st.markdown("### 📈 Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Temperature profile chart
        if inputs["hex_type"] == "Evaporator":
            x = ["Secondary In", "Secondary Out", "Refrigerant In", "Refrigerant Out"]
            y = [inputs['T_sec_in'], results['secondary_fluid_outlet_temp'], 
                 inputs['T_ref'], results['refrigerant_outlet_temp']]
        else:
            x = ["Refrigerant In", "Refrigerant Out", "Secondary In", "Secondary Out"]
            y = [inputs['T_ref'], results['refrigerant_outlet_temp'],
                 inputs['T_sec_in'], results['secondary_fluid_outlet_temp']]
        
        fig = go.Figure(data=[
            go.Bar(x=x, y=y, marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ])
        fig.update_layout(
            title="Temperature Profile",
            yaxis_title="Temperature (°C)",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Heat transfer coefficients
        labels = ['Tube Side', 'Shell Side', 'Overall']
        values = [results['tube_h_coeff'], results['shell_h_coeff'], results['overall_u']]
        
        fig = go.Figure(data=[
            go.Bar(x=labels, y=values, marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ])
        fig.update_layout(
            title="Heat Transfer Coefficients",
            yaxis_title="h/U (W/m²·K)",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Design recommendations
    st.markdown("### 💡 Design Recommendations")
    
    if results['area_effectiveness'] < 90:
        st.warning(f"""
        **Design is undersized!** 
        - Available area is only {results['area_effectiveness']:.1f}% of required area
        - Consider increasing: 
          * Number of tubes (currently {inputs['n_tubes']})
          * Tube length (currently {inputs['tube_length']} m)
          * Tube diameter
        """)
    elif results['area_effectiveness'] > 110:
        st.info(f"""
        **Design is oversized!**
        - Available area is {results['area_effectiveness']:.1f}% of required area
        - Consider decreasing:
          * Number of tubes (currently {inputs['n_tubes']})
          * Tube length (currently {inputs['tube_length']} m)
        """)
    else:
        st.success(f"""
        **Design is adequate!**
        - Available area is {results['area_effectiveness']:.1f}% of required area
        - All parameters are within acceptable ranges
        """)
    
    # Pressure drop check
    if results['tube_pressure_drop_kpa'] > 100:
        st.warning(f"**High tube side pressure drop:** {results['tube_pressure_drop_kpa']:.1f} kPa. Consider larger tubes or fewer passes.")
    
    if results['shell_pressure_drop_kpa'] > 50:
        st.warning(f"**High shell side pressure drop:** {results['shell_pressure_drop_kpa']:.1f} kPa. Consider larger shell or fewer baffles.")
    
    # Export option
    st.markdown("---")
    st.markdown("### 💾 Export Results")
    
    if st.button("📥 Download Design Report"):
        # Create a DataFrame with results
        report_data = {
            "Parameter": [
                "Heat Exchanger Type", "Refrigerant", "Heat Duty (kW)",
                "Secondary Fluid Outlet Temp (°C)", "Shell Diameter (mm)",
                "Overall U (W/m²·K)", "LMTD (K)", "Required Area (m²)",
                "Available Area (m²)", "Area Effectiveness (%)",
                "Tube Side ΔP (kPa)", "Shell Side ΔP (kPa)", "Design Status"
            ],
            "Value": [
                inputs["hex_type"], inputs["refrigerant"], f"{results['heat_duty_kw']:.2f}",
                f"{results['secondary_fluid_outlet_temp']:.1f}", f"{results['shell_diameter_m']*1000:.1f}",
                f"{results['overall_u']:.1f}", f"{results['lmtd']:.1f}", f"{results['required_area']:.2f}",
                f"{results['available_area']:.2f}", f"{results['area_effectiveness']:.1f}",
                f"{results['tube_pressure_drop_kpa']:.1f}", f"{results['shell_pressure_drop_kpa']:.1f}",
                results['design_status']
            ]
        }
        
        df_report = pd.DataFrame(report_data)
        csv = df_report.to_csv(index=False)
        
        st.download_button(
            label="Download CSV Report",
            data=csv,
            file_name="heat_exchanger_design_report.csv",
            mime="text/csv"
        )

# Main application
st.markdown("<h1 class='main-header'>🌡️ Shell & Tube Heat Exchanger Designer</h1>", unsafe_allow_html=True)
st.markdown("### For Refrigerant Evaporators and Condensers")

# Initialize designer
designer = HeatExchangerDesign()

# Create layout
col1, col2 = st.columns([3, 1])

with col2:
    # Input section in sidebar
    inputs = create_input_section()
    
    # Calculate button
    if st.sidebar.button("🚀 Calculate Design", type="primary", use_container_width=True):
        with st.spinner("Calculating design..."):
            # Convert inputs for calculation
            calc_inputs = inputs.copy()
            calc_inputs["hex_type"] = calc_inputs["hex_type"].lower()
            calc_inputs["flow_arrangement"] = calc_inputs["flow_arrangement"].split()[0].lower()
            
            # Perform calculation
            results = designer.design_heat_exchanger(calc_inputs)
            
            # Store results in session state
            st.session_state.results = results
            st.session_state.inputs = inputs
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info("""
    This tool designs shell & tube heat exchangers for refrigeration applications.
    
    **Features:**
    - Evaporator & condenser design
    - Multiple refrigerants
    - Water/glycol mixtures
    - Custom geometry inputs
    - Performance analysis
    """)

with col1:
    # Display results if available
    if 'results' in st.session_state:
        display_results(st.session_state.results, st.session_state.inputs)
    else:
        # Show instructions
        st.markdown("""
        ## 📝 How to Use
        
        1. **Configure Inputs** in the sidebar
        2. **Select heat exchanger type** (Evaporator or Condenser)
        3. **Choose fluids** for tube and shell sides
        4. **Enter refrigerant parameters**: flow rate, temperature, superheat/subcool
        5. **Enter secondary fluid parameters**: flow rate, temperature, glycol %
        6. **Set geometry parameters**: tube size, material, passes, baffles, etc.
        7. **Click 'Calculate Design'** to run the analysis
        
        ### 📋 Assumptions & Limitations
        
        - Simplified heat transfer correlations
        - Constant fluid properties (temperature-averaged)
        - Clean tubes (no fouling considered)
        - Perfect insulation (no heat losses)
        - Steady-state operation
        
        ### 🔧 Engineering Methods
        
        - **Heat Transfer**: ε-NTU method with LMTD correction
        - **Tube Side**: Dittus-Boelter equation for turbulent flow
        - **Shell Side**: Simplified Bell-Delaware method
        - **Pressure Drop**: Darcy-Weisbach equation with minor losses
        - **Shell Diameter**: TEMA standards estimation
        
        ### ⚠️ Important Notes
        
        This tool provides **preliminary design estimates**. For final design:
        - Consult relevant standards (ASME, TEMA)
        - Perform detailed simulation
        - Consider safety factors
        - Verify with manufacturer data
        - Consult with qualified engineers
        """)
        
        # Quick start example
        if st.button("🚀 Load Example Design"):
            st.session_state.example_loaded = True
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🔬 Engineering Design Tool | For educational and preliminary design purposes</p>
    <p>⚠️ Always verify designs with detailed calculations and professional review</p>
</div>
""", unsafe_allow_html=True)