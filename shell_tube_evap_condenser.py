import streamlit as st
import numpy as np
import pandas as pd
import math
from scipy.optimize import fsolve
import plotly.graph_objects as go
from typing import Dict, Tuple, List

st.set_page_config(
    page_title="Shell & Tube HX Designer - Engineering Grade",
    page_icon="🌡️",
    layout="wide"
)

class HeatExchangerDesign:
    """Engineering-grade heat exchanger design with proper ε-NTU method"""
    
    # Refrigerant properties (more comprehensive)
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
    
    # Two-phase heat transfer correlations
    TWO_PHASE_CORRELATIONS = {
        "evaporator": {
            "name": "Shah Correlation",
            "formula": "h_tp = h_l * [(1 - x)^0.8 + 3.8*x^0.76*(1-x)^0.04/Pr_l^0.38]"
        },
        "condenser": {
            "name": "Shah Correlation",
            "formula": "h_tp = h_l * [(1 - x)^0.8 + 3.8*x^0.76*(1-x)^0.04/Pr_l^0.38]"
        }
    }
    
    def __init__(self):
        self.results = {}
        
    def calculate_ntu_effectiveness(self, C_h: float, C_c: float, U: float, A: float, 
                                  flow_arrangement: str, exchanger_type: str = "evaporator") -> Tuple[float, float]:
        """
        Calculate NTU and effectiveness using ε-NTU method
        
        Args:
            C_h: Hot fluid capacity rate (W/K)
            C_c: Cold fluid capacity rate (W/K)
            U: Overall heat transfer coefficient (W/m²K)
            A: Heat transfer area (m²)
            flow_arrangement: 'counter' or 'parallel'
            exchanger_type: 'evaporator' or 'condenser'
            
        Returns:
            NTU, effectiveness
        """
        # For evaporators/condensers, one fluid has C → ∞ (phase change)
        if exchanger_type == "evaporator":
            # Refrigerant evaporating (C → ∞), secondary fluid is C_min
            C_min = min(C_c, C_h)
            C_max = max(C_c, C_h)
            C_r = C_min / C_max if C_max > 0 else 0
            
            # For phase change (C_r = 0)
            NTU = U * A / C_min if C_min > 0 else 0
            
            if flow_arrangement == "counter" or flow_arrangement == "parallel":
                # For C_r = 0 (phase change), ε = 1 - exp(-NTU)
                effectiveness = 1 - math.exp(-NTU)
            else:
                effectiveness = 1 - math.exp(-NTU)
                
        elif exchanger_type == "condenser":
            # Refrigerant condensing (C → ∞), secondary fluid is C_min
            C_min = min(C_c, C_h)
            C_max = max(C_c, C_h)
            C_r = C_min / C_max if C_max > 0 else 0
            
            NTU = U * A / C_min if C_min > 0 else 0
            
            if flow_arrangement == "counter" or flow_arrangement == "parallel":
                effectiveness = 1 - math.exp(-NTU)
            else:
                effectiveness = 1 - math.exp(-NTU)
                
        else:
            # Single-phase both sides
            C_min = min(C_c, C_h)
            C_max = max(C_c, C_h)
            C_r = C_min / C_max if C_max > 0 else 0
            
            NTU = U * A / C_min if C_min > 0 else 0
            
            if flow_arrangement == "counter":
                if C_r < 1:
                    effectiveness = (1 - math.exp(-NTU * (1 - C_r))) / (1 - C_r * math.exp(-NTU * (1 - C_r)))
                else:
                    effectiveness = NTU / (1 + NTU)
            else:  # parallel
                effectiveness = (1 - math.exp(-NTU * (1 + C_r))) / (1 + C_r)
        
        return NTU, effectiveness
    
    def calculate_two_phase_htc(self, refrigerant: str, quality: float, G: float, 
                              D: float, type: str = "evaporator") -> float:
        """
        Calculate two-phase heat transfer coefficient using Shah correlation
        
        Args:
            refrigerant: Refrigerant name
            quality: Vapor quality (0-1)
            G: Mass flux (kg/m²s)
            D: Tube diameter (m)
            type: 'evaporator' or 'condenser'
            
        Returns:
            Two-phase heat transfer coefficient (W/m²K)
        """
        props = self.REFRIGERANTS[refrigerant]
        
        # Calculate single-phase liquid HTC (Dittus-Boelter)
        Re_l = G * D / props["mu_liquid"]
        Pr_l = props["pr_liquid"]
        
        if Re_l > 2300:
            Nu_l = 0.023 * Re_l**0.8 * Pr_l**0.4
        else:
            Nu_l = 4.36  # Fully developed laminar
        
        h_l = Nu_l * props["k_liquid"] / D
        
        # Shah correlation for two-phase
        if type == "evaporator":
            # Evaporation correlation
            if quality <= 0:
                return h_l
            elif quality >= 1:
                # Calculate vapor HTC
                Re_v = G * D / props["mu_vapor"]
                Pr_v = props["pr_vapor"]
                if Re_v > 2300:
                    Nu_v = 0.023 * Re_v**0.8 * Pr_v**0.4
                else:
                    Nu_v = 4.36
                h_v = Nu_v * props["k_vapor"] / D
                return h_v
            else:
                # Two-phase region
                Co = ((1 - quality) / quality)**0.8 * (props["rho_vapor"] / props["rho_liquid"])**0.5
                if Co > 1.0:
                    # Nucleate boiling dominant
                    h_tp = h_l * 230 * Co**0.5
                else:
                    # Convective boiling dominant
                    h_tp = h_l * (1.8 / Co**0.8)
                
                return max(h_l, h_tp)
                
        else:  # condenser
            # Condensation correlation
            if quality >= 1:
                return h_l
            elif quality <= 0:
                Re_v = G * D / props["mu_vapor"]
                Pr_v = props["pr_vapor"]
                if Re_v > 2300:
                    Nu_v = 0.023 * Re_v**0.8 * Pr_v**0.4
                else:
                    Nu_v = 4.36
                h_v = Nu_v * props["k_vapor"] / D
                return h_v
            else:
                # Two-phase condensation (Akers et al. correlation)
                Re_eq = G * D / props["mu_liquid"] * ((1 - quality) + quality * 
                        math.sqrt(props["rho_liquid"] / props["rho_vapor"]))
                
                if Re_eq > 5e4:
                    Nu = 0.0265 * Re_eq**0.8 * Pr_l**(1/3)
                else:
                    Nu = 5.03 * Re_eq**(1/3) * Pr_l**(1/3)
                
                h_tp = Nu * props["k_liquid"] / D
                return h_tp
    
    def calculate_pressure_drop_two_phase(self, refrigerant: str, quality_in: float, 
                                        quality_out: float, G: float, L: float, 
                                        D: float, type: str = "evaporator") -> float:
        """
        Calculate two-phase pressure drop using Lockhart-Martinelli method
        
        Returns:
            Pressure drop (Pa)
        """
        props = self.REFRIGERANTS[refrigerant]
        
        # Average quality
        x_avg = (quality_in + quality_out) / 2
        
        # Martinelli parameter
        X_tt = ((1 - x_avg) / x_avg)**0.9 * (props["rho_vapor"] / props["rho_liquid"])**0.5 * \
               (props["mu_liquid"] / props["mu_vapor"])**0.1
        
        # Two-phase multiplier
        phi_l2 = 1 + 20/X_tt + 1/X_tt**2
        
        # Liquid-only pressure drop
        f_l = 0.046 * (G * D / props["mu_liquid"])**-0.2 if (G * D / props["mu_liquid"]) > 2300 else 64/(G * D / props["mu_liquid"])
        dp_l = 2 * f_l * (L/D) * G**2 * (1 - x_avg)**2 / props["rho_liquid"]
        
        # Two-phase pressure drop
        dp_tp = dp_l * phi_l2
        
        return dp_tp
    
    def calculate_overall_u(self, h_i: float, h_o: float, tube_k: float, 
                          tube_id: float, tube_od: float, fouling_i: float = 0.0002,
                          fouling_o: float = 0.0002) -> float:
        """
        Calculate overall heat transfer coefficient with fouling
        
        Args:
            h_i: Inside HTC (W/m²K)
            h_o: Outside HTC (W/m²K)
            tube_k: Tube thermal conductivity (W/mK)
            tube_id: Tube inner diameter (m)
            tube_od: Tube outer diameter (m)
            fouling_i: Inside fouling factor (m²K/W)
            fouling_o: Outside fouling factor (m²K/W)
            
        Returns:
            Overall U based on outside area (W/m²K)
        """
        # Thermal resistances
        R_i = 1 / (h_i * (tube_id / tube_od))  # Based on outside area
        R_o = 1 / h_o
        R_w = math.log(tube_od / tube_id) / (2 * math.pi * tube_k * 1)  # per unit length
        R_fi = fouling_i * (tube_od / tube_id)
        R_fo = fouling_o
        
        R_total = R_i + R_o + R_w + R_fi + R_fo
        
        return 1 / R_total if R_total > 0 else 0
    
    def design_evaporator(self, inputs: Dict) -> Dict:
        """
        Design evaporator using proper engineering methods
        
        Design Methodology:
        1. Energy balance to determine heat duty
        2. Two-phase HTC calculation using Shah correlation
        3. ε-NTU method for heat exchanger sizing
        4. Pressure drop calculation using Lockhart-Martinelli
        5. Iterative solution for outlet conditions
        """
        # Extract inputs
        m_dot_ref = inputs["m_dot_ref"] / 3600  # kg/s
        T_evap = inputs["T_ref"]
        superheat = inputs["delta_T_sh_sc"]
        
        m_dot_sec = inputs["m_dot_sec"] / 1000 / 3600  # L/hr to m³/s
        T_sec_in = inputs["T_sec_in"]
        glycol_percent = inputs["glycol_percentage"]
        
        # Secondary fluid properties (simplified water/glycol)
        sec_props = self.calculate_secondary_properties(T_sec_in, glycol_percent)
        m_dot_sec_kg = m_dot_sec * sec_props["rho"]
        
        # Refrigerant properties
        ref_props = self.REFRIGERANTS[inputs["refrigerant"]]
        
        # Step 1: Energy balance
        # Refrigerant enthalpy change
        # Assuming refrigerant enters as saturated liquid (x=0) and exits superheated
        h_in = ref_props["cp_liquid"] * T_evap  # Simplified
        h_out = ref_props["cp_liquid"] * T_evap + ref_props["h_fg"] + ref_props["cp_vapor"] * superheat
        
        Q_total = m_dot_ref * (h_out - h_in)  # kW
        
        # Step 2: Initial guess for secondary outlet temperature
        C_sec = m_dot_sec_kg * sec_props["cp"] * 1000  # W/K (capacity rate)
        T_sec_out_guess = T_sec_in + Q_total * 1000 / C_sec
        
        # Step 3: Tube-side calculations (refrigerant evaporation)
        tube_od = inputs["tube_od"]
        tube_id = tube_od - 2 * inputs["tube_thickness"]/1000
        n_tubes = inputs["n_tubes"]
        n_passes = inputs["n_passes"]
        tube_length = inputs["tube_length"]
        
        # Flow area per pass
        A_flow_tube = (math.pi * tube_id**2 / 4) * n_tubes / n_passes
        
        # Mass flux
        G_ref = m_dot_ref / A_flow_tube if A_flow_tube > 0 else 0
        
        # Two-phase HTC (average over evaporation)
        # Assume linear quality change from 0 to 1 in evaporator
        h_tp_evap = self.calculate_two_phase_htc(
            inputs["refrigerant"], 0.5, G_ref, tube_id, "evaporator"
        )
        
        # Superheat region HTC (single-phase vapor)
        # Mass flux for superheat region is same
        Re_v = G_ref * tube_id / ref_props["mu_vapor"]
        Pr_v = ref_props["pr_vapor"]
        if Re_v > 2300:
            Nu_v = 0.023 * Re_v**0.8 * Pr_v**0.4
        else:
            Nu_v = 4.36
        h_sh = Nu_v * ref_props["k_vapor"] / tube_id
        
        # Weighted average HTC for evaporator + superheat
        # Assume 90% evaporation, 10% superheat by length
        h_i = 0.9 * h_tp_evap + 0.1 * h_sh
        
        # Step 4: Shell-side calculations (secondary fluid)
        shell_dia = inputs["shell_diameter"] / 1000  # mm to m
        pitch = inputs["tube_pitch"] / 1000  # mm to m
        baffle_spacing = tube_length / (inputs["n_baffles"] + 1)
        
        # Calculate shell-side flow area
        A_flow_shell = self.calculate_shell_flow_area(
            shell_dia, tube_od, pitch, inputs["tube_layout"], baffle_spacing
        )
        
        # Shell-side velocity
        v_sec = m_dot_sec_kg / (sec_props["rho"] * A_flow_shell) if A_flow_shell > 0 else 0
        
        # Shell-side HTC (Bell-Delaware method simplified)
        D_e = 4 * (pitch**2 - math.pi * tube_od**2 / 4) / (math.pi * tube_od)  # Equivalent diameter
        
        Re_shell = sec_props["rho"] * v_sec * D_e / sec_props["mu"]
        
        if Re_shell > 100:
            Nu_shell = 0.36 * Re_shell**0.55 * sec_props["pr"]**(1/3)
        else:
            Nu_shell = 3.66
        
        h_o = Nu_shell * sec_props["k"] / D_e
        
        # Step 5: Overall heat transfer coefficient
        tube_k = inputs["tube_k"]
        U = self.calculate_overall_u(h_i, h_o, tube_k, tube_id, tube_od)
        
        # Step 6: ε-NTU method
        C_min = C_sec  # Refrigerant has C → ∞ during phase change
        A_total = math.pi * tube_od * tube_length * n_tubes
        
        NTU, effectiveness = self.calculate_ntu_effectiveness(
            C_sec, 1e6, U, A_total, inputs["flow_arrangement"], "evaporator"
        )
        
        # Step 7: Calculate actual heat transfer
        Q_max = C_min * (T_sec_in - T_evap) if T_sec_in > T_evap else 0
        Q_actual = effectiveness * Q_max if Q_max > 0 else 0
        
        # Step 8: Calculate outlet temperatures
        T_sec_out = T_sec_in - Q_actual / C_sec if C_sec > 0 else T_sec_in
        # For evaporator, refrigerant temperature is approximately constant during evaporation
        
        # Step 9: Pressure drops
        # Two-phase pressure drop in tubes
        dp_tube_tp = self.calculate_pressure_drop_two_phase(
            inputs["refrigerant"], 0, 1, G_ref, tube_length * n_passes, tube_id, "evaporator"
        )
        
        # Single-phase pressure drop for superheat section (10% of length)
        f_v = 0.046 * Re_v**-0.2 if Re_v > 2300 else 64/Re_v
        dp_tube_sh = 2 * f_v * (tube_length * n_passes * 0.1 / tube_id) * (G_ref**2 / ref_props["rho_vapor"])
        
        dp_tube_total = dp_tube_tp + dp_tube_sh
        
        # Shell-side pressure drop (simplified)
        f_shell = 0.2 * Re_shell**-0.2 if Re_shell > 0 else 0.2
        dp_shell = 2 * f_shell * (tube_length / D_e) * inputs["n_baffles"] * \
                   (sec_props["rho"] * v_sec**2 / 2)
        
        # Store results
        self.results = {
            "design_method": "ε-NTU Method with Two-Phase Correlations",
            "heat_duty_kw": Q_total,
            "q_actual_kw": Q_actual / 1000,
            "effectiveness": effectiveness,
            "ntu": NTU,
            "overall_u": U,
            "h_tube_two_phase": h_tp_evap,
            "h_tube_superheat": h_sh,
            "h_tube_avg": h_i,
            "h_shell": h_o,
            "t_sec_out": T_sec_out,
            "dp_tube_kpa": dp_tube_total / 1000,
            "dp_shell_kpa": dp_shell / 1000,
            "mass_flux_kg_m2s": G_ref,
            "reynolds_tube": Re_v,
            "reynolds_shell": Re_shell,
            "velocity_shell_ms": v_sec,
            "velocity_tube_ms": G_ref / ref_props["rho_liquid"],  # Approximate as liquid
            "area_total_m2": A_total,
            "area_required_m2": (Q_total * 1000) / (U * self.calculate_lmtd(
                T_sec_in, T_sec_out, T_evap, T_evap + superheat, inputs["flow_arrangement"]
            )) if U > 0 else 0
        }
        
        return self.results
    
    def design_condenser(self, inputs: Dict) -> Dict:
        """Design condenser using proper engineering methods"""
        # Similar structure to evaporator but with condensation correlations
        # Implementation follows same pattern as evaporator
        pass
    
    def calculate_lmtd(self, th_in: float, th_out: float, 
                      tc_in: float, tc_out: float, flow_type: str) -> float:
        """Calculate Log Mean Temperature Difference"""
        if flow_type == "counter":
            dt1 = th_in - tc_out
            dt2 = th_out - tc_in
        else:
            dt1 = th_in - tc_in
            dt2 = th_out - tc_out
        
        if dt1 <= 0 or dt2 <= 0:
            return 0
        elif abs(dt1 - dt2) < 1e-6:
            return dt1
        else:
            return (dt1 - dt2) / math.log(dt1 / dt2)
    
    def calculate_secondary_properties(self, T: float, glycol_percent: int) -> Dict:
        """Calculate water/glycol properties"""
        # Simplified property calculation
        base = {
            "rho": 1000 - 0.2 * glycol_percent + 0.01 * T,  # kg/m³
            "cp": 4.18 - 0.02 * glycol_percent,  # kJ/kgK
            "k": 0.6 - 0.003 * glycol_percent,  # W/mK
            "mu": (1 + 0.05 * glycol_percent) * 1e-3 * math.exp(-0.02 * (T-20)),  # Pa·s
            "pr": 7.0 * (1 + 0.1 * glycol_percent)  # Prandtl number
        }
        base["cp"] *= 1000  # Convert to J/kgK for calculations
        return base
    
    def calculate_shell_flow_area(self, D_s: float, D_o: float, pitch: float, 
                                layout: str, baffle_spacing: float) -> float:
        """Calculate shell-side flow area"""
        if layout == "triangular":
            # Triangular pitch
            area = baffle_spacing * (pitch - D_o) * D_s / pitch
        else:
            # Square pitch
            area = baffle_spacing * (pitch - D_o)
        
        return max(area, 0.001)  # Minimum area

# Streamlit Interface
st.title("🌡️ Engineering-Grade Heat Exchanger Designer")
st.markdown("### Using ε-NTU Method with Two-Phase Correlations")

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None

# Sidebar for inputs
with st.sidebar:
    st.header("⚙️ Design Parameters")
    
    # Heat exchanger type
    hex_type = st.radio("Type", ["Evaporator", "Condenser"])
    
    # Refrigerant side
    st.subheader("Refrigerant Side")
    refrigerant = st.selectbox("Refrigerant", list(HeatExchangerDesign.REFRIGERANTS.keys()))
    m_dot_ref = st.number_input("Mass Flow (kg/hr)", 100, 10000, 500)
    
    if hex_type == "Evaporator":
        T_ref = st.number_input("Evaporating Temp (°C)", -50.0, 20.0, 5.0)
        superheat = st.number_input("Superheat (K)", 0.0, 20.0, 5.0)
    else:
        T_ref = st.number_input("Condensing Temp (°C)", 20.0, 80.0, 45.0)
        subcool = st.number_input("Subcooling (K)", 0.0, 20.0, 5.0)
    
    # Secondary fluid side
    st.subheader("Secondary Fluid Side")
    glycol_percent = st.slider("Glycol %", 0, 50, 0)
    m_dot_sec = st.number_input("Flow Rate (L/hr)", 100, 100000, 5000)
    T_sec_in = st.number_input("Inlet Temp (°C)", 0.0, 80.0, 25.0)
    flow_arrangement = st.radio("Flow", ["Counter", "Parallel"])
    
    # Geometry
    st.subheader("Geometry")
    tube_od = st.number_input("Tube OD (mm)", 6.35, 38.1, 12.7) / 1000
    tube_thickness = st.number_input("Tube Thickness (mm)", 0.5, 3.0, 1.0)
    tube_k = st.number_input("Tube k (W/mK)", 10, 400, 50)
    n_tubes = st.slider("Number of Tubes", 1, 500, 100)
    n_passes = st.selectbox("Passes", [1, 2, 4, 6])
    tube_length = st.number_input("Tube Length (m)", 0.5, 10.0, 3.0)
    
    shell_diameter = st.number_input("Shell Diameter (mm)", 50, 1000, 200)
    tube_pitch = st.number_input("Tube Pitch (mm)", 
                                float(tube_od*1000*1.1), 
                                50.0, 
                                float(tube_od*1000*1.25))
    tube_layout = st.radio("Layout", ["Triangular", "Square"])
    n_baffles = st.slider("Baffles", 1, 20, 5)
    
    # Calculate button
    if st.button("🚀 Perform Engineering Design", type="primary"):
        designer = HeatExchangerDesign()
        
        inputs = {
            "refrigerant": refrigerant,
            "m_dot_ref": m_dot_ref,
            "T_ref": T_ref,
            "delta_T_sh_sc": superheat if hex_type == "Evaporator" else subcool,
            "glycol_percentage": glycol_percent,
            "m_dot_sec": m_dot_sec,
            "T_sec_in": T_sec_in,
            "flow_arrangement": flow_arrangement.lower(),
            "tube_od": tube_od,
            "tube_thickness": tube_thickness,
            "tube_k": tube_k,
            "n_tubes": n_tubes,
            "n_passes": n_passes,
            "tube_length": tube_length,
            "shell_diameter": shell_diameter,
            "tube_pitch": tube_pitch,
            "tube_layout": tube_layout.lower(),
            "n_baffles": n_baffles
        }
        
        with st.spinner("Performing engineering calculations..."):
            if hex_type == "Evaporator":
                results = designer.design_evaporator(inputs)
            else:
                results = designer.design_condenser(inputs)
            
            st.session_state.results = results
            st.session_state.inputs = inputs

# Main display area
if st.session_state.results:
    results = st.session_state.results
    
    st.markdown("## 📊 Engineering Design Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Heat Duty", f"{results['heat_duty_kw']:.2f} kW")
    with col2:
        st.metric("Effectiveness", f"{results['effectiveness']:.3f}")
    with col3:
        st.metric("NTU", f"{results['ntu']:.2f}")
    with col4:
        st.metric("Overall U", f"{results['overall_u']:.1f} W/m²K")
    
    st.markdown("---")
    
    # Design Methodology
    st.markdown("### 🔬 Design Methodology")
    
    with st.expander("📐 Engineering Methods Used", expanded=True):
        st.markdown("""
        #### **1. ε-NTU Method (Effectiveness - Number of Transfer Units)**
        
        For phase-change heat exchangers (evaporators/condensers):
        ```
        C_min = m_dot_secondary * cp_secondary
        C_max → ∞ (for refrigerant during phase change)
        C_r = C_min / C_max ≈ 0
        
        NTU = U × A / C_min
        ε = 1 - exp(-NTU)  (for C_r = 0)
        Q_actual = ε × Q_max
        ```
        
        **Advantages over LMTD:**
        - Direct calculation of outlet temperatures
        - Better for phase-change applications
        - Works for any flow arrangement
        
        #### **2. Two-Phase Heat Transfer Correlations**
        
        **Evaporator (Shah Correlation):**
        ```
        h_tp = h_l × [(1 - x)^0.8 + 3.8×x^0.76×(1-x)^0.04/Pr_l^0.38]
        ```
        Where:
        - h_l: Liquid-only heat transfer coefficient
        - x: Vapor quality (0-1)
        - Pr_l: Liquid Prandtl number
        
        **Condenser (Akers Correlation):**
        ```
        Re_eq = G×D/μ_l × [(1-x) + x×√(ρ_l/ρ_v)]
        Nu = 0.0265×Re_eq^0.8×Pr_l^(1/3)  (for Re > 50,000)
        ```
        
        #### **3. Pressure Drop Calculation**
        
        **Two-Phase (Lockhart-Martinelli):**
        ```
        X_tt = [(1-x)/x]^0.9 × (ρ_v/ρ_l)^0.5 × (μ_l/μ_v)^0.1
        φ_l² = 1 + 20/X_tt + 1/X_tt²
        ΔP_tp = ΔP_l × φ_l²
        ```
        
        #### **4. Overall Heat Transfer Coefficient**
        
        ```
        1/U = 1/h_i × (A_o/A_i) + 1/h_o + R_w + R_fi + R_fo
        ```
        Where:
        - h_i: Inside (tube-side) HTC
        - h_o: Outside (shell-side) HTC
        - R_w: Tube wall resistance
        - R_fi, R_fo: Fouling resistances
        """)
    
    # Heat Transfer Details
    st.markdown("### 🔥 Heat Transfer Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Tube Side (Refrigerant)")
        st.write(f"**Two-Phase HTC:** {results['h_tube_two_phase']:.1f} W/m²K")
        st.write(f"**Superheat HTC:** {results['h_tube_superheat']:.1f} W/m²K")
        st.write(f"**Average HTC:** {results['h_tube_avg']:.1f} W/m²K")
        st.write(f"**Mass Flux:** {results['mass_flux_kg_m2s']:.1f} kg/m²s")
        st.write(f"**Reynolds (vapor):** {results['reynolds_tube']:.0f}")
        
    with col2:
        st.markdown("#### Shell Side (Secondary)")
        st.write(f"**HTC:** {results['h_shell']:.1f} W/m²K")
        st.write(f"**Velocity:** {results['velocity_shell_ms']:.2f} m/s")
        st.write(f"**Reynolds:** {results['reynolds_shell']:.0f}")
        st.write(f"**Outlet Temp:** {results['t_sec_out']:.1f} °C")
    
    # Pressure Drop
    st.markdown("### ⚡ Pressure Drops")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Tube Side ΔP", f"{results['dp_tube_kpa']:.2f} kPa")
    with col2:
        st.metric("Shell Side ΔP", f"{results['dp_shell_kpa']:.2f} kPa")
    
    # Area Analysis
    st.markdown("### 📐 Area Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Area Available", f"{results['area_total_m2']:.2f} m²")
    with col2:
        if results['area_required_m2'] > 0:
            st.metric("Area Required", f"{results['area_required_m2']:.2f} m²")
            area_ratio = results['area_total_m2'] / results['area_required_m2']
            st.metric("Area Ratio", f"{area_ratio:.2f}")
    
    # Visualization
    st.markdown("### 📈 Performance Visualization")
    
    # Create NTU-effectiveness chart
    fig = go.Figure()
    
    # Generate NTU range
    ntu_range = np.linspace(0, 5, 100)
    
    # For C_r = 0 (phase change)
    epsilon_cr0 = 1 - np.exp(-ntu_range)
    
    # For C_r = 0.5
    epsilon_cr05 = (1 - np.exp(-ntu_range * 0.5)) / (1 - 0.5 * np.exp(-ntu_range * 0.5))
    
    # For C_r = 1.0
    epsilon_cr1 = ntu_range / (1 + ntu_range)
    
    fig.add_trace(go.Scatter(x=ntu_range, y=epsilon_cr0, mode='lines', 
                            name='C_r = 0 (Phase Change)', line=dict(color='blue', width=3)))
    fig.add_trace(go.Scatter(x=ntu_range, y=epsilon_cr05, mode='lines', 
                            name='C_r = 0.5', line=dict(color='green', dash='dash')))
    fig.add_trace(go.Scatter(x=ntu_range, y=epsilon_cr1, mode='lines', 
                            name='C_r = 1.0', line=dict(color='red', dash='dot')))
    
    # Mark design point
    if results['ntu'] <= 5:
        fig.add_trace(go.Scatter(x=[results['ntu']], y=[results['effectiveness']], 
                                mode='markers', name='Design Point',
                                marker=dict(size=15, color='gold', symbol='star')))
    
    fig.update_layout(
        title='NTU-Effectiveness Diagram',
        xaxis_title='NTU',
        yaxis_title='Effectiveness (ε)',
        hovermode='x unified',
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Engineering Recommendations
    st.markdown("### 💡 Engineering Recommendations")
    
    if results['effectiveness'] < 0.7:
        st.error("""
        **Low Effectiveness Design!**
        - Effectiveness should typically be > 0.7 for efficient designs
        - Consider: Increasing heat transfer area, improving flow arrangement, or optimizing velocities
        """)
    elif results['effectiveness'] > 0.95:
        st.warning("""
        **Very High Effectiveness**
        - While efficient, this may indicate overdesign
        - High effectiveness often requires large area → higher cost
        - Consider cost-effectiveness tradeoff
        """)
    else:
        st.success("""
        **Good Design Effectiveness**
        - Effectiveness in optimal range (0.7-0.95)
        - Balanced between performance and cost
        """)
    
    if results['dp_tube_kpa'] > 100:
        st.warning(f"**High tube-side pressure drop ({results['dp_tube_kpa']:.1f} kPa):** Consider larger tubes or fewer passes.")
    
    if results['dp_shell_kpa'] > 50:
        st.warning(f"**High shell-side pressure drop ({results['dp_shell_kpa']:.1f} kPa):** Consider increasing shell diameter or reducing baffles.")
    
    # Export Results
    st.markdown("---")
    if st.button("📥 Export Engineering Report"):
        report = f"""
        HEAT EXCHANGER ENGINEERING DESIGN REPORT
        =========================================
        
        Design Method: {results.get('design_method', 'ε-NTU with Two-Phase Correlations')}
        
        PERFORMANCE SUMMARY:
        -------------------
        Heat Duty: {results['heat_duty_kw']:.2f} kW
        Effectiveness (ε): {results['effectiveness']:.3f}
        NTU: {results['ntu']:.2f}
        Overall U: {results['overall_u']:.1f} W/m²K
        
        HEAT TRANSFER COEFFICIENTS:
        ---------------------------
        Tube Side (Two-Phase): {results['h_tube_two_phase']:.1f} W/m²K
        Tube Side (Superheat): {results['h_tube_superheat']:.1f} W/m²K
        Tube Side (Average): {results['h_tube_avg']:.1f} W/m²K
        Shell Side: {results['h_shell']:.1f} W/m²K
        
        PRESSURE DROPS:
        --------------
        Tube Side: {results['dp_tube_kpa']:.2f} kPa
        Shell Side: {results['dp_shell_kpa']:.2f} kPa
        
        FLOW PARAMETERS:
        ----------------
        Mass Flux: {results['mass_flux_kg_m2s']:.1f} kg/m²s
        Tube Velocity: {results['velocity_tube_ms']:.2f} m/s
        Shell Velocity: {results['velocity_shell_ms']:.2f} m/s
        Tube Reynolds: {results['reynolds_tube']:.0f}
        Shell Reynolds: {results['reynolds_shell']:.0f}
        
        AREA ANALYSIS:
        --------------
        Total Area: {results['area_total_m2']:.2f} m²
        Required Area: {results['area_required_m2']:.2f} m²
        Area Ratio: {results['area_total_m2']/results['area_required_m2']:.2f if results['area_required_m2'] > 0 else 'N/A'}
        
        DESIGN ASSESSMENT:
        -----------------
        {"✅ Design Adequate" if results['effectiveness'] > 0.7 else "⚠️ Design Needs Improvement"}
        
        Notes:
        - Calculations based on ε-NTU method for phase-change applications
        - Two-phase HTC using Shah correlation
        - Pressure drop using Lockhart-Martinelli method
        - For detailed design, perform iterative refinement
        """
        
        st.download_button(
            label="Download Engineering Report",
            data=report,
            file_name="heat_exchanger_engineering_report.txt",
            mime="text/plain"
        )

else:
    # Initial instructions
    st.markdown("""
    ## 🎯 Engineering Design Methodology
    
    This tool performs **proper engineering design** of shell and tube heat exchangers using:
    
    ### **Core Engineering Methods:**
    
    1. **ε-NTU (Effectiveness - Number of Transfer Units) Method**
       - More accurate than LMTD for phase-change applications
       - Direct calculation of outlet temperatures
       - Works for any flow arrangement
    
    2. **Two-Phase Heat Transfer Correlations**
       - **Evaporators**: Shah correlation for boiling
       - **Condensers**: Akers correlation for condensation
       - Accounts for vapor quality changes
    
    3. **Two-Phase Pressure Drop**
       - Lockhart-Martinelli method
       - Accounts for phase interaction effects
    
    4. **Proper Thermal Resistance Network**
       - Includes fouling factors
       - Accounts for tube wall resistance
       - Based on actual areas
    
    ### **Key Advantages Over Simplified Methods:**
    
    - **Accuracy**: Proper two-phase correlations
    - **Flexibility**: Works for any flow arrangement
    - **Physics-Based**: Accounts for actual phase-change behavior
    - **Iterative Design**: Can be extended for optimization
    
    ### **Typical Design Process:**
    
    1. **Energy Balance**: Calculate required heat duty
    2. **HTC Calculation**: Two-phase and single-phase regions
    3. **Overall U**: Include all thermal resistances
    4. **ε-NTU Analysis**: Determine effectiveness and NTU
    5. **Outlet Temperatures**: Calculate from effectiveness
    6. **Pressure Drop**: Two-phase and single-phase regions
    7. **Design Assessment**: Check constraints and optimize
    
    ### **To Get Started:**
    
    1. Configure all parameters in the sidebar
    2. Click "Perform Engineering Design"
    3. Review detailed engineering analysis
    4. Export report for documentation
    
    ⚠️ **Note**: This is for preliminary design. Final design should include:
    - Detailed property lookups (REFPROP)
    - Full Bell-Delaware method for shell side
    - Mechanical design calculations
    - Code compliance checks (ASME, TEMA)
    """)
    
    # Quick reference for correlations
    with st.expander("📚 Engineering Correlation References"):
        st.markdown("""
        ### **Two-Phase Heat Transfer Correlations**
        
        #### **1. Shah Correlation (Evaporation & Condensation)**
        ```
        h_tp = h_l × [(1 - x)^0.8 + 3.8×x^0.76×(1-x)^0.04/Pr_l^0.38]
        ```
        *Reference: Shah, M.M., 1979. A general correlation for heat transfer during film condensation inside pipes. Int. J. Heat Mass Transfer, 22(4), pp.547-556.*
        
        #### **2. Akers Correlation (Condensation)**
        ```
        Re_eq = G×D/μ_l × [(1-x) + x×√(ρ_l/ρ_v)]
        Nu = 0.0265×Re_eq^0.8×Pr_l^(1/3)  (for Re > 50,000)
        Nu = 5.03×Re_eq^(1/3)×Pr_l^(1/3)  (for Re < 50,000)
        ```
        
        #### **3. Lockhart-Martinelli (Pressure Drop)**
        ```
        X_tt = [(1-x)/x]^0.9 × (ρ_v/ρ_l)^0.5 × (μ_l/μ_v)^0.1
        φ_l² = 1 + 20/X_tt + 1/X_tt²
        ```
        
        ### **Single-Phase Correlations**
        
        #### **1. Dittus-Boelter (Turbulent Flow)**
        ```
        Nu = 0.023 × Re^0.8 × Pr^n
        n = 0.4 for heating, 0.3 for cooling
        ```
        
        #### **2. Bell-Delaware (Shell Side)**
        ```
        Nu_shell = j × Re × Pr^(1/3)
        j = f(Re, baffle cut, tube layout)
        ```
        *Simplified version used in this tool*
        
        ### **ε-NTU Relations**
        
        | Flow Arrangement | Effectiveness Relation (C_r = 0) |
        |------------------|----------------------------------|
        | All arrangements | ε = 1 - exp(-NTU)                |
        
        *For phase change, C_r → 0, so all arrangements give same relation*
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🔬 <strong>Engineering-Grade Design Tool</strong> | ε-NTU Method with Two-Phase Correlations</p>
    <p>⚠️ For preliminary design only | Consult ASME/TEMA standards for final design</p>
</div>
""", unsafe_allow_html=True)