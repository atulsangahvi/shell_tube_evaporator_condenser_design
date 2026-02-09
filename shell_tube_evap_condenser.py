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
    page_title="Shell & Tube DX Evaporator & Condenser Designer",
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
    .evaporator-type-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-weight: bold;
        margin-left: 0.5rem;
    }
    .dx-badge {
        background-color: #3B82F6;
        color: white;
    }
    .flooded-badge {
        background-color: #10B981;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

class DXHeatExchangerDesign:
    """DX (Direct Expansion) Shell & Tube Heat Exchanger Design"""
    
    REFRIGERANTS = {
        "R134a": {
            "cp_vapor": 0.85, "cp_liquid": 1.43, "h_fg": 198.7,
            "rho_vapor": 14.4, "rho_liquid": 1278,
            "mu_vapor": 1.11e-5, "mu_liquid": 2.04e-4,
            "k_vapor": 0.0116, "k_liquid": 0.085,
            "pr_vapor": 0.82, "pr_liquid": 3.43,
            "sigma": 0.0085
        },
        "R404A": {
            "cp_vapor": 0.82, "cp_liquid": 1.55, "h_fg": 163.3,
            "rho_vapor": 33.2, "rho_liquid": 1132,
            "mu_vapor": 1.23e-5, "mu_liquid": 1.98e-4,
            "k_vapor": 0.0108, "k_liquid": 0.072,
            "pr_vapor": 0.94, "pr_liquid": 4.26,
            "sigma": 0.0068
        },
        # ... [keep other refrigerants]
    }
    
    # Tube materials properties
    TUBE_MATERIALS = {
        "Copper": {"k": 386, "density": 8960, "cost_factor": 1.0},
        "Cu-Ni 90/10": {"k": 40, "density": 8940, "cost_factor": 1.8},
        # ... [other materials]
    }
    
    # Tube sizes (inches to meters)
    TUBE_SIZES = {
        "1/4\"": 0.00635, "3/8\"": 0.009525, "1/2\"": 0.0127,
        "5/8\"": 0.015875, "3/4\"": 0.01905, "1\"": 0.0254,
        "1.25\"": 0.03175, "1.5\"": 0.0381
    }
    
    # DX-specific parameters
    DX_PARAMETERS = {
        "min_superheat": 3.0,  # K - minimum for TXV operation
        "max_superheat": 8.0,  # K - typical maximum
        "oil_circulation_rate": 0.03,  # 3% oil in refrigerant
        "distribution_header_length": 0.3,  # m - for refrigerant distribution
    }
    
    def __init__(self):
        self.results = {}
        self.evaporator_type = "DX"  # Fixed for this class
    
    def calculate_dx_evaporator_htc(self, refrigerant: str, quality: float, G: float,
                                  D: float, T_sat: float) -> float:
        """DX evaporator HTC - refrigerant boiling inside tubes"""
        # Use Shah correlation for flow boiling in tubes
        props = self.REFRIGERANTS[refrigerant]
        
        Re_l = G * D / props["mu_liquid"]
        Pr_l = props["pr_liquid"]
        
        # Shah correlation parameters
        Co = ((1 - quality) / quality)**0.8 * (props["rho_vapor"] / props["rho_liquid"])**0.5
        Bo = G * props["h_fg"] * 1000 / (props["k_liquid"] * T_sat) if T_sat > 0 else 0.0001
        
        if Co <= 0.65:
            N = Co
        else:
            N = 0.38 * Co**-0.3
        
        # Enhancement factor
        F = 14.7 * Bo**0.56 * N if Bo > 0.0011 else 15.43 * Bo**0.56 * N
        
        # Single-phase liquid Nusselt (Gnielinski)
        if Re_l < 2300:
            Nu_l = 4.36
        else:
            f_l = (0.79 * math.log(Re_l) - 1.64)**-2
            Nu_l = (f_l/8) * (Re_l - 1000) * Pr_l / (1 + 12.7 * (f_l/8)**0.5 * (Pr_l**(2/3) - 1))
        
        # Two-phase Nusselt
        Nu_tp = Nu_l * (1 + 2.4e4 * Bo**1.16 + 1.37 * Co**-0.86)
        
        return Nu_tp * props["k_liquid"] / D
    
    def calculate_dx_pressure_drop(self, refrigerant: str, G: float, D: float,
                                 L: float, n_passes: int, quality_in: float,
                                 quality_out: float) -> float:
        """DX evaporator pressure drop with acceleration term"""
        props = self.REFRIGERANTS[refrigerant]
        
        # Use average quality
        quality_avg = (quality_in + quality_out) / 2
        
        # Friedel correlation for two-phase frictional drop
        Re_l = G * D / props["mu_liquid"]
        Re_v = G * D / props["mu_vapor"]
        
        f_lo = (0.79 * math.log(Re_l) - 1.64)**-2 if Re_l > 2300 else 64/Re_l
        
        # Two-phase multiplier
        Fr = G**2 / (9.81 * D * props["rho_liquid"]**2)
        We = G**2 * D / (props["rho_liquid"] * props["sigma"])
        
        A = (1 - quality_avg)**2 + quality_avg**2 * (props["rho_liquid"] * f_lo) / (props["rho_vapor"] * f_lo)
        B = quality_avg**0.78 * (1 - quality_avg)**0.224
        C = (props["rho_liquid"] / props["rho_vapor"])**0.91 * (props["mu_vapor"] / props["mu_liquid"])**0.19 * (1 - props["mu_vapor"] / props["mu_liquid"])**0.7
        
        phi_lo2 = A + 3.24 * B * C / (Fr**0.045 * We**0.035)
        
        # Frictional pressure drop
        dp_friction = 2 * f_lo * G**2 / (D * props["rho_liquid"]) * phi_lo2 * L * n_passes
        
        # Acceleration pressure drop (significant in evaporation)
        # ΔP_acc = G² * [(x²/ρ_v) + ((1-x)²/ρ_l)]_out - [(x²/ρ_v) + ((1-x)²/ρ_l)]_in
        acc_in = (quality_in**2 / props["rho_vapor"]) + ((1 - quality_in)**2 / props["rho_liquid"])
        acc_out = (quality_out**2 / props["rho_vapor"]) + ((1 - quality_out)**2 / props["rho_liquid"])
        dp_acceleration = G**2 * (acc_out - acc_in)
        
        return dp_friction + dp_acceleration
    
    def calculate_refrigerant_distribution(self, n_tubes: int, m_dot_ref: float,
                                         tube_id: float) -> Dict:
        """Calculate refrigerant distribution for DX evaporator"""
        # Critical for DX evaporators - maldistribution reduces performance
        
        # Flow per tube
        m_dot_per_tube = m_dot_ref / n_tubes
        
        # Recommended minimum flow per tube for good distribution
        # Based on typical distributor design
        min_flow_per_tube = 0.001  # kg/s (3.6 kg/hr)
        
        distribution_status = "Good" if m_dot_per_tube >= min_flow_per_tube else "Poor"
        
        # Velocity in distribution header (assuming 1" header)
        header_id = 0.0254  # 1" tube
        header_area = math.pi * header_id**2 / 4
        header_velocity = m_dot_ref / (props["rho_liquid"] * header_area) if hasattr(self, 'props') else 0
        
        return {
            "flow_per_tube_kg_hr": m_dot_per_tube * 3600,
            "distribution_status": distribution_status,
            "header_velocity_ms": header_velocity,
            "recommended_min_flow_per_tube_kg_hr": min_flow_per_tube * 3600
        }
    
    def calculate_superheat_uniformity(self, tube_length: float, G: float,
                                     quality_in: float) -> float:
        """Calculate superheat uniformity in DX evaporator"""
        # In DX evaporators, some tubes may have different superheat
        # due to maldistribution or different circuit lengths
        
        # Simplified calculation of dryness point
        # Where in the tube does complete evaporation occur?
        h_fg = self.props["h_fg"] * 1000 if hasattr(self, 'props') else 200000  # J/kg
        cp_vapor = self.props["cp_vapor"] * 1000 if hasattr(self, 'props') else 1000  # J/kgK
        
        # Heat flux (simplified)
        q = G * h_fg / tube_length  # W/m²
        
        # Length to dryout
        L_dryout = (1 - quality_in) * h_fg * G / q if q > 0 else tube_length
        
        # Superheat length
        L_superheat = tube_length - L_dryout
        
        # Superheat uniformity factor (0-1, 1 = perfect)
        uniformity = min(1.0, L_superheat / tube_length) if tube_length > 0 else 0
        
        return uniformity
    
    def design_dx_evaporator(self, inputs: Dict) -> Dict:
        """Design DX evaporator specifically"""
        # Store props for convenience
        self.props = self.REFRIGERANTS[inputs["refrigerant"]]
        
        # Fixed refrigerant flow from compressor
        m_dot_ref = inputs["m_dot_ref"] / 3600  # kg/s
        T_evap = inputs["T_ref"]
        delta_T_superheat = inputs["delta_T_sh_sc"]
        
        # Water/glycol side
        m_dot_sec_L = inputs["m_dot_sec"] / 3600  # L/s
        T_sec_in = inputs["T_sec_in"]
        
        # Geometry
        tube_od = self.TUBE_SIZES[inputs["tube_size"]]
        tube_id = max(tube_od - 2 * inputs["tube_thickness"]/1000, tube_od * 0.8)
        n_tubes = inputs["n_tubes"]
        n_passes = inputs["n_passes"]
        tube_length = inputs["tube_length"]
        
        # DX-specific calculations
        # Refrigerant distribution check
        distribution = self.calculate_refrigerant_distribution(n_tubes, m_dot_ref, tube_id)
        
        # Tube flow area and mass flux
        tube_flow_area = (math.pi * tube_id**2 / 4) * n_tubes / n_passes
        G_ref = m_dot_ref / tube_flow_area
        
        # Quality range (typical for DX)
        quality_in = 0.2  # Typical after expansion valve
        quality_out = 1.0  # Fully evaporated
        quality_avg = 0.6
        
        # Heat transfer coefficient
        h_ref = self.calculate_dx_evaporator_htc(
            inputs["refrigerant"], quality_avg, G_ref, tube_id, T_evap
        )
        
        # Pressure drop
        dp_tube = self.calculate_dx_pressure_drop(
            inputs["refrigerant"], G_ref, tube_id, tube_length,
            n_passes, quality_in, quality_out
        )
        
        # Superheat uniformity
        uniformity = self.calculate_superheat_uniformity(tube_length, G_ref, quality_in)
        
        # Store DX-specific results
        self.results.update({
            "evaporator_type": "DX",
            "distribution_status": distribution["distribution_status"],
            "flow_per_tube_kg_hr": distribution["flow_per_tube_kg_hr"],
            "superheat_uniformity": uniformity,
            "dryout_risk": "Low" if uniformity > 0.3 else "High",
            "oil_circulation_warning": "Monitor" if n_passes > 2 else "OK",
            "dx_specific_note": "Ensure proper distributor design for uniform feeding"
        })
        
        return self.results
    
    def design_condenser(self, inputs: Dict) -> Dict:
        """Design condenser (same tube-side refrigerant)"""
        # Similar to previous condenser design but with clear labeling
        self.props = self.REFRIGERANTS[inputs["refrigerant"]]
        
        # ... [condenser calculations]
        
        self.results.update({
            "heat_exchanger_type": "Condenser",
            "refrigerant_side": "Tube side",
            "water_side": "Shell side"
        })
        
        return self.results
    
    def design_heat_exchanger(self, inputs: Dict) -> Dict:
        """Main design function with type-specific calculations"""
        hex_type = inputs["hex_type"].lower()
        
        if hex_type == "evaporator":
            return self.design_dx_evaporator(inputs)
        else:
            return self.design_condenser(inputs)

class FloodedEvaporatorDesign:
    """Separate class for Flooded Evaporator Design"""
    
    # Different correlations needed
    # Different geometry considerations
    # Different control parameters
    
    def __init__(self):
        self.evaporator_type = "Flooded"
    
    def calculate_pool_boiling_htc(self, refrigerant: str, T_sat: float,
                                 tube_od: float, tube_pitch: float) -> float:
        """Pool boiling HTC for flooded evaporator"""
        # Use Cooper or Rohsenow correlations for pool boiling
        # Shell-side refrigerant, tube-side water
        pass
    
    def calculate_natural_circulation(self, shell_height: float, liquid_level: float,
                                    vapor_density: float, liquid_density: float) -> float:
        """Natural circulation rate in flooded evaporator"""
        # Driven by density difference
        pass
    
    def design_flooded_evaporator(self, inputs: Dict) -> Dict:
        """Design flooded evaporator"""
        # Water in tubes, refrigerant boiling in shell
        # Different geometry: liquid level, vapor disengagement space
        # Different controls: float valve, liquid level sensors
        pass

def create_input_section():
    """Create input section with clear DX/Flooded distinction"""
    st.sidebar.header("⚙️ DX Evaporator & Condenser Design")
    
    # Important disclaimer
    st.sidebar.warning("""
    **⚠️ This tool designs:**
    - **DX Evaporators**: Refrigerant in tubes, water in shell
    - **Condensers**: Refrigerant in tubes, water in shell
    
    **For Flooded Evaporators** (water in tubes, refrigerant in shell),
    use the separate Flooded Evaporator Designer.
    """)
    
    inputs = {}
    
    # Evaporator type indicator
    evaporator_type_display = st.sidebar.selectbox(
        "Heat Exchanger Type",
        ["DX Evaporator", "Condenser"],
        help="DX Evaporator: Refrigerant evaporates in tubes\nCondenser: Refrigerant condenses in tubes"
    )
    
    # Set the actual type
    if "DX Evaporator" in evaporator_type_display:
        inputs["hex_type"] = "evaporator"
        inputs["evaporator_subtype"] = "dx"
    else:
        inputs["hex_type"] = "condenser"
    
    st.sidebar.markdown("---")
    
    # Refrigerant parameters
    st.sidebar.subheader("🔧 Refrigerant (Compressor Specs)")
    
    designer = DXHeatExchangerDesign()
    inputs["refrigerant"] = st.sidebar.selectbox(
        "Refrigerant Type",
        list(designer.REFRIGERANTS.keys())
    )
    
    inputs["m_dot_ref"] = st.sidebar.number_input(
        "Refrigerant Mass Flow (kg/hr)",
        min_value=10.0, max_value=10000.0, value=500.0, step=10.0,
        help="From compressor specification sheet"
    )
    
    if inputs["hex_type"] == "evaporator":
        inputs["T_ref"] = st.sidebar.number_input(
            "Evaporating Temperature (°C)",
            min_value=-50.0, max_value=20.0, value=5.0, step=1.0
        )
        inputs["delta_T_sh_sc"] = st.sidebar.number_input(
            "Superheat at Exit (K)",
            min_value=3.0, max_value=15.0, value=5.0, step=0.5,
            help="DX evaporators require 3-8K superheat for TXV operation"
        )
        
        # DX-specific parameters
        with st.sidebar.expander("⚡ DX-Specific Parameters"):
            inputs["quality_inlet"] = st.slider(
                "Quality after Expansion Valve",
                min_value=0.1, max_value=0.3, value=0.2, step=0.05,
                help="Typical quality after TXV or electronic expansion valve"
            )
            
            inputs["distribution_type"] = st.selectbox(
                "Refrigerant Distribution",
                ["Standard Distributor", "Enhanced Distributor", "Individual TXVs"],
                help="Critical for DX evaporator performance"
            )
            
            st.info("""
            **DX Design Considerations:**
            1. Refrigerant distribution is critical
            2. Ensure minimum flow per tube for good distribution
            3. Superheat should be 3-8K for TXV operation
            4. Oil return must be considered
            """)
    else:
        inputs["T_ref"] = st.sidebar.number_input(
            "Condensing Temperature (°C)",
            min_value=20.0, max_value=80.0, value=45.0, step=1.0
        )
        inputs["delta_T_sh_sc"] = st.sidebar.number_input(
            "Subcool at Exit (K)",
            min_value=0.0, max_value=20.0, value=5.0, step=0.5
        )
    
    # ... [rest of input section similar to before]
    
    return inputs

def display_dx_specific_results(results: Dict, inputs: Dict):
    """Display DX-specific results and warnings"""
    
    # DX-specific header
    if inputs.get("hex_type") == "evaporator":
        st.markdown(f"""
        <div style='display: flex; align-items: center;'>
            <h2>📊 DX Evaporator Design Results</h2>
            <span class='evaporator-type-badge dx-badge'>DX Type</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='display: flex; align-items: center;'>
            <h2>📊 Condenser Design Results</h2>
            <span class='evaporator-type-badge' style='background-color: #8B5CF6; color: white;'>Condenser</span>
        </div>
        """, unsafe_allow_html=True)
    
    # DX-specific warnings
    if inputs.get("hex_type") == "evaporator":
        if results.get("distribution_status") == "Poor":
            st.error("""
            ⚠️ **POOR REFRIGERANT DISTRIBUTION**
            
            **Issue:** Flow per tube ({:.1f} kg/hr) is below recommended minimum ({:.1f} kg/hr)
            
            **Consequences:**
            - Uneven cooling capacity
            - Some tubes may flood while others starve
            - Reduced overall efficiency
            - Potential freeze risk in water circuit
            
            **Solutions:**
            1. **Increase refrigerant flow** (if compressor allows)
            2. **Reduce number of tubes** 
            3. **Use enhanced distributor**
            4. **Consider individual TXVs per circuit**
            """.format(
                results.get("flow_per_tube_kg_hr", 0),
                results.get("recommended_min_flow_per_tube_kg_hr", 3.6)
            ))
        
        if results.get("superheat_uniformity", 1) < 0.3:
            st.warning("""
            ⚠️ **LOW SUPERHEAT UNIFORMITY**
            
            **Issue:** Superheat varies significantly between tubes
            
            **Causes:**
            - Poor refrigerant distribution
            - Unequal circuit lengths
            - Different pressure drops
            
            **Solutions:**
            1. Improve distributor design
            2. Balance circuit lengths
            3. Use individual superheat controls
            """)
        
        if results.get("dryout_risk") == "High":
            st.warning("""
            ⚠️ **HIGH DRYOUT RISK**
            
            **Issue:** Refrigerant may completely evaporate too early in tubes
            
            **Consequences:**
            - Reduced heat transfer in dry region
            - Tube wall temperature increase
            - Potential oil logging
            
            **Solutions:**
            1. Increase refrigerant flow
            2. Reduce heat load per tube
            3. Consider recirculation design
            """)
        
        # DX design best practices
        with st.expander("🔧 DX Evaporator Design Best Practices"):
            st.markdown("""
            **1. Refrigerant Distribution:**
            - Minimum 3-5 kg/hr per tube for good distribution
            - Use properly sized distributor
            - Consider individual TXVs for large systems
            
            **2. Tube Circuiting:**
            - Balance circuit lengths
            - Similar pressure drop in parallel circuits
            - Consider U-tube or hairpin designs
            
            **3. Superheat Control:**
            - Maintain 3-8K superheat for TXV operation
            - Position bulb correctly
            - Consider electronic expansion valves
            
            **4. Oil Management:**
            - Ensure oil return in suction line
            - Consider oil separators for large systems
            - Proper piping slopes
            
            **5. Freeze Protection:**
            - Water velocity > 0.3 m/s
            - Proper control sequencing
            - Low ambient operation considerations
            """)
    
    # ... [rest of display function]

def main():
    """Main application with clear DX focus"""
    
    if not check_password():
        st.stop()
    
    st.markdown("<h1 class='main-header'>🌡️ Shell & Tube DX Evaporator & Condenser Designer</h1>", unsafe_allow_html=True)
    st.markdown("### Direct Expansion (DX) Type | Refrigerant in Tubes | Water in Shell")
    
    # Important note about flooded evaporators
    st.warning("""
    **⚠️ IMPORTANT: This tool designs DX (Direct Expansion) evaporators only.**
    
    **DX Evaporator Characteristics:**
    - ✅ **Refrigerant flows inside tubes** (evaporates as it flows)
    - ✅ **Water/Glycol flows in shell side**
    - ✅ **Uses thermostatic or electronic expansion valve**
    - ✅ **Lower refrigerant charge**
    - ✅ **Common in chillers, AC systems, smaller applications**
    
    **For Flooded Evaporators** (water in tubes, refrigerant boiling in shell), 
    please use the separate **Flooded Evaporator Designer** application.
    
    **Not sure which type you need?**
    - **DX**: Most common, simpler control, lower charge
    - **Flooded**: Higher efficiency, better oil management, larger systems
    """)
    
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    # Create layout
    col1, col2 = st.columns([3, 1])
    
    with col2:
        inputs = create_input_section()
        
        # Calculate button
        if st.sidebar.button("🚀 Calculate DX Design", type="primary", use_container_width=True):
            with st.spinner("Performing DX evaporator calculations..."):
                designer = DXHeatExchangerDesign()
                results = designer.design_heat_exchanger(inputs)
                st.session_state.results = results
                st.session_state.inputs = inputs
                st.rerun()
        
        # Link to flooded evaporator designer
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Need Flooded Evaporator Design?**")
        if st.sidebar.button("🔄 Go to Flooded Evaporator Designer", use_container_width=True):
            st.info("""
            **Flooded Evaporator Designer** will be a separate application.
            
            **Key differences:**
            - Water flows in tubes
            - Refrigerant boils in shell
            - Different heat transfer correlations
            - Liquid level control
            - Higher refrigerant charge
            
            **Contact developer for access to flooded evaporator design tool.**
            """)
    
    with col1:
        if st.session_state.results is not None:
            display_dx_specific_results(st.session_state.results, st.session_state.inputs)
            # ... [rest of display]
        else:
            st.markdown("""
            ## 🔧 DX Evaporator Design Focus
            
            **This tool is optimized for DX (Direct Expansion) evaporators:**
            
            ### **Key DX-Specific Features:**
            
            1. **Refrigerant Distribution Analysis**
               - Checks minimum flow per tube
               - Distribution header sizing
               - Maldistribution warnings
            
            2. **DX-Specific Heat Transfer**
               - Shah correlation for flow boiling in tubes
               - Acceleration pressure drop included
               - Superheat uniformity calculation
            
            3. **Control Considerations**
               - TXV superheat requirements (3-8K)
               - Electronic expansion valve options
               - Freeze protection warnings
            
            4. **Oil Management**
               - Oil circulation considerations
               - Return line sizing
               - Separator recommendations
            
            ### **Typical DX Evaporator Applications:**
            - Water chillers (5-500 kW)
            - Air conditioning systems
            - Process cooling
            - Heat pumps
            
            ### **When to Use DX vs Flooded:**
            
            | Parameter | DX Evaporator | Flooded Evaporator |
            |-----------|---------------|-------------------|
            | **System Size** | Small to medium | Medium to large |
            | **Efficiency** | Good | Better |
            | **Charge Amount** | Lower | Higher |
            | **Control** | Simpler | More complex |
            | **Freeze Risk** | Higher | Lower |
            | **Oil Return** | More critical | Easier |
            
            ### **Password Protected**
            Enter password: **Semaanju**
            """)
    
    # Footer with clear distinction
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🔧 <strong>DX Evaporator & Condenser Designer</strong> | Refrigerant in Tubes | Water in Shell</p>
        <p>⚠️ For flooded evaporators (water in tubes, refrigerant in shell), use separate tool</p>
        <p>🎯 Optimized for DX systems with proper refrigerant distribution and superheat control</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
