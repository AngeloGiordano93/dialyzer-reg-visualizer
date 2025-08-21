import streamlit as st
import numpy as np
import joblib
from matplotlib.figure import Figure

st.set_page_config(
    layout="wide",
    page_title="Dialyzer Clearance Visualizer",
    page_icon="📈"
)

# CONFIGURATION DICTIONARY FOR PARAMETER RANGES
RANGE_CONFIG = {
    'km': {
        'type': 'multiplicative',
        'min_factor': 0.1,
        'max_factor': 10,
        'max_clamp': 2e-5,
        'min_clamp': 5e-6
    },
    'theta': {
        'type': 'additive',
        'multiplier': 0.5, # More constrained variation
        'min_clamp': 3.0,
        'max_clamp': 12.0
    },
    'eps_d': {
        'type': 'additive',
        'multiplier': 0.5,
        'min_clamp': 0.3,
        'max_clamp': 0.5
    },
    'Qb': {
        'type': 'additive',
        'multiplier': 1.5,
        'min_clamp': 150.0,
        'max_clamp': 400.0,
    },
    'Qd': {
        'type': 'additive',
        'multiplier': 1.5,
        'min_clamp': 350.0,
        'max_clamp': 700.0
    },
    'Lp': {
        'type': 'additive',
        'multiplier': 1.5,
        'min_clamp': 10.0,
        'max_clamp': 260.0
    },
    'Pb': {
        'type': 'additive',
        'multiplier': 1.5,
        'min_clamp': -4500.0,
        'max_clamp': 1000.0
    },
    'Am': {
        'type': 'additive',
        'multiplier': 1.5,
        'min_clamp': 0.7,
        'max_clamp': 2.3
    }
}

# -----------------------------------------------------------------------------
# 1. MODEL AND SCALER LOADING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load("polyreg_d4_model.pkl")
        scaler = joblib.load("polyreg_d4_scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error("Error: 'polyreg_d4_model.pkl' and 'polyreg_d4_scaler.pkl' files not found. Please ensure they are in the same folder as the script.")
        return None, None

# Loads model and scaler on startup
poly_model, scaler = load_model_and_scaler()

if not poly_model or not scaler:
    st.stop() # Stops the script execution gracefully if the model or scaler is not found

# Parameter names (internal)
param_names = ['Qb', 'Qd', 'theta', 'Lp', 'eps_d', 'Pb', 'km', 'Am']

# Mapping dictionary for display names
PARAM_DISPLAY_MAP = {
    'Qb': 'Qb [ml/min]',
    'Qd': 'Qd [ml/min]',
    'theta': 'Aspect Ratio [-]',
    'Lp': 'Lp [ml/(m^2 h mmHg)]',
    'eps_d': 'Porosity [-]',
    'Pb': 'Pb [Pa]',
    'km': 'km [m/s]',
    'Am': 'Am [m^2]'
}

# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def predict_with_model(input_params_list, model, scaler_instance):
    """
    Uses the loaded model and scaler to generate a clearance prediction.
    """
    if model is None or scaler_instance is None:
        return 0

    input_array = np.array(input_params_list).reshape(1, -1)
    input_scaled = scaler_instance.transform(input_array)
    prediction = model.predict(input_scaled)
    return prediction[0]

def get_parameter_range(param_name, center_value):
    """Calculates the (min, max) range for a given parameter more robustly."""
    config = {'type': 'additive', 'multiplier': 1.5, 'min_clamp': 0.0}

    if param_name in RANGE_CONFIG:
        config.update(RANGE_CONFIG[param_name])

    if config['type'] == 'multiplicative':
        min_val = center_value * config.get('min_factor', 0.1)
        max_val = center_value * config.get('max_factor', 10)
    else: # Additive logic
        if center_value != 0:
            delta = abs(center_value * config.get('multiplier', 1.5))
        else:
            delta = 2.0

        min_delta = 0.1 if center_value > 1 else abs(center_value * 0.5) + 1e-6
        delta = max(delta, min_delta)

        min_val = center_value - delta
        max_val = center_value + delta

    if 'min_clamp' in config:
        min_val = max(min_val, config['min_clamp'])
    if 'max_clamp' in config:
        max_val = min(max_val, config['max_clamp'])

    if min_val >= max_val:
        min_val = max(0, center_value * 0.1)
        max_val = center_value * 10 if center_value != 0 else 1

    return min_val, max_val

def get_widget_config(value):
    """Returns an appropriate step and format string for a given value."""
    if value == 0:
        return 0.01, "%.4f"

    magnitude = 10**np.floor(np.log10(abs(value)))
    step = magnitude / 10

    if magnitude < 0.01:
        return step, "%.6f"
    else:
        return step, "%.2f"

# -----------------------------------------------------------------------------
# 3. GRAPH PLOTTING FUNCTION
# -----------------------------------------------------------------------------
def generate_plots_for_streamlit(current_params_values, selected_fixed_param_name, fixed_param_user_value):
    param_to_be_fixed_name = selected_fixed_param_name
    actual_fixed_value = fixed_param_user_value

    params_for_plotting = current_params_values.copy()
    params_for_plotting[param_to_be_fixed_name] = actual_fixed_value

    varying_param_names = [name for name in param_names if name != param_to_be_fixed_name]
    num_points_per_plot = 30

    # --- PHASE 1: DATA CALCULATION AND GLOBAL MIN/MAX ---
    plots_data = {}
    global_min_y = float('inf')
    global_max_y = float('-inf')

    for param_to_vary_name in varying_param_names:
        center_value = params_for_plotting[param_to_vary_name]
        vary_min, vary_max = get_parameter_range(param_to_vary_name, center_value)

        p_vary_values_for_axis = np.linspace(vary_min, vary_max, num_points_per_plot)
        y_output_values = []

        for varying_val_point in p_vary_values_for_axis:
            temp_params_for_calculation = params_for_plotting.copy()
            temp_params_for_calculation[param_to_vary_name] = varying_val_point
            ordered_params = [temp_params_for_calculation[name] for name in param_names]
            y = predict_with_model(ordered_params, poly_model, scaler)
            y_output_values.append(y)

        if y_output_values:
            current_min = np.min(y_output_values)
            current_max = np.max(y_output_values)
            if current_min < global_min_y:
                global_min_y = current_min
            if current_max > global_max_y:
                global_max_y = current_max

        plots_data[param_to_vary_name] = (p_vary_values_for_axis, y_output_values)

    if global_min_y != float('inf') and global_max_y != float('-inf'):
        padding = (global_max_y - global_min_y) * 0.05
        final_min_y = global_min_y - padding
        final_max_y = global_max_y + padding
    else: # Fallback in case of no data
        final_min_y, final_max_y = 0, 1

    # --- PHASE 2: PLOTTING THE GRAPHS ---
    fig = Figure(figsize=(18, 16), dpi=100)
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.2)
    subplot_positions = [gs[0, 0], gs[0, 1], gs[0, 2], gs[1, 0], gs[1, 1], gs[1, 2], gs[2, 0]]

    for i, param_to_vary_name in enumerate(varying_param_names):
        ax = fig.add_subplot(subplot_positions[i])
        
        x_values, y_values = plots_data[param_to_vary_name]
        ax.plot(x_values, y_values)
        ax.set_ylim(final_min_y, final_max_y)

        display_name_varying = PARAM_DISPLAY_MAP.get(param_to_vary_name, param_to_vary_name)
        ax.set_xlabel(display_name_varying, fontweight='bold', fontsize=12)
        ax.set_ylabel("Clearance [ml/min]", fontweight='bold', fontsize=12)
        ax.grid(True)

    fig.set_layout_engine('constrained')
    return fig

# -----------------------------------------------------------------------------
# 4. SESSION STATE AND UI LAYOUT
# -----------------------------------------------------------------------------

# Initialize session state...
if 'current_params_values' not in st.session_state:
    st.session_state.current_params_values = {
        'Qb': 275.0, 'Qd': 500.0, 'theta': 7, 'Lp': 150.0,
        'eps_d': 0.4, 'Pb': -2000, 'km': 1e-5, 'Am': 1.6
    }
if 'selected_fixed_param_name' not in st.session_state:
    st.session_state.selected_fixed_param_name = param_names[0]
if 'fixed_param_user_value' not in st.session_state:
    st.session_state.fixed_param_user_value = float(st.session_state.current_params_values[st.session_state.selected_fixed_param_name])

# --- Sidebar Controls ---
st.sidebar.header("Control Panel")
st.sidebar.subheader("Parameter Values:")

for p_name in param_names:
    current_val = float(st.session_state.current_params_values[p_name])
    step, format_str = get_widget_config(current_val)

    new_val = st.sidebar.number_input(
        label=PARAM_DISPLAY_MAP.get(p_name, p_name), # Use the mapping dict for the label
        value=current_val,
        key=f"entry_{p_name}",
        step=step,
        format=format_str
    )
    if new_val != st.session_state.current_params_values[p_name]:
        st.session_state.current_params_values[p_name] = new_val
        if p_name == st.session_state.selected_fixed_param_name:
            st.session_state.fixed_param_user_value = new_val

def update_slider_value_on_param_select():
    newly_selected_param = st.session_state.selectbox_fixed_param
    st.session_state.fixed_param_user_value = float(st.session_state.current_params_values[newly_selected_param])

st.session_state.selected_fixed_param_name = st.sidebar.selectbox(
    "1. Select parameter to modify:",
    options=param_names,
    format_func=lambda name: PARAM_DISPLAY_MAP.get(name, name), # Use format_func for display
    index=param_names.index(st.session_state.selected_fixed_param_name),
    key='selectbox_fixed_param',
    on_change=update_slider_value_on_param_select
)

slider_center = float(st.session_state.fixed_param_user_value)
min_val_slider, max_val_slider = get_parameter_range(
    st.session_state.selected_fixed_param_name,
    slider_center
)
slider_step, _ = get_widget_config(slider_center) # We only need the step

st.session_state.fixed_param_user_value = st.sidebar.slider(
    "2. Set new fixed value:",
    min_value=min_val_slider,
    max_value=max_val_slider,
    value=slider_center,
    step=slider_step,
    key='slider_fixed_value'
)
st.session_state.current_params_values[st.session_state.selected_fixed_param_name] = st.session_state.fixed_param_user_value

if slider_step < 0.01:
    st.sidebar.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;Precise value: **{st.session_state.fixed_param_user_value:.6f}**")

st.sidebar.markdown("---")
st.sidebar.button("Update Charts", type="primary")

st.header("Dialyzer Clearance as a function of all parameters")

# --- AUTOMATIC VISUALIZATION LOGIC ---
if poly_model and scaler:
    fig_to_show = generate_plots_for_streamlit(
        st.session_state.current_params_values,
        st.session_state.selected_fixed_param_name,
        st.session_state.fixed_param_user_value
    )
    st.pyplot(fig_to_show)
else:
    st.warning("Waiting for models to load...")

st.info("See the Documentation page for more information regarding the parameters.")