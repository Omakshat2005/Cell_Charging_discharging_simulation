import streamlit as st
import pandas as pd
import numpy as np
import random
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
st.set_page_config(
    page_title="⚡ Battery Dashboard",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stMetric > div > div > div > div {
        color: white !important;
    }
    
    .charging-card {
        background: linear-gradient(135deg, #10b981, #34d399);
        padding: 1rem;
        border-radius: 15px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }
    
    .discharging-card {
        background: linear-gradient(135deg, #ef4444, #f87171);
        padding: 1rem;
        border-radius: 15px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);
    }
    
    .idle-card {
        background: linear-gradient(135deg, #6b7280, #9ca3af);
        padding: 1rem;
        border-radius: 15px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(107, 114, 128, 0.3);
    }
    
    .voltage-normal {
        color: #3b82f6;
        font-weight: bold;
    }
    
    .voltage-low {
        color: #dc2626;
        font-weight: bold;
    }
    
    .voltage-high {
        color: #f59e0b;
        font-weight: bold;
    }
    
    .stSelectbox > div > div > div {
        background-color: #f8fafc;
    }
    
    .header-gradient {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

if 'cells_data' not in st.session_state:
    st.session_state.cells_data = {}
if 'num_cells' not in st.session_state:
    st.session_state.num_cells = 8
if 'auto_update' not in st.session_state:
    st.session_state.auto_update = False
if 'voltage_animations' not in st.session_state:
    st.session_state.voltage_animations = {}

def get_voltage_class(voltage, min_voltage, max_voltage):
    """Determine voltage status class"""
    if voltage <= min_voltage:
        return "voltage-low"
    elif voltage >= max_voltage:
        return "voltage-high"
    else:
        return "voltage-normal"

def get_voltage_color(voltage, min_voltage, max_voltage):
    """Get color for voltage based on range"""
    if voltage <= min_voltage:
        return "#dc2626"  # Red
    elif voltage >= max_voltage:
        return "#f59e0b"  # Yellow
    else:
        return "#3b82f6"  # Blue

def simulate_voltage_change(cell_index, cell_type, mode, current_voltage=None):
    """Simulate voltage changes based on mode"""
    base_voltage = 3.2 if cell_type == 'lfp' else 3.6
    max_voltage = 3.6 if cell_type == 'lfp' else 4.0
    min_voltage = 2.8 if cell_type == 'lfp' else 3.2
    
    if current_voltage is None:
        current_voltage = base_voltage
    
    if mode == 'charging':
        new_voltage = min(current_voltage + 0.02, max_voltage)
        if new_voltage >= max_voltage:
            new_voltage = base_voltage  # Reset to base when max reached
    elif mode == 'discharging':
        new_voltage = max(current_voltage - 0.02, min_voltage)
        if new_voltage <= min_voltage:
            new_voltage = base_voltage  # Reset to base when min reached
    else:  # idle
        new_voltage = base_voltage
    
    return round(new_voltage, 2)

def create_voltage_gauge(voltage, min_voltage, max_voltage, cell_name):
    """Create a voltage gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = voltage,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"{cell_name} Voltage"},
        delta = {'reference': (min_voltage + max_voltage) / 2},
        gauge = {
            'axis': {'range': [None, max_voltage * 1.1]},
            'bar': {'color': get_voltage_color(voltage, min_voltage, max_voltage)},
            'steps': [
                {'range': [0, min_voltage], 'color': "lightgray"},
                {'range': [min_voltage, max_voltage], 'color': "lightblue"},
                {'range': [max_voltage, max_voltage * 1.1], 'color': "lightyellow"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_voltage
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        font={'size': 12}
    )
    
    return fig

def create_battery_overview_chart(cells_data):
    """Create overview charts for all batteries"""
    if not cells_data:
        return None, None
    
  
    cell_names = list(cells_data.keys())
    voltages = [data['voltage'] for data in cells_data.values()]
    currents = [data['current'] for data in cells_data.values()]
    temperatures = [data['temp'] for data in cells_data.values()]
    capacities = [data['capacity'] for data in cells_data.values()]
    modes = [data['mode'] for data in cells_data.values()]
    
  
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Voltage Distribution', 'Current vs Capacity', 
                       'Temperature Distribution', 'Mode Distribution'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "pie"}]]
    )
    
    
    colors = [get_voltage_color(v, cells_data[name]['minVoltage'], cells_data[name]['maxVoltage']) 
              for v, name in zip(voltages, cell_names)]
    
    fig.add_trace(
        go.Bar(x=cell_names, y=voltages, name="Voltage", marker_color=colors),
        row=1, col=1
    )
    
  
    mode_colors = {'charging': '#10b981', 'discharging': '#ef4444', 'idle': '#6b7280'}
    scatter_colors = [mode_colors.get(mode, '#3b82f6') for mode in modes]
    
    fig.add_trace(
        go.Scatter(x=currents, y=capacities, mode='markers', name="Current vs Capacity",
                  marker=dict(size=10, color=scatter_colors), text=cell_names),
        row=1, col=2
    )
    
 
    fig.add_trace(
        go.Histogram(x=temperatures, name="Temperature", marker_color='#f59e0b', nbinsx=10),
        row=2, col=1
    )
    
   
    mode_counts = pd.Series(modes).value_counts()
    fig.add_trace(
        go.Pie(labels=mode_counts.index, values=mode_counts.values, name="Modes",
               marker_colors=[mode_colors.get(mode, '#3b82f6') for mode in mode_counts.index]),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="Battery System Overview",
        title_x=0.5
    )
    
    return fig


st.markdown('<h1 class="header-gradient">⚡ Battery Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #64748b;">Advanced Battery Cell Monitoring & Control System</p>', unsafe_allow_html=True)


with st.sidebar:
    st.header("🔧 Control Panel")
    
    
    num_cells = st.number_input(
        "🔋 Number of Cells", 
        min_value=1, 
        max_value=20, 
        value=st.session_state.num_cells,
        help="Select the number of battery cells to monitor"
    )
    st.session_state.num_cells = num_cells
    
   
    auto_update = st.checkbox(
        "🔄 Auto Update", 
        value=st.session_state.auto_update,
        help="Automatically update voltage simulations"
    )
    st.session_state.auto_update = auto_update
    
    
    if auto_update:
        update_interval = st.slider(
            "Update Interval (seconds)", 
            min_value=1, 
            max_value=10, 
            value=2
        )
    
    st.divider()
    

    if st.button("🔄 Reset All Cells", use_container_width=True):
        st.session_state.cells_data = {}
        st.session_state.voltage_animations = {}
        st.rerun()
    
    if st.button("⚡ Generate Data", use_container_width=True):
        # Generate data for all cells
        cells_data = {}
        for i in range(1, num_cells + 1):
            cell_key = f"cell_{i}"
            if cell_key in st.session_state:
                cell_config = st.session_state[cell_key]
                cell_type = cell_config.get('type', 'lfp')
                current = cell_config.get('current', 1.5)
                mode = cell_config.get('mode', 'charging')
                
                # Get current voltage or set base voltage
                current_voltage = st.session_state.voltage_animations.get(cell_key, 
                    3.2 if cell_type == 'lfp' else 3.6)
                
                voltage = simulate_voltage_change(i, cell_type, mode, current_voltage)
                st.session_state.voltage_animations[cell_key] = voltage
                
                max_voltage = 3.6 if cell_type == 'lfp' else 4.0
                min_voltage = 2.8 if cell_type == 'lfp' else 3.2
                temp = round(random.uniform(25, 40), 1)
                capacity = round(voltage * current, 2) if mode != 'idle' else 0.0
                
                cell_name = f"cell_{i}_{cell_type}"
                cells_data[cell_name] = {
                    'voltage': voltage,
                    'current': current if mode != 'idle' else 0,
                    'temp': temp,
                    'capacity': capacity,
                    'maxVoltage': max_voltage,
                    'minVoltage': min_voltage,
                    'mode': mode
                }
        
        st.session_state.cells_data = cells_data
        st.success(f"Generated data for {num_cells} cells!")
        st.rerun()


col1, col2 = st.columns([2, 1])

with col1:
    st.header("🔋 Battery Cells Configuration")
    
    # Create cell configuration in a grid
    cols_per_row = 2
    cell_rows = [list(range(i, min(i + cols_per_row, num_cells + 1))) 
                 for i in range(1, num_cells + 1, cols_per_row)]
    
    for row in cell_rows:
        cols = st.columns(len(row))
        for idx, cell_num in enumerate(row):
            with cols[idx]:
                cell_key = f"cell_{cell_num}"
                
              
                if cell_key not in st.session_state:
                    st.session_state[cell_key] = {
                        'type': 'lfp',
                        'current': 1.5,
                        'mode': 'charging'
                    }
                
            
                cell_config = st.session_state[cell_key]
                
 
                mode = cell_config.get('mode', 'charging')
                if mode == 'charging':
                    card_class = "charging-card"
                    mode_icon = "⚡"
                elif mode == 'discharging':
                    card_class = "discharging-card"
                    mode_icon = "⬇️"
                else:
                    card_class = "idle-card"
                    mode_icon = "⏸️"
                
                st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
                st.markdown(f"### {mode_icon} Cell {cell_num}")
                
               
                cell_type = st.selectbox(
                    "Type",
                    options=['lfp', 'nmc'],
                    index=0 if cell_config['type'] == 'lfp' else 1,
                    key=f"type_{cell_num}",
                    format_func=lambda x: f"🔋 {x.upper()}"
                )
                
         
                current_disabled = (mode == 'idle')
                current_value = 0.0 if current_disabled else cell_config.get('current', 1.5)
                
                current = st.number_input(
                    "Current (A)",
                    min_value=0.0,
                    max_value=10.0,
                    value=current_value,
                    step=0.1,
                    key=f"current_{cell_num}",
                    disabled=current_disabled,
                    help="Current is automatically set to 0 in idle mode"
                )
                
             
                mode = st.selectbox(
                    "Mode",
                    options=['charging', 'discharging', 'idle'],
                    index=['charging', 'discharging', 'idle'].index(cell_config['mode']),
                    key=f"mode_{cell_num}",
                    format_func=lambda x: f"⚡ {x.title()}" if x == 'charging' 
                                       else f"⬇️ {x.title()}" if x == 'discharging'
                                       else f"⏸️ {x.title()}"
                )
                
            
                st.session_state[cell_key] = {
                    'type': cell_type,
                    'current': current,
                    'mode': mode
                }
                
              
                if cell_key in st.session_state.voltage_animations:
                    voltage = st.session_state.voltage_animations[cell_key]
                    max_voltage = 3.6 if cell_type == 'lfp' else 4.0
                    min_voltage = 2.8 if cell_type == 'lfp' else 3.2
                    
                    voltage_class = get_voltage_class(voltage, min_voltage, max_voltage)
                    st.markdown(f'<p class="{voltage_class}">Current Voltage: {voltage}V</p>', 
                              unsafe_allow_html=True)
                    
                  
                    progress = (voltage - min_voltage) / (max_voltage - min_voltage)
                    st.progress(max(0, min(1, progress)))
                
                st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.header("📊 System Metrics")
    
    if st.session_state.cells_data:
        # Overall system metrics
        total_capacity = sum(data['capacity'] for data in st.session_state.cells_data.values())
        avg_voltage = np.mean([data['voltage'] for data in st.session_state.cells_data.values()])
        avg_temp = np.mean([data['temp'] for data in st.session_state.cells_data.values()])
        total_current = sum(data['current'] for data in st.session_state.cells_data.values())
        
        st.metric("Total Capacity", f"{total_capacity:.2f} Ah", 
                 help="Sum of all cell capacities")
        st.metric("Average Voltage", f"{avg_voltage:.2f} V",
                 help="Average voltage across all cells")
        st.metric("Average Temperature", f"{avg_temp:.1f} °C",
                 help="Average temperature across all cells")
        st.metric("Total Current", f"{total_current:.2f} A",
                 help="Sum of all cell currents")
        
        modes = [data['mode'] for data in st.session_state.cells_data.values()]
        mode_counts = pd.Series(modes).value_counts()
        
        st.subheader("🔄 Mode Distribution")
        for mode, count in mode_counts.items():
            icon = "⚡" if mode == 'charging' else "⬇️" if mode == 'discharging' else "⏸️"
            st.write(f"{icon} {mode.title()}: {count} cells")
    else:
        st.info("Generate data to see system metrics")


if auto_update and st.session_state.cells_data:
    time.sleep(update_interval)
    

    updated = False
    for cell_name, data in st.session_state.cells_data.items():
        cell_type = 'lfp' if 'lfp' in cell_name else 'nmc'
        mode = data['mode']
        current_voltage = data['voltage']
        
        new_voltage = simulate_voltage_change(0, cell_type, mode, current_voltage)
        if new_voltage != current_voltage:
            st.session_state.cells_data[cell_name]['voltage'] = new_voltage
            st.session_state.voltage_animations[f"cell_{cell_name.split('_')[1]}"] = new_voltage
            
           
            current = data['current']
            capacity = round(new_voltage * current, 2) if mode != 'idle' else 0.0
            st.session_state.cells_data[cell_name]['capacity'] = capacity
            updated = True
    
    if updated:
        st.rerun()


if st.session_state.cells_data:
    st.header("📈 Battery Data Overview")
    
    
    df_data = []
    for cell_name, data in st.session_state.cells_data.items():
        df_data.append({
            'Cell Name': cell_name,
            'Type': cell_name.split('_')[-1].upper(),
            'Voltage (V)': data['voltage'],
            'Current (A)': data['current'],
            'Temperature (°C)': data['temp'],
            'Capacity (Ah)': data['capacity'],
            'Min Voltage (V)': data['minVoltage'],
            'Max Voltage (V)': data['maxVoltage'],
            'Mode': data['mode'].title()
        })
    
    df = pd.DataFrame(df_data)
    
    
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Voltage (V)": st.column_config.NumberColumn(
                "Voltage (V)",
                format="%.2f V"
            ),
            "Current (A)": st.column_config.NumberColumn(
                "Current (A)",
                format="%.2f A"
            ),
            "Temperature (°C)": st.column_config.NumberColumn(
                "Temperature (°C)",
                format="%.1f °C"
            ),
            "Capacity (Ah)": st.column_config.NumberColumn(
                "Capacity (Ah)",
                format="%.2f Ah"
            ),
            "Mode": st.column_config.TextColumn(
                "Mode",
                width="small"
            )
        }
    )
    
  
    st.header("📊 Data Visualization")
    
    
    overview_fig = create_battery_overview_chart(st.session_state.cells_data)
    if overview_fig:
        st.plotly_chart(overview_fig, use_container_width=True)
    
   
    st.subheader("⚡ Individual Cell Voltages")
    gauge_cols = st.columns(min(3, len(st.session_state.cells_data)))
    
    for idx, (cell_name, data) in enumerate(st.session_state.cells_data.items()):
        col_idx = idx % len(gauge_cols)
        with gauge_cols[col_idx]:
            gauge_fig = create_voltage_gauge(
                data['voltage'], 
                data['minVoltage'], 
                data['maxVoltage'], 
                cell_name
            )
            st.plotly_chart(gauge_fig, use_container_width=True)
    
  
    st.header("💾 Export Data")
    

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.download_button(
            label="📊 Download CSV",
            data=csv_data,
            file_name=f"battery_dashboard_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

else:
    st.info("👆 Configure your battery cells and click 'Generate Data' to start monitoring!")


st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #64748b;">⚡ Battery Dashboard - Advanced Monitoring System</p>',
    unsafe_allow_html=True
)
