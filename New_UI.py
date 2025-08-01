import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time
from datetime import datetime, timedelta
import io
import json

# Page configuration
st.set_page_config(
    page_title="Battery Cell Management System",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .page-header {
        font-size: 2rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    .task-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if 'cells' not in st.session_state:
        st.session_state.cells = []
    if 'tasks' not in st.session_state:
        st.session_state.tasks = []
    if 'real_time_data' not in st.session_state:
        st.session_state.real_time_data = {
            'timestamps': [],
            'voltages': {},
            'temperatures': {},
            'capacities': {},
            'currents': {}
        }
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
    if 'last_update' not in st.session_state:
        st.session_state.last_update = datetime.now()

# Cell parameter defaults based on type
CELL_DEFAULTS = {
    'lfp': {
        'voltage': 3.2,
        'current': 2.0,
        'temperature': 25.0,
        'capacity': 100.0,
        'min_voltage': 2.5,
        'max_voltage': 3.6
    },
    'nmc': {
        'voltage': 3.7,
        'current': 2.5,
        'temperature': 25.0,
        'capacity': 120.0,
        'min_voltage': 3.0,
        'max_voltage': 4.2
    }
}

# Task types and their parameters
TASK_TYPES = {
    'CC_CV': ['current', 'voltage', 'time'],
    'IDLE': ['time'],
    'CC_CD': ['current', 'voltage', 'time']
}

def validate_cell_input(cell_type, voltage, current, temperature, capacity, min_voltage, max_voltage):
    """Validate cell input parameters"""
    errors = []
    
    if voltage <= 0:
        errors.append("Voltage must be positive")
    if current <= 0:
        errors.append("Current must be positive")
    if temperature < -40 or temperature > 80:
        errors.append("Temperature must be between -40°C and 80°C")
    if capacity <= 0:
        errors.append("Capacity must be positive")
    if min_voltage >= max_voltage:
        errors.append("Minimum voltage must be less than maximum voltage")
    if voltage < min_voltage or voltage > max_voltage:
        errors.append("Voltage must be between min and max voltage")
    
    return errors

def validate_task_input(task_type, parameters):
    """Validate task input parameters"""
    errors = []
    
    if 'current' in parameters and parameters['current'] <= 0:
        errors.append("Current must be positive")
    if 'voltage' in parameters and parameters['voltage'] <= 0:
        errors.append("Voltage must be positive")
    if 'time' in parameters and parameters['time'] <= 0:
        errors.append("Time must be positive")
    
    return errors

def generate_real_time_data():
    """Generate simulated real-time data for cells"""
    if not st.session_state.cells:
        return
    
    current_time = datetime.now()
    st.session_state.real_time_data['timestamps'].append(current_time)
    
    # Keep only last 100 data points
    if len(st.session_state.real_time_data['timestamps']) > 100:
        st.session_state.real_time_data['timestamps'] = st.session_state.real_time_data['timestamps'][-100:]
    
    for i, cell in enumerate(st.session_state.cells):
        cell_id = f"Cell_{i+1}"
        
        # Initialize cell data if not exists
        if cell_id not in st.session_state.real_time_data['voltages']:
            st.session_state.real_time_data['voltages'][cell_id] = []
            st.session_state.real_time_data['temperatures'][cell_id] = []
            st.session_state.real_time_data['capacities'][cell_id] = []
            st.session_state.real_time_data['currents'][cell_id] = []
        
        # Simulate realistic variations
        base_voltage = cell['voltage']
        base_temp = cell['temperature']
        base_capacity = cell['capacity']
        base_current = cell['current']
        
        # Add some random variation
        voltage_variation = np.random.normal(0, 0.05)
        temp_variation = np.random.normal(0, 1.0)
        capacity_variation = np.random.normal(0, 2.0)
        current_variation = np.random.normal(0, 0.1)
        
        new_voltage = max(cell['min_voltage'], min(cell['max_voltage'], 
                         base_voltage + voltage_variation))
        new_temp = max(-40, min(80, base_temp + temp_variation))
        new_capacity = max(0, base_capacity + capacity_variation)
        new_current = max(0, base_current + current_variation)
        
        st.session_state.real_time_data['voltages'][cell_id].append(new_voltage)
        st.session_state.real_time_data['temperatures'][cell_id].append(new_temp)
        st.session_state.real_time_data['capacities'][cell_id].append(new_capacity)
        st.session_state.real_time_data['currents'][cell_id].append(new_current)
        
        # Keep only last 100 data points for each parameter
        for param in ['voltages', 'temperatures', 'capacities', 'currents']:
            if len(st.session_state.real_time_data[param][cell_id]) > 100:
                st.session_state.real_time_data[param][cell_id] = st.session_state.real_time_data[param][cell_id][-100:]

def page_1_setup_cells():
    """Page 1: Setup Cells"""
    st.markdown('<h2 class="page-header">🔋 Cell Setup & Configuration</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Add New Cell")
        
        with st.form("add_cell_form"):
            cell_type = st.selectbox("Cell Type", ["lfp", "nmc"], help="Choose between LFP and NMC cell types")
            
            # Auto-fill defaults based on cell type
            defaults = CELL_DEFAULTS[cell_type]
            
            voltage = st.number_input("Voltage (V)", value=defaults['voltage'], min_value=0.1, step=0.1)
            current = st.number_input("Current (A)", value=defaults['current'], min_value=0.1, step=0.1)
            temperature = st.number_input("Temperature (°C)", value=defaults['temperature'], min_value=-40.0, max_value=80.0, step=0.1)
            capacity = st.number_input("Capacity (Ah)", value=defaults['capacity'], min_value=0.1, step=0.1)
            min_voltage = st.number_input("Min Voltage (V)", value=defaults['min_voltage'], min_value=0.1, step=0.1)
            max_voltage = st.number_input("Max Voltage (V)", value=defaults['max_voltage'], min_value=0.1, step=0.1)
            
            submitted = st.form_submit_button("Add Cell", use_container_width=True)
            
            if submitted:
                errors = validate_cell_input(cell_type, voltage, current, temperature, capacity, min_voltage, max_voltage)
                
                if errors:
                    for error in errors:
                        st.markdown(f'<div class="error-message">❌ {error}</div>', unsafe_allow_html=True)
                else:
                    new_cell = {
                        'type': cell_type,
                        'voltage': voltage,
                        'current': current,
                        'temperature': temperature,
                        'capacity': capacity,
                        'min_voltage': min_voltage,
                        'max_voltage': max_voltage,
                        'created_at': datetime.now()
                    }
                    st.session_state.cells.append(new_cell)
                    st.markdown('<div class="success-message">✅ Cell added successfully!</div>', unsafe_allow_html=True)
                    st.rerun()
    
    with col2:
        st.markdown("### Current Cells")
        
        if st.session_state.cells:
            # Display metrics
            col_metrics = st.columns(4)
            with col_metrics[0]:
                st.metric("Total Cells", len(st.session_state.cells))
            with col_metrics[1]:
                lfp_count = sum(1 for cell in st.session_state.cells if cell['type'] == 'lfp')
                st.metric("LFP Cells", lfp_count)
            with col_metrics[2]:
                nmc_count = sum(1 for cell in st.session_state.cells if cell['type'] == 'nmc')
                st.metric("NMC Cells", nmc_count)
            with col_metrics[3]:
                avg_capacity = np.mean([cell['capacity'] for cell in st.session_state.cells])
                st.metric("Avg Capacity", f"{avg_capacity:.1f} Ah")
            
            # Display cells table
            cells_df = pd.DataFrame(st.session_state.cells)
            cells_df.index = [f"Cell_{i+1}" for i in range(len(cells_df))]
            
            st.dataframe(
                cells_df[['type', 'voltage', 'current', 'temperature', 'capacity', 'min_voltage', 'max_voltage']],
                use_container_width=True,
                column_config={
                    'type': st.column_config.SelectboxColumn('Type', options=['lfp', 'nmc']),
                    'voltage': st.column_config.NumberColumn('Voltage (V)', format="%.2f"),
                    'current': st.column_config.NumberColumn('Current (A)', format="%.2f"),
                    'temperature': st.column_config.NumberColumn('Temperature (°C)', format="%.1f"),
                    'capacity': st.column_config.NumberColumn('Capacity (Ah)', format="%.1f"),
                    'min_voltage': st.column_config.NumberColumn('Min V', format="%.2f"),
                    'max_voltage': st.column_config.NumberColumn('Max V', format="%.2f"),
                }
            )
            
            # Edit/Delete cells
            st.markdown("### Manage Cells")
            if st.button("Clear All Cells", type="secondary"):
                st.session_state.cells = []
                st.session_state.real_time_data = {
                    'timestamps': [],
                    'voltages': {},
                    'temperatures': {},
                    'capacities': {},
                    'currents': {}
                }
                st.rerun()
            
            # Individual cell management
            for i, cell in enumerate(st.session_state.cells):
                with st.expander(f"Cell_{i+1} ({cell['type'].upper()}) - {cell['voltage']}V"):
                    col_edit, col_delete = st.columns([3, 1])
                    with col_edit:
                        st.write(f"**Type:** {cell['type'].upper()}")
                        st.write(f"**Voltage:** {cell['voltage']}V")
                        st.write(f"**Current:** {cell['current']}A")
                        st.write(f"**Temperature:** {cell['temperature']}°C")
                        st.write(f"**Capacity:** {cell['capacity']}Ah")
                        st.write(f"**Voltage Range:** {cell['min_voltage']}V - {cell['max_voltage']}V")
                    with col_delete:
                        if st.button(f"Delete", key=f"delete_cell_{i}"):
                            st.session_state.cells.pop(i)
                            st.rerun()
        else:
            st.info("No cells configured yet. Add your first cell using the form on the left.")

def page_2_add_tasks():
    """Page 2: Add Tasks"""
    st.markdown('<h2 class="page-header">⚡ Task Management</h2>', unsafe_allow_html=True)
    
    if not st.session_state.cells:
        st.warning("Please configure cells first in the Cell Setup page.")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Add New Task")
        
        with st.form("add_task_form"):
            task_type = st.selectbox("Task Type", list(TASK_TYPES.keys()))
            task_name = st.text_input("Task Name", value=f"{task_type}_{len(st.session_state.tasks)+1}")
            
            # Dynamic parameters based on task type
            parameters = {}
            required_params = TASK_TYPES[task_type]
            
            if 'current' in required_params:
                parameters['current'] = st.number_input("Current (A)", min_value=0.1, value=2.0, step=0.1)
            if 'voltage' in required_params:
                parameters['voltage'] = st.number_input("Voltage (V)", min_value=0.1, value=3.5, step=0.1)
            if 'time' in required_params:
                parameters['time'] = st.number_input("Time (hours)", min_value=0.1, value=1.0, step=0.1)
            
            # Cell selection
            cell_options = [f"Cell_{i+1}" for i in range(len(st.session_state.cells))]
            selected_cells = st.multiselect("Assign to Cells", cell_options, default=cell_options[:1])
            
            priority = st.selectbox("Priority", ["Low", "Medium", "High"], index=1)
            
            submitted = st.form_submit_button("Add Task", use_container_width=True)
            
            if submitted:
                if not task_name.strip():
                    st.error("Task name cannot be empty")
                elif not selected_cells:
                    st.error("Please select at least one cell")
                else:
                    errors = validate_task_input(task_type, parameters)
                    
                    if errors:
                        for error in errors:
                            st.markdown(f'<div class="error-message">❌ {error}</div>', unsafe_allow_html=True)
                    else:
                        new_task = {
                            'name': task_name,
                            'type': task_type,
                            'parameters': parameters,
                            'assigned_cells': selected_cells,
                            'priority': priority,
                            'status': 'Pending',
                            'progress': 0,
                            'created_at': datetime.now()
                        }
                        st.session_state.tasks.append(new_task)
                        st.markdown('<div class="success-message">✅ Task added successfully!</div>', unsafe_allow_html=True)
                        st.rerun()
    
    with col2:
        st.markdown("### Current Tasks")
        
        if st.session_state.tasks:
            # Task metrics
            col_metrics = st.columns(4)
            with col_metrics[0]:
                st.metric("Total Tasks", len(st.session_state.tasks))
            with col_metrics[1]:
                pending_tasks = sum(1 for task in st.session_state.tasks if task['status'] == 'Pending')
                st.metric("Pending", pending_tasks)
            with col_metrics[2]:
                running_tasks = sum(1 for task in st.session_state.tasks if task['status'] == 'Running')
                st.metric("Running", running_tasks)
            with col_metrics[3]:
                completed_tasks = sum(1 for task in st.session_state.tasks if task['status'] == 'Completed')
                st.metric("Completed", completed_tasks)
            
            # Tasks table
            for i, task in enumerate(st.session_state.tasks):
                with st.expander(f"📋 {task['name']} ({task['type']})"):
                    col_info, col_actions = st.columns([3, 1])
                    
                    with col_info:
                        st.write(f"**Type:** {task['type']}")
                        st.write(f"**Priority:** {task['priority']}")
                        st.write(f"**Status:** {task['status']}")
                        st.write(f"**Assigned Cells:** {', '.join(task['assigned_cells'])}")
                        st.write(f"**Parameters:** {task['parameters']}")
                        
                        # Progress bar
                        progress = task.get('progress', 0)
                        st.progress(progress / 100, text=f"Progress: {progress}%")
                    
                    with col_actions:
                        if st.button("Start", key=f"start_task_{i}", disabled=(task['status'] == 'Running')):
                            st.session_state.tasks[i]['status'] = 'Running'
                            st.rerun()
                        
                        if st.button("Stop", key=f"stop_task_{i}", disabled=(task['status'] != 'Running')):
                            st.session_state.tasks[i]['status'] = 'Pending'
                            st.rerun()
                        
                        if st.button("Delete", key=f"delete_task_{i}"):
                            st.session_state.tasks.pop(i)
                            st.rerun()
            
            if st.button("Clear All Tasks", type="secondary"):
                st.session_state.tasks = []
                st.rerun()
        else:
            st.info("No tasks configured yet. Add your first task using the form on the left.")

def page_3_real_time_dashboard():
    """Page 3: Real-time Analysis Dashboard"""
    st.markdown('<h2 class="page-header">📊 Real-Time Analysis Dashboard</h2>', unsafe_allow_html=True)
    
    if not st.session_state.cells:
        st.warning("Please configure cells first in the Cell Setup page.")
        return
    
    # Control panel
    col_control1, col_control2, col_control3 = st.columns(3)
    
    with col_control1:
        if st.button("Start Simulation", type="primary" if not st.session_state.simulation_running else "secondary"):
            st.session_state.simulation_running = True
    
    with col_control2:
        if st.button("Stop Simulation", type="secondary"):
            st.session_state.simulation_running = False
    
    with col_control3:
        if st.button("Clear Data", type="secondary"):
            st.session_state.real_time_data = {
                'timestamps': [],
                'voltages': {},
                'temperatures': {},
                'capacities': {},
                'currents': {}
            }
    
    # Generate real-time data if simulation is running
    if st.session_state.simulation_running:
        generate_real_time_data()
        time.sleep(1)  # Small delay for realistic updates
        st.rerun()
    
    # Status indicators
    col_status = st.columns(len(st.session_state.cells))
    for i, cell in enumerate(st.session_state.cells):
        with col_status[i]:
            cell_id = f"Cell_{i+1}"
            status = "🟢 Active" if st.session_state.simulation_running else "🔴 Inactive"
            st.metric(f"Cell {i+1}", status)
    
    if st.session_state.real_time_data['timestamps']:
        # Real-time charts
        timestamps = st.session_state.real_time_data['timestamps']
        
        # Voltage Chart
        st.markdown("### 🔋 Voltage Monitoring")
        voltage_fig = go.Figure()
        for cell_id, voltages in st.session_state.real_time_data['voltages'].items():
            if voltages:
                voltage_fig.add_trace(go.Scatter(
                    x=timestamps[-len(voltages):],
                    y=voltages,
                    mode='lines+markers',
                    name=cell_id,
                    line=dict(width=3)
                ))
        
        voltage_fig.update_layout(
            title="Real-Time Voltage Monitoring",
            xaxis_title="Time",
            yaxis_title="Voltage (V)",
            height=400,
            template="plotly_dark"
        )
        st.plotly_chart(voltage_fig, use_container_width=True)
        
        # Temperature Chart
        st.markdown("### 🌡️ Temperature Monitoring")
        temp_fig = go.Figure()
        for cell_id, temperatures in st.session_state.real_time_data['temperatures'].items():
            if temperatures:
                temp_fig.add_trace(go.Scatter(
                    x=timestamps[-len(temperatures):],
                    y=temperatures,
                    mode='lines+markers',
                    name=cell_id,
                    line=dict(width=3)
                ))
        
        temp_fig.update_layout(
            title="Real-Time Temperature Monitoring",
            xaxis_title="Time",
            yaxis_title="Temperature (°C)",
            height=400,
            template="plotly_dark"
        )
        st.plotly_chart(temp_fig, use_container_width=True)
        
        # Combined Dashboard
        st.markdown("### 📈 Multi-Parameter Dashboard")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Voltage (V)', 'Temperature (°C)', 'Capacity (Ah)', 'Current (A)'],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # Add traces for each cell
        for cell_id in st.session_state.real_time_data['voltages'].keys():
            if st.session_state.real_time_data['voltages'][cell_id]:
                # Voltage
                fig.add_trace(
                    go.Scatter(
                        x=timestamps[-len(st.session_state.real_time_data['voltages'][cell_id]):],
                        y=st.session_state.real_time_data['voltages'][cell_id],
                        mode='lines',
                        name=f"{cell_id}_V",
                        showlegend=True
                    ),
                    row=1, col=1
                )
                
                # Temperature
                fig.add_trace(
                    go.Scatter(
                        x=timestamps[-len(st.session_state.real_time_data['temperatures'][cell_id]):],
                        y=st.session_state.real_time_data['temperatures'][cell_id],
                        mode='lines',
                        name=f"{cell_id}_T",
                        showlegend=False
                    ),
                    row=1, col=2
                )
                
                # Capacity
                fig.add_trace(
                    go.Scatter(
                        x=timestamps[-len(st.session_state.real_time_data['capacities'][cell_id]):],
                        y=st.session_state.real_time_data['capacities'][cell_id],
                        mode='lines',
                        name=f"{cell_id}_C",
                        showlegend=False
                    ),
                    row=2, col=1
                )
                
                # Current
                fig.add_trace(
                    go.Scatter(
                        x=timestamps[-len(st.session_state.real_time_data['currents'][cell_id]):],
                        y=st.session_state.real_time_data['currents'][cell_id],
                        mode='lines',
                        name=f"{cell_id}_I",
                        showlegend=False
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(height=600, template="plotly_dark", title_text="Multi-Parameter Real-Time Dashboard")
        st.plotly_chart(fig, use_container_width=True)
        
        # Current values table
        st.markdown("### 📋 Current Values")
        current_data = []
        for i, cell in enumerate(st.session_state.cells):
            cell_id = f"Cell_{i+1}"
            if cell_id in st.session_state.real_time_data['voltages'] and st.session_state.real_time_data['voltages'][cell_id]:
                current_data.append({
                    'Cell': cell_id,
                    'Type': cell['type'].upper(),
                    'Voltage (V)': f"{st.session_state.real_time_data['voltages'][cell_id][-1]:.3f}",
                    'Temperature (°C)': f"{st.session_state.real_time_data['temperatures'][cell_id][-1]:.1f}",
                    'Capacity (Ah)': f"{st.session_state.real_time_data['capacities'][cell_id][-1]:.1f}",
                    'Current (A)': f"{st.session_state.real_time_data['currents'][cell_id][-1]:.2f}"
                })
        
        if current_data:
            st.dataframe(pd.DataFrame(current_data), use_container_width=True)
    else:
        st.info("Start the simulation to see real-time data visualization.")

def page_4_data_export():
    """Page 4: Data Download/Export"""
    st.markdown('<h2 class="page-header">💾 Data Export & Analysis</h2>', unsafe_allow_html=True)
    
    # Summary Statistics
    st.markdown("### 📊 Summary Statistics")
    
    if st.session_state.cells:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Cell Summary")
            cells_summary = {
                'Total Cells': len(st.session_state.cells),
                'LFP Cells': sum(1 for cell in st.session_state.cells if cell['type'] == 'lfp'),
                'NMC Cells': sum(1 for cell in st.session_state.cells if cell['type'] == 'nmc'),
                'Average Voltage': f"{np.mean([cell['voltage'] for cell in st.session_state.cells]):.2f} V",
                'Average Capacity': f"{np.mean([cell['capacity'] for cell in st.session_state.cells]):.1f} Ah",
                'Total Capacity': f"{sum([cell['capacity'] for cell in st.session_state.cells]):.1f} Ah"
            }
            
            for key, value in cells_summary.items():
                st.metric(key, value)
        
        with col2:
            st.markdown("#### Task Summary")
            if st.session_state.tasks:
                tasks_summary = {
                    'Total Tasks': len(st.session_state.tasks),
                    'Pending Tasks': sum(1 for task in st.session_state.tasks if task['status'] == 'Pending'),
                    'Running Tasks': sum(1 for task in st.session_state.tasks if task['status'] == 'Running'),
                    'Completed Tasks': sum(1 for task in st.session_state.tasks if task['status'] == 'Completed'),
                    'CC_CV Tasks': sum(1 for task in st.session_state.tasks if task['type'] == 'CC_CV'),
                    'IDLE Tasks': sum(1 for task in st.session_state.tasks if task['type'] == 'IDLE')
                }
                
                for key, value in tasks_summary.items():
                    st.metric(key, value)
            else:
                st.info("No tasks to summarize")
    
    # Data Tables
    st.markdown("### 📋 Data Tables")
    
    tab1, tab2, tab3 = st.tabs(["Cell Data", "Task Data", "Real-Time Data"])
    
    with tab1:
        if st.session_state.cells:
            cells_df = pd.DataFrame(st.session_state.cells)
            cells_df.index = [f"Cell_{i+1}" for i in range(len(cells_df))]
            st.dataframe(cells_df, use_container_width=True)
        else:
            st.info("No cell data available")
    
    with tab2:
        if st.session_state.tasks:
            tasks_df = pd.DataFrame(st.session_state.tasks)
            # Convert parameters dict to string for display
            tasks_df['parameters_str'] = tasks_df['parameters'].apply(lambda x: str(x))
            tasks_df['assigned_cells_str'] = tasks_df['assigned_cells'].apply(lambda x: ', '.join(x))
            
            display_tasks_df = tasks_df[['name', 'type', 'status', 'priority', 'progress', 'parameters_str', 'assigned_cells_str']]
            display_tasks_df.columns = ['Name', 'Type', 'Status', 'Priority', 'Progress (%)', 'Parameters', 'Assigned Cells']
            st.dataframe(display_tasks_df, use_container_width=True)
        else:
            st.info("No task data available")
    
    with tab3:
        if st.session_state.real_time_data['timestamps']:
            st.markdown("#### Latest Real-Time Data Points")
            
            # Create a comprehensive real-time data table
            rt_data = []
            timestamps = st.session_state.real_time_data['timestamps']
            
            for i, cell in enumerate(st.session_state.cells):
                cell_id = f"Cell_{i+1}"
                if cell_id in st.session_state.real_time_data['voltages'] and st.session_state.real_time_data['voltages'][cell_id]:
                    for j, timestamp in enumerate(timestamps[-10:]):  # Last 10 data points
                        if j < len(st.session_state.real_time_data['voltages'][cell_id]):
                            rt_data.append({
                                'Cell': cell_id,
                                'Timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                                'Voltage (V)': f"{st.session_state.real_time_data['voltages'][cell_id][-(10-j)]:.3f}",
                                'Temperature (°C)': f"{st.session_state.real_time_data['temperatures'][cell_id][-(10-j)]:.1f}",
                                'Capacity (Ah)': f"{st.session_state.real_time_data['capacities'][cell_id][-(10-j)]:.1f}",
                                'Current (A)': f"{st.session_state.real_time_data['currents'][cell_id][-(10-j)]:.2f}"
                            })
            
            if rt_data:
                rt_df = pd.DataFrame(rt_data)
                st.dataframe(rt_df, use_container_width=True)
            else:
                st.info("No real-time data points available")
        else:
            st.info("No real-time data available. Start simulation in Dashboard to generate data.")
    
    # Export Options
    st.markdown("### 💾 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📁 Export Cell Data", use_container_width=True):
            if st.session_state.cells:
                cells_df = pd.DataFrame(st.session_state.cells)
                cells_df.index = [f"Cell_{i+1}" for i in range(len(cells_df))]
                
                # Convert to CSV
                csv_buffer = io.StringIO()
                cells_df.to_csv(csv_buffer, index=True)
                csv_data = csv_buffer.getvalue()
                
                st.download_button(
                    label="Download Cell Data CSV",
                    data=csv_data,
                    file_name=f"cell_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.warning("No cell data to export")
    
    with col2:
        if st.button("📋 Export Task Data", use_container_width=True):
            if st.session_state.tasks:
                tasks_df = pd.DataFrame(st.session_state.tasks)
                # Flatten the parameters and assigned_cells for CSV export
                tasks_export_df = tasks_df.copy()
                tasks_export_df['parameters_json'] = tasks_export_df['parameters'].apply(json.dumps)
                tasks_export_df['assigned_cells_list'] = tasks_export_df['assigned_cells'].apply(', '.join)
                
                # Select relevant columns for export
                export_columns = ['name', 'type', 'status', 'priority', 'progress', 'parameters_json', 'assigned_cells_list', 'created_at']
                tasks_export_df = tasks_export_df[export_columns]
                
                csv_buffer = io.StringIO()
                tasks_export_df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                
                st.download_button(
                    label="Download Task Data CSV",
                    data=csv_data,
                    file_name=f"task_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.warning("No task data to export")
    
    with col3:
        if st.button("📊 Export Real-Time Data", use_container_width=True):
            if st.session_state.real_time_data['timestamps']:
                # Prepare real-time data for export
                rt_export_data = []
                timestamps = st.session_state.real_time_data['timestamps']
                
                # Get all data points for all cells
                for i, timestamp in enumerate(timestamps):
                    for cell_id in st.session_state.real_time_data['voltages'].keys():
                        if (i < len(st.session_state.real_time_data['voltages'][cell_id]) and
                            i < len(st.session_state.real_time_data['temperatures'][cell_id]) and
                            i < len(st.session_state.real_time_data['capacities'][cell_id]) and
                            i < len(st.session_state.real_time_data['currents'][cell_id])):
                            
                            rt_export_data.append({
                                'Timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'),
                                'Cell_ID': cell_id,
                                'Voltage_V': st.session_state.real_time_data['voltages'][cell_id][i],
                                'Temperature_C': st.session_state.real_time_data['temperatures'][cell_id][i],
                                'Capacity_Ah': st.session_state.real_time_data['capacities'][cell_id][i],
                                'Current_A': st.session_state.real_time_data['currents'][cell_id][i]
                            })
                
                if rt_export_data:
                    rt_df = pd.DataFrame(rt_export_data)
                    csv_buffer = io.StringIO()
                    rt_df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    
                    st.download_button(
                        label="Download Real-Time Data CSV",
                        data=csv_data,
                        file_name=f"realtime_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.warning("No real-time data points to export")
            else:
                st.warning("No real-time data to export")
    
    # Combined Export
    st.markdown("### 📦 Combined Export")
    if st.button("🗂️ Export All Data (Combined)", use_container_width=True):
        if st.session_state.cells or st.session_state.tasks or st.session_state.real_time_data['timestamps']:
            # Create a comprehensive export with multiple sheets worth of data
            export_data = {
                'metadata': {
                    'export_timestamp': datetime.now().isoformat(),
                    'total_cells': len(st.session_state.cells),
                    'total_tasks': len(st.session_state.tasks),
                    'simulation_running': st.session_state.simulation_running
                },
                'cells': st.session_state.cells,
                'tasks': st.session_state.tasks,
                'real_time_data_summary': {
                    'total_data_points': len(st.session_state.real_time_data['timestamps']),
                    'data_range': {
                        'start': st.session_state.real_time_data['timestamps'][0].isoformat() if st.session_state.real_time_data['timestamps'] else None,
                        'end': st.session_state.real_time_data['timestamps'][-1].isoformat() if st.session_state.real_time_data['timestamps'] else None
                    }
                }
            }
            
            # Convert to JSON for comprehensive export
            json_data = json.dumps(export_data, indent=2, default=str)
            
            st.download_button(
                label="Download Complete Dataset (JSON)",
                data=json_data,
                file_name=f"battery_system_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        else:
            st.warning("No data available to export")
    
    # Data Analysis
    st.markdown("### 🔍 Quick Data Analysis")
    
    if st.session_state.real_time_data['timestamps']:
        analysis_col1, analysis_col2 = st.columns(2)
        
        with analysis_col1:
            st.markdown("#### Statistical Summary")
            
            # Calculate statistics for each cell
            for cell_id in st.session_state.real_time_data['voltages'].keys():
                if st.session_state.real_time_data['voltages'][cell_id]:
                    st.markdown(f"**{cell_id}:**")
                    
                    voltages = st.session_state.real_time_data['voltages'][cell_id]
                    temps = st.session_state.real_time_data['temperatures'][cell_id]
                    
                    col_v, col_t = st.columns(2)
                    with col_v:
                        st.write(f"Voltage: {np.mean(voltages):.3f}V ± {np.std(voltages):.3f}")
                        st.write(f"Range: {np.min(voltages):.3f} - {np.max(voltages):.3f}V")
                    with col_t:
                        st.write(f"Temperature: {np.mean(temps):.1f}°C ± {np.std(temps):.1f}")
                        st.write(f"Range: {np.min(temps):.1f} - {np.max(temps):.1f}°C")
        
        with analysis_col2:
            st.markdown("#### Trend Analysis")
            
            # Simple trend analysis
            for cell_id in st.session_state.real_time_data['voltages'].keys():
                if len(st.session_state.real_time_data['voltages'][cell_id]) > 1:
                    voltages = st.session_state.real_time_data['voltages'][cell_id]
                    temps = st.session_state.real_time_data['temperatures'][cell_id]
                    
                    # Calculate simple linear trends
                    voltage_trend = "📈" if voltages[-1] > voltages[0] else "📉"
                    temp_trend = "📈" if temps[-1] > temps[0] else "📉"
                    
                    st.write(f"**{cell_id}:**")
                    st.write(f"Voltage trend: {voltage_trend}")
                    st.write(f"Temperature trend: {temp_trend}")

def main():
    """Main application function"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🔋 Battery Cell Management System</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to:",
        ["🔋 Cell Setup", "⚡ Task Management", "📊 Real-Time Dashboard", "💾 Data Export"],
        index=0
    )
    
    # System status in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Status")
    st.sidebar.metric("Active Cells", len(st.session_state.cells))
    st.sidebar.metric("Active Tasks", len(st.session_state.tasks))
    
    simulation_status = "🟢 Running" if st.session_state.simulation_running else "🔴 Stopped"
    st.sidebar.metric("Simulation", simulation_status)
    
    if st.session_state.real_time_data['timestamps']:
        st.sidebar.metric("Data Points", len(st.session_state.real_time_data['timestamps']))
    
    # Quick actions in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Actions")
    
    if st.sidebar.button("🗑️ Reset All Data"):
        st.session_state.cells = []
        st.session_state.tasks = []
        st.session_state.real_time_data = {
            'timestamps': [],
            'voltages': {},
            'temperatures': {},
            'capacities': {},
            'currents': {}
        }
        st.session_state.simulation_running = False
        st.rerun()
    
    # Page routing
    if page == "🔋 Cell Setup":
        page_1_setup_cells()
    elif page == "⚡ Task Management":
        page_2_add_tasks()
    elif page == "📊 Real-Time Dashboard":
        page_3_real_time_dashboard()
    elif page == "💾 Data Export":
        page_4_data_export()
    
    # Footer
    st.markdown("---")
    st.markdown("### About")
    st.info("""
    **Battery Cell Management System** - A comprehensive tool for managing battery cells, tasks, and real-time monitoring.
    
    **Features:**
    - Multi-cell configuration with LFP/NMC support
    - Task management with CC_CV, IDLE, and CC_CD operations
    - Real-time monitoring and visualization
    - Data export capabilities
    - Interactive dashboard with live updates
    
    **Usage:** Navigate through the pages using the sidebar to configure cells, add tasks, monitor real-time data, and export results.
    """)

if __name__ == "__main__":
    main()
