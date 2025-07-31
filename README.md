# Cell_Charging_discharging_simulation
An interactive Battery Cell Monitoring Dashboard built with Streamlit. It simulates multiple battery cells with charging and discharging modes, colorful UI, and real-time voltage, temperature, and capacity updates. Users can configure cell parameters, view live charts, switch themes, and export data as CSV easily.
Battery Cell Dashboard ⚡ (Streamlit App)
This project is an interactive and visually appealing Battery Cell Monitoring Dashboard built using Python (Streamlit). It simulates and displays real-time data for multiple battery cells with charging/discharging animations, colorful UI, and CSV export support.

Key Features
🔋 Dynamic cell management: Choose the number of cells and configure each cell's type (LFP/NMC) and current values.

⚡ Charging & discharging modes: Real-time voltage updates with color-coded cards:

Green (⚡ Charging)

Red (⬇️ Discharging)

📊 Live line charts: Each cell displays a mini voltage trend chart using Plotly/Matplotlib (or Streamlit charts).

🌈 Colorful, interactive UI: Gradient backgrounds, icons, and smooth animations.

📥 Export CSV: Download all cell data with a single click.

🌙 Dark mode support (optional).

📱 Responsive design: Works on all devices via the Streamlit web interface.

Tech Stack
Python: Core logic (battery simulation)

Streamlit: Interactive UI

Plotly/Matplotlib: Real-time charts

Pandas: Data storage & CSV export

How It Works
Install dependencies (see below).

Run the Streamlit app:

bash
Copy
Edit
streamlit run app.py
Open the link in your browser.

Choose the number of cells, configure their type and current, and toggle between Charging and Discharging.

Monitor live voltage, temperature, and capacity updates with charts.

Download the current data as a CSV file.

Installation
bash
Copy
Edit
git clone https://github.com/<your-username>/battery-cell-dashboard.git
cd battery-cell-dashboard
pip install -r requirements.txt
streamlit run app.py
