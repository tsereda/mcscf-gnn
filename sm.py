import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide")
st.title("GNN Performance Dashboard")
st.markdown("---")

# --- 1. Data Generation (Cached & Unchanged) ---
@st.cache_data
def generate_all_fold_data():
    NUM_EPOCHS, NUM_FOLDS = 100, 4
    TASK_CONFIG = {
        'Node': {'initial': 2.5, 'final': 0.2, 'decay': 0.08, 'noise': 0.05},
        'Edge': {'initial': 3.0, 'final': 0.5, 'decay': 0.06, 'noise': 0.07},
        'Graph': {'initial': 1.5, 'final': 0.1, 'decay': 0.1, 'noise': 0.04}
    }
    all_curves = []
    epochs = np.arange(1, NUM_EPOCHS + 1)
    for fold_num in range(1, NUM_FOLDS + 1):
        for task, config in TASK_CONFIG.items():
            fold_factor = 1 + (np.random.rand() - 0.5) * 0.2
            train_noise = (np.random.rand(NUM_EPOCHS) - 0.5) * config['noise']
            train_loss = (config['final'] * fold_factor + (config['initial'] - config['final']) * np.exp(-config['decay'] * epochs) + train_noise)
            val_loss = train_loss + (0.1 * fold_factor) + (np.random.rand(NUM_EPOCHS) - 0.5) * config['noise']
            for epoch, tr_loss, val_loss_val in zip(epochs, train_loss, val_loss):
                all_curves.extend([
                    [epoch, tr_loss, 'Training', task, 'MSE', f'Fold {fold_num}'],
                    [epoch, val_loss_val, 'Validation', task, 'MSE', f'Fold {fold_num}'],
                    [epoch, tr_loss * 0.4, 'Training', task, 'MAE', f'Fold {fold_num}'],
                    [epoch, val_loss_val * 0.4, 'Validation', task, 'MAE', f'Fold {fold_num}']
                ])
    df_folds = pd.DataFrame(all_curves, columns=['Epoch', 'Loss', 'Type', 'Task', 'Metric', 'Fold'])
    df_avg = df_folds.groupby(['Epoch', 'Type', 'Task', 'Metric'])['Loss'].mean().reset_index()
    df_avg['Fold'] = 'Average'
    return pd.concat([df_avg, df_folds])

df_combined = generate_all_fold_data()

# --- 2. Simplified Controls ---
cols = st.columns(4)
selected_fold = cols[0].radio(
    "Select Fold:",
    options=[f'Fold {i}' for i in range(1, 5)],
    horizontal=True,
)
show_training = cols[1].checkbox("Training (Blue)", value=True)
show_validation = cols[2].checkbox("Validation (Orange)", value=True)
compare_with_avg = cols[3].checkbox("Compare with Average (Dotted)", value=True)

# --- 3. Condensed Data Filtering ---
types_to_plot = [
    t for t, checked in [('Training', show_training), ('Validation', show_validation)] if checked
]
selections_to_plot = [selected_fold] + (['Average'] if compare_with_avg else [])

if not types_to_plot or not selections_to_plot:
    st.warning("Please select a line type to display.")
    st.stop()

filtered_data = df_combined[
    df_combined['Fold'].isin(selections_to_plot) & df_combined['Type'].isin(types_to_plot)
]

# --- 4. Plotting Function (Unchanged logic, accepts pre-filtered data) ---
def create_plot(data, task, metric):
    fig = px.line(
        data,
        x='Epoch', y='Loss', color='Type', line_dash='Fold',
        title=f'<b>{task} {metric} Loss</b>',
        color_discrete_map={'Training': '#1f77b4', 'Validation': '#ff7f0e'},
        line_dash_map={'Average': 'dot', **{f'Fold {i}': 'solid' for i in range(1, 5)}}
    )
    fig.update_layout(showlegend=False, margin=dict(l=40, r=20, t=40, b=30))
    return fig

# --- 5. Looped Display Grid ---
st.markdown("---")
tasks = ['Node', 'Edge', 'Graph']
metrics = ['MSE', 'MAE']

for metric in metrics:
    cols = st.columns(len(tasks))
    for i, task in enumerate(tasks):
        plot_data = filtered_data[
            (filtered_data['Task'] == task) & (filtered_data['Metric'] == metric)
        ]
        if not plot_data.empty:
            fig = create_plot(plot_data, task, metric)
            cols[i].plotly_chart(fig, use_container_width=True)