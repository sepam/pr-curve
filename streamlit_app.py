import streamlit as st
import altair as alt
import pandas as pd

from utils import create_noisy_predictions, \
    create_random_predictions, \
    create_perfect_predictions, plot_roc_curve, plot_pr_curve, create_labels


model = st.selectbox('Select a model', ['Noisy', 'Random', 'Perfect'])
if model == 'Noisy':
    noise = st.slider('Noise level', value=0.1, min_value=0.0, max_value=3.0)
bad_rate = st.slider('Bad rate', value=0.5, min_value=0.0, max_value=1.0)
n = st.number_input('Number of samples', value=1000, min_value=100, max_value=10000)
labels = create_labels(n=n, bad_rate=bad_rate)

# Fetch data
if model == 'Perfect':
    predictions = create_perfect_predictions(labels=labels)
elif model == 'Noisy':
    predictions = create_noisy_predictions(labels=labels, noise_level=noise)
else:
    predictions = create_random_predictions(labels=labels)

data = pd.DataFrame({'labels': labels, 'predictions': predictions})
chart = alt.Chart(data).mark_point().encode(
    x='predictions',
    y='labels',
    color='labels',
)

st.altair_chart(chart, use_container_width=True, )


col1, col2 = st.columns(2)
with col1:
    # Plot ROC curve
    auc_fig = plot_roc_curve(labels, predictions)
    st.pyplot(auc_fig)

with col2:
    # Plot PR curve
    pr_fig = plot_pr_curve(labels, predictions)
    st.pyplot(pr_fig)
