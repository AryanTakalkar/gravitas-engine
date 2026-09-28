import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from gravitas.experiment import Experiment
from gravitas.verdict import screen, _membrane_permeability_at_g, STANDARD_G
from gravitas.solver.kinetics import dimensionless_groups
from gravitas.solver.reaction_diffusion import solve_steady_state

st.set_page_config(page_title="GRAVITAS Dashboard", layout="wide")

st.title("GRAVITAS: Gravity Sensitivity Screening Engine")

# Discover examples
examples_dir = Path("examples")
example_files = list(examples_dir.glob("*.yaml"))
example_names = [f.name for f in example_files]

selected_example = st.sidebar.selectbox("Select Experiment Configuration", example_names)
target_g_input = st.sidebar.number_input("Target Gravity (m/s^2)", value=9.80665e-3, format="%.5e")

if selected_example:
    experiment_path = examples_dir / selected_example
    experiment = Experiment.from_yaml(str(experiment_path))
    
    st.sidebar.subheader("Experiment Summary")
    st.sidebar.text(f"Name: {experiment.name}")
    st.sidebar.text(f"Radius: {experiment.aggregate_radius * 1e6:.1f} µm")
    
    # Run screen
    verdict = screen(experiment, target_g=target_g_input)
    
    st.header("Screening Verdict")
    
    if verdict.result == "PASS":
        st.success(f"**PASS**: {verdict.limiting_reason}")
    elif verdict.result == "MARGINAL":
        st.warning(f"**MARGINAL**: {verdict.limiting_reason}")
    else:
        st.error(f"**FAIL**: {verdict.limiting_reason}")
        
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Mechanism Ranking")
        
        # Extract mechanism data
        mechanisms = verdict.mechanism_ranking
        names = [m.name for m in mechanisms]
        changes = [m.relative_change for m in mechanisms]
        crossed = [m.crossed for m in mechanisms]
        
        fig, ax = plt.subplots(figsize=(6, 4))
        y_pos = np.arange(len(names))
        colors = ['red' if c else 'gray' for c in crossed]
        
        ax.barh(y_pos, changes, align='center', color=colors)
        ax.set_yticks(y_pos, labels=names)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Relative Change')
        ax.set_title('Mechanism Sensitivity (Red = Crossed Boundary)')
        
        st.pyplot(fig)
        
    with col2:
        st.subheader(f"Limiting Species Profile: {verdict.limiting_mechanism}")
        
        # Re-solve to get the profile for plotting
        worst_species = next(s for s in experiment.species if s.name == verdict.limiting_mechanism)
        
        alpha, K = dimensionless_groups(worst_species, experiment.aggregate_radius)
        H_1g = experiment.membrane_permeability_1g[worst_species.name] \
            * experiment.aggregate_radius / worst_species.diffusivity
        H_target = _membrane_permeability_at_g(H_1g, target_g_input, STANDARD_G)
        
        r_nodes, C_1g = solve_steady_state(alpha=alpha, K=K, H=H_1g)
        _, C_target = solve_steady_state(alpha=alpha, K=K, H=H_target)
        
        # Convert to dimensional r (microns) and C (mol/m^3)
        r_dim = r_nodes * experiment.aggregate_radius * 1e6
        C_1g_dim = C_1g * worst_species.bulk_concentration
        C_target_dim = C_target * worst_species.bulk_concentration
        
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(r_dim, C_1g_dim, label="1g (Standard)", color="black", linestyle="--")
        ax2.plot(r_dim, C_target_dim, label=f"Target g ({target_g_input:.1e})", color="red")
        
        # Show threshold
        delta = abs(C_1g_dim[0] - C_target_dim[0])
        ax2.axvline(x=0, color='gray', alpha=0.3, linestyle=':')
        
        ax2.set_xlabel('Radius (µm)')
        ax2.set_ylabel(f'{worst_species.name.capitalize()} Concentration (mol/m³)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig2)

    st.subheader("Provenance / Data Sources")
    with st.expander("View Parameter Citations"):
        st.json(verdict.provenance)
