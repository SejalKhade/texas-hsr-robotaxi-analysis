"""
Visualization helpers — chart builders for both Streamlit and standalone use.
All functions return Plotly figures.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def mode_shift_bar(results: pd.DataFrame) -> go.Figure:
    """Stacked bar — shift from flights vs cars per route."""
    data = pd.melt(
        results,
        id_vars=["route"],
        value_vars=["shift_from_flights", "shift_from_cars"],
        var_name="mode", value_name="passengers"
    )
    data["mode"] = data["mode"].map({
        "shift_from_flights": "From Flights",
        "shift_from_cars":    "From Cars"
    })
    return px.bar(
        data, x="route", y="passengers", color="mode", barmode="stack",
        labels={"passengers": "Annual Passengers", "route": "Corridor"},
        color_discrete_map={"From Flights": "#636EFA", "From Cars": "#EF553B"},
        title="Mode Shift by Route"
    )


def cost_comparison_bar(results: pd.DataFrame) -> go.Figure:
    """Bar chart comparing door-to-door cost per mode."""
    avg = results.mean(numeric_only=True)
    modes  = ["Car",           "Flight",           "HSR + Robotaxi"]
    costs  = [avg["car_cost_usd"], avg["flight_cost_usd"], avg["hsr_robotaxi_cost_usd"]]
    colors = ["#EF553B",       "#636EFA",          "#00CC96"]

    return px.bar(
        x=modes, y=costs, color=modes, color_discrete_sequence=colors,
        labels={"x": "Mode", "y": "Cost (USD)"},
        title="Door-to-Door Cost Comparison (Average across routes)"
    )


def scenario_co2_bar(all_results: pd.DataFrame) -> go.Figure:
    """Bar chart of total CO2 avoided per scenario."""
    summary = (
        all_results
        .groupby("scenario")["avoided_metric_tons_co2"]
        .sum()
        .reset_index()
    )
    return px.bar(
        summary, x="scenario", y="avoided_metric_tons_co2",
        color="scenario",
        labels={"avoided_metric_tons_co2": "CO₂ Avoided (metric tons/yr)", "scenario": ""},
        color_discrete_map={"Low": "#FFA15A", "Medium": "#636EFA", "High": "#00CC96"},
        title="CO₂ Avoided by Adoption Scenario"
    )


def monte_carlo_histogram(mc: pd.DataFrame) -> go.Figure:
    """Histogram of CO2 avoidance across Monte Carlo simulations with P10/P50/P90."""
    fig = px.histogram(
        mc, x="total_avoided_tons_co2", nbins=60,
        labels={"total_avoided_tons_co2": "CO₂ Avoided (metric tons/yr)"},
        title="Monte Carlo: CO₂ Avoidance Distribution (1,000 simulations)",
        color_discrete_sequence=["#636EFA"],
    )
    for q, label, color in [
        (0.10, "P10", "red"),
        (0.50, "P50", "orange"),
        (0.90, "P90", "green"),
    ]:
        val = mc["total_avoided_tons_co2"].quantile(q)
        fig.add_vline(
            x=val, line_dash="dash", line_color=color,
            annotation_text=f"{label}: {val:,.0f}",
            annotation_position="top right"
        )
    return fig


def riders_by_scenario_line(all_results: pd.DataFrame) -> go.Figure:
    """Line chart of annual riders across scenarios per route."""
    return px.line(
        all_results, x="scenario", y="hsr_annual_riders", color="route",
        markers=True,
        labels={"hsr_annual_riders": "Annual Riders", "scenario": "Scenario"},
        title="Annual HSR Ridership by Scenario and Route"
    )


def grid_demand_bar(results: pd.DataFrame) -> go.Figure:
    """Bar chart of estimated peak grid demand per route."""
    return px.bar(
        results, x="route", y="estimated_peak_mw",
        labels={"estimated_peak_mw": "Peak Demand (MW)", "route": "Corridor"},
        title="Estimated ERCOT Grid Demand Increase per Route",
        color="route"
    )
