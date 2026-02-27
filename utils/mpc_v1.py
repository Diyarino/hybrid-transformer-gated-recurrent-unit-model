# -*- coding: utf-8 -*-
"""
Model Predictive Control (MPC) module for heat generation scheduling.

This module optimizes the scheduling of a Wood Heating Power Plant (HWK), 
multiple Block Heating Power Plants (BHKW), and a Gas Turbine (GT) based on 
heat demand forecasting and electricity prices.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import List, Dict, Any, Tuple


class ImprovedMPCHeatingController:
    """
    Model Predictive Controller for a combined heating power plant system.
    """

    def __init__(self) -> None:
        # Plant parameters
        self.hwk_max_mw: float = 3.0  # MW Max heat capacity of the wood heating plant (HWK)
        self.bhkw_heat_mw: float = 1.1  # MW Heat capacity of a single block heating plant (BHKW)
        self.gt_heat_mw: float = 9.0  # MW Heat capacity of the gas turbine (GT)
        self.min_gt_runtime_h: int = 4  # Minimum required runtime for the gas turbine in hours
        self.anzahl_bhkw: int = 4  # Total number of available BHKWs
        
        # MPC horizons
        self.prediction_horizon: int = 24  # Hours
        self.control_horizon: int = 8  # Hours
        
        # Cost parameters (Penalty weights for the objective function)
        self.cost_underproduction: float = 10000.0  # Heavy quadratic penalty for unmet demand
        self.cost_overproduction: float = 1.0       # Linear penalty for overproduction
        self.cost_gt_switch: float = 30.0           # Cost penalty for switching the GT
        self.cost_bhkw_switch: float = 5.0          # Cost penalty for switching a BHKW
        self.cost_gt_operation: float = 20.0        # Hourly operation cost of the GT
        self.cost_bhkw_operation: float = 10.0      # Hourly operation cost of a BHKW
        
        # Internal state variables
        self.gt_runtime_remaining: int = 0
        self.current_bhkw_on: int = 0
        self.current_gt_on: int = 0
        self.current_hwk_load: float = 0.0
        self.gt_forced_off: bool = False

    def predict_heat_demand(self, current_time: int, heat_demand_history: List[float]) -> np.ndarray:
        """
        Predicts future heat demand based on historical daily patterns.

        Parameters
        ----------
        current_time : int
            The current hour index in the simulation.
        heat_demand_history : List[float]
            Historical array of past heat demands.

        Returns
        -------
        np.ndarray
            Predicted heat demand for the duration of the prediction horizon.
        """
        history_len = len(heat_demand_history)
        
        if history_len < 24:
            default_val = np.mean(heat_demand_history) if history_len > 0 else 5.0
            return np.full(self.prediction_horizon, default_val)
        
        # Extract daily periodicity
        daily_pattern: List[float] = []
        for h in range(24):
            # Fetch all past data points corresponding to the same hour of the day
            idxs = [i for i in range(history_len - 24, history_len) if i % 24 == h]
            if idxs:
                daily_pattern.append(float(np.mean([heat_demand_history[i] for i in idxs])))
            else:
                daily_pattern.append(5.0)  # Fallback default
        
        # Build prediction array
        prediction: List[float] = []
        for i in range(self.prediction_horizon):
            hour_of_day = (current_time + i) % 24
            prediction.append(daily_pattern[hour_of_day])
            
        return np.array(prediction)

    def calculate_heat_production(self, u_hwk: float, u_bhkw: float, u_gt: float) -> float:
        """
        Calculates the total heat production based on current control signals.

        Parameters
        ----------
        u_hwk : float
            Load ratio of the HWK [0.0 to 1.0].
        u_bhkw : float
            Number of active BHKWs.
        u_gt : float
            Binary state of the GT (1 or 0).

        Returns
        -------
        float
            Total produced heat in MW.
        """
        return (u_hwk * self.hwk_max_mw) + (u_bhkw * self.bhkw_heat_mw) + (u_gt * self.gt_heat_mw)

    def cost_function(
        self, 
        u: np.ndarray, 
        heat_demand_prediction: np.ndarray, 
        current_state: Dict[str, Any]
    ) -> float:
        """
        Objective function for the MPC optimizer.
        
        Evaluates the operational cost and penalties over the prediction horizon.
        Formula utilizes quadratic penalty $J = \sum (\text{demand} - \text{production})^2$ 
        for underproduction to heavily penalize deficit.

        Parameters
        ----------
        u : np.ndarray
            Flattened array of control signals for the entire horizon.
        heat_demand_prediction : np.ndarray
            The predicted heat demand.
        current_state : Dict[str, Any]
            The current state of the generators.

        Returns
        -------
        float
            The total calculated cost (scalar).
        """
        ph = self.prediction_horizon
        u_hwk = u[0 : ph]
        u_bhkw = u[ph : 2 * ph]
        u_gt = u[2 * ph : 3 * ph]
        
        total_cost: float = 0.0
        prev_gt: int = current_state['gt_on']
        prev_bhkw: int = current_state['bhkw_on']
        
        for i in range(ph):
            # Heat balance calculation
            heat_prod = self.calculate_heat_production(u_hwk[i], u_bhkw[i], u_gt[i])
            heat_diff = heat_demand_prediction[i] - heat_prod
            
            # Asymmetric penalty: quadratic for deficit, linear for surplus
            if heat_diff > 0:
                total_cost += self.cost_underproduction * (heat_diff ** 2)
            else:
                total_cost += self.cost_overproduction * abs(heat_diff)
            
            # Generator operational costs
            total_cost += u_gt[i] * self.cost_gt_operation
            total_cost += u_bhkw[i] * self.cost_bhkw_operation
            
            # Switching penalties
            if i > 0:
                if u_gt[i] != u_gt[i - 1]:
                    total_cost += self.cost_gt_switch
                if u_bhkw[i] != u_bhkw[i - 1]:
                    total_cost += self.cost_bhkw_switch
            else:
                if u_gt[i] != prev_gt:
                    total_cost += self.cost_gt_switch
                if u_bhkw[i] != prev_bhkw:
                    total_cost += self.cost_bhkw_switch
            
            # Strict enforcement of minimum gas turbine runtime
            if u_gt[i] == 1 and (i == 0 or u_gt[i - 1] == 0):
                remaining_horizon = min(ph - i, self.min_gt_runtime_h)
                for j in range(i + 1, i + remaining_horizon):
                    if j < ph and u_gt[j] == 0:
                        total_cost += 1e6  # Massive penalty for violating constraints
        
        return total_cost

    def optimize_control(
        self, 
        heat_demand_prediction: np.ndarray, 
        current_state: Dict[str, Any]
    ) -> np.ndarray:
        """
        Runs the numerical optimization to find the best control signals.

        Parameters
        ----------
        heat_demand_prediction : np.ndarray
            Array of predicted heat demands.
        current_state : Dict[str, Any]
            Current operational state.

        Returns
        -------
        np.ndarray
            Optimized control signals (rounded for discrete variables).
        """
        ph = self.prediction_horizon
        u0 = np.zeros(3 * ph)
        
        # Initial guess based on current state
        u0[0 : ph] = current_state['hwk_load'] / self.hwk_max_mw
        u0[ph : 2 * ph] = current_state['bhkw_on']
        u0[2 * ph : 3 * ph] = current_state['gt_on']
        
        # Define boundary constraints
        bounds: List[Tuple[float, float]] = []
        for i in range(3 * ph):
            if i < ph:
                bounds.append((0.0, 1.0))  # HWK: Continuous 0% to 100%
            elif i < 2 * ph:
                # CRITICAL FIX: BHKW must be able to scale up to the max number of available units
                bounds.append((0.0, float(self.anzahl_bhkw))) 
            else:
                bounds.append((0.0, 1.0))  # GT: Binary (Continuous for Powell, rounded later)
        
        # Execute Scipy Powell optimization (Pseudo-Mixed-Integer)
        res = minimize(
            self.cost_function, 
            u0, 
            args=(heat_demand_prediction, current_state),
            bounds=bounds, 
            method='Powell', 
            options={'maxiter': 1000, 'disp': False} # Turned off verbose scipy output for clean console
        )
        
        optimal_u = res.x
        
        # Enforce discrete (integer) variables for BHKW and GT
        optimal_u[ph : 2 * ph] = np.round(optimal_u[ph : 2 * ph])
        optimal_u[2 * ph : 3 * ph] = np.round(optimal_u[2 * ph : 3 * ph])
        
        return optimal_u

    def control_step(
        self, 
        current_heat_demand: float, 
        electricity_price: float, 
        history: List[float], 
        current_time: int
    ) -> Dict[str, float]:
        """
        Executes a single MPC iteration and returns the immediate optimal control signal.

        Parameters
        ----------
        current_heat_demand : float
            The heat demand at the current time step.
        electricity_price : float
            Current electricity price.
        history : List[float]
            History of previous heat demands.
        current_time : int
            Current simulation index.

        Returns
        -------
        Dict[str, float]
            Control variables for the immediate next step.
        """
        heat_demand_pred = self.predict_heat_demand(current_time, history)
        
        current_state = {
            'gt_on': self.current_gt_on,
            'bhkw_on': self.current_bhkw_on,
            'hwk_load': self.current_hwk_load,
            'gt_runtime': max(0, self.gt_runtime_remaining)
        }
        
        optimal_u = self.optimize_control(heat_demand_pred, current_state)
        
        # Extract the very first step of the optimized horizon (MPC core principle)
        ph = self.prediction_horizon
        u_hwk = optimal_u[0]
        u_bhkw = int(round(optimal_u[ph]))
        u_gt = int(round(optimal_u[2 * ph]))
        
        # State tracking: GT minimum runtime constraint update
        if u_gt == 1 and self.current_gt_on == 0:
            self.gt_runtime_remaining = self.min_gt_runtime_h
        elif u_gt == 1:
            self.gt_runtime_remaining = max(0, self.gt_runtime_remaining - 1)
            
        # Hard override based on electricity prices
        if electricity_price <= 0 and u_gt == 1:
            u_gt = 0
            self.gt_runtime_remaining = 0
            
        # Finalize state variables
        self.current_gt_on = u_gt
        self.current_bhkw_on = min(u_bhkw, self.anzahl_bhkw)
        self.current_hwk_load = u_hwk * self.hwk_max_mw
        
        return {
            'hwk_load': self.current_hwk_load,
            'bhkw_on': float(self.current_bhkw_on),
            'gt_on': float(self.current_gt_on),
            'total_heat': self.calculate_heat_production(u_hwk, self.current_bhkw_on, u_gt)
        }


def improved_mpc_steuere_anlagen(
    waermebedarf_vektor: np.ndarray, 
    strompreis_vektor: np.ndarray, 
    latex_output: bool = False
) -> plt.Figure:
    """
    Main simulation loop evaluating the MPC over a time series.

    Parameters
    ----------
    waermebedarf_vektor : np.ndarray
        Array representing the heat demand over time.
    strompreis_vektor : np.ndarray
        Array representing electricity prices over time.
    latex_output : bool, optional
        If True, prints a formatted LaTeX table. Defaults to False.

    Returns
    -------
    plt.Figure
        The generated matplotlib figure showing the results.
    """
    controller = ImprovedMPCHeatingController()
    
    schaltmatrix: List[List[Any]] = []
    erzeugte_waerme: List[float] = []
    history: List[float] = []
    
    print("Starting MPC Simulation... This may take a moment.")
    
    for i in range(len(waermebedarf_vektor)):
        current_demand = waermebedarf_vektor[i]
        current_price = strompreis_vektor[i]
        history.append(current_demand)
        
        control = controller.control_step(current_demand, current_price, history, i)
        
        erzeugte_waerme.append(control['total_heat'])
        
        # Generate binary states for the 4 BHKWs based on the continuous integer
        bhkw_count = int(control['bhkw_on'])
        bhkw_states = [1 if j < bhkw_count else 0 for j in range(controller.anzahl_bhkw)]
        
        hwk_ratio = control['hwk_load'] / controller.hwk_max_mw
        row = [f"{hwk_ratio * 100:.1f}"] + bhkw_states + [int(control['gt_on'])]
        schaltmatrix.append(row)
        
    # Standardized output formatting
    if latex_output:
        print("\n\\begin{table}[h]")
        print("\\centering")
        print("\\begin{tabular}{c c c c c c | c}")
        print("HWK (\\%) & BHKW 1 & BHKW 2 & BHKW 3 & BHKW 4 & GT & Gesamtwärme (MW) \\\\")
        print("\\hline")
        for i, row in enumerate(schaltmatrix):
            print(" & ".join(map(str, row)) + f" & {erzeugte_waerme[i]:.2f} \\\\")
        print("\\end{tabular}")
        print("\\caption{Ergebnisse der MPC Kraftwerkssteuerung}")
        print("\\end{table}\n")
    else:
        print(f"\n{'HWK(%)':<8} | {'B1':<2} | {'B2':<2} | {'B3':<2} | {'B4':<2} | {'GT':<2} | {'Gesamtwärme'}")
        print("-" * 48)
        for i, row in enumerate(schaltmatrix):
            print(f"{row[0]:<8} | {row[1]:<2} | {row[2]:<2} | {row[3]:<2} | {row[4]:<2} | {row[5]:<2} | {erzeugte_waerme[i]:.2f} MW")

    # Plotting
    fig = plt.figure(figsize=(12, 6))
    
    # Calculate capacities per generator type
    hwk_leistung = [float(row[0]) / 100.0 * controller.hwk_max_mw for row in schaltmatrix]
    bhkw_leistung = [sum(int(x) for x in row[1:5]) * controller.bhkw_heat_mw for row in schaltmatrix]
    gt_leistung = [int(row[5]) * controller.gt_heat_mw for row in schaltmatrix]
    
    time_axis = range(len(waermebedarf_vektor))
    
    plt.stackplot(
        time_axis, hwk_leistung, bhkw_leistung, gt_leistung,
        labels=['HWK (Holz)', 'BHKW (Blockheizkraftwerk)', 'GT (Gasturbine)'], 
        colors=['#2ca02c', '#ff7f0e', '#9467bd'], 
        alpha=0.7
    )
    
    plt.plot(
        time_axis, waermebedarf_vektor, 
        label="Benötigte Wärme (MW)", 
        linestyle='--', linewidth=2.5, color='red'
    )
    
    plt.title("MPC Kraftwerkssteuerung - Wärmeproduktion vs. Bedarf", fontsize=14)
    plt.xlabel("Zeit (Stunden)", fontsize=12)
    plt.ylabel("Wärmeleistung (MW)", fontsize=12)
    plt.ylim(0, max(waermebedarf_vektor) * 1.3)
    plt.legend(loc='upper left', framealpha=0.9)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    return fig


if __name__ == "__main__":
    # Test parameters
    HOURS = 48  # Reduced for quicker demonstration (was 192)
    t = np.linspace(0, 4 * np.pi, HOURS)
    
    # Generate semi-realistic heat demand profile
    waermebedarf = 8 + 6 * np.sin(t) + 2 * np.sin(2 * t + 1) + np.random.normal(0, 0.5, HOURS)
    waermebedarf = np.maximum(3.0, waermebedarf)
    
    # Generate electricity prices
    strompreis = np.array([1.0 if 6 <= i % 24 <= 22 else 0.0 for i in range(HOURS)])
    
    # Run simulation
    improved_mpc_steuere_anlagen(waermebedarf, strompreis, latex_output=False)