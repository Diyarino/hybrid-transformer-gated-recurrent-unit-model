# -*- coding: utf-8 -*-
"""
Optimized Model Predictive Control (MPC) module for heat generation scheduling.

This module features enhanced demand forecasting and strict operational constraints
for a simulated heating power plant system.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import List, Dict, Any, Tuple


class OptimizedMPCHeatingController:
    """
    Advanced Model Predictive Controller for a combined heating power plant system.
    """

    def __init__(self) -> None:
        # Plant parameters
        self.hwk_max_mw: float = 3.0       # Max heat capacity of the wood heating plant (HWK)
        self.bhkw_heat_mw: float = 1.1     # Heat capacity of a single block heating plant (BHKW)
        self.gt_heat_mw: float = 9.0       # Heat capacity of the gas turbine (GT)
        self.min_gt_runtime_h: int = 16    # Minimum required runtime for the GT in hours
        self.anzahl_bhkw: int = 4          # Total number of available BHKWs
        
        # MPC parameters
        self.prediction_horizon: int = 24  # Hours
        self.control_horizon: int = 8      # Hours
        
        # Cost parameters (Penalty weights)
        self.cost_underproduction: float = 10000.0  # Heavy penalty for unmet demand
        self.cost_overproduction: float = 1000.0    # Lower penalty for overproduction
        self.cost_gt_switch: float = 100.0          # High cost for switching GT
        self.cost_bhkw_switch: float = 5.0          # Cost for switching BHKW
        self.cost_gt_operation: float = 50.0        # High operational cost for GT
        self.cost_bhkw_operation: float = 10.0      # Operational cost for BHKW
        
        # Internal state variables
        self.gt_runtime_remaining: int = 0
        self.current_bhkw_on: int = 0
        self.current_gt_on: int = 0
        self.current_hwk_load: float = 0.0
        self.gt_forced_off: bool = False

    def predict_heat_demand(self, current_time: int, heat_demand_history: List[float]) -> np.ndarray:
        """
        Predicts future heat demand based on daily patterns and short-term fluctuations.

        Parameters
        ----------
        current_time : int
            The current hour index.
        heat_demand_history : List[float]
            Historical array of past heat demands.

        Returns
        -------
        np.ndarray
            Predicted heat demand for the prediction horizon.
        """
        history_len = len(heat_demand_history)
        
        if history_len < 24:
            default_val = np.mean(heat_demand_history) if history_len > 0 else 5.0
            return np.full(self.prediction_horizon, default_val)
        
        # Calculate daily periodicity
        daily_pattern: List[float] = []
        for h in range(24):
            idxs = [i for i in range(history_len - 24, history_len) if i % 24 == h]
            daily_pattern.append(float(np.mean([heat_demand_history[i] for i in idxs])) if idxs else 5.0)
        
        # Calculate short-term fluctuation factor (last 6 hours vs their daily average)
        short_term = heat_demand_history[-6:] if history_len >= 6 else [5.0] * 6
        mean_daily = np.mean(daily_pattern)
        
        # Safe division to prevent ZeroDivisionError
        short_term_factor = (np.mean(short_term) / mean_daily) if mean_daily > 0 else 1.0
        
        # Build prediction array
        prediction: List[float] = []
        for i in range(self.prediction_horizon):
            hour_of_day = (current_time + i) % 24
            prediction.append(daily_pattern[hour_of_day] * short_term_factor)
            
        return np.array(prediction)

    def calculate_heat_production(self, u_hwk: float, u_bhkw: float, u_gt: float) -> float:
        """Calculates total heat production from control signals."""
        return (u_hwk * self.hwk_max_mw) + (u_bhkw * self.bhkw_heat_mw) + (u_gt * self.gt_heat_mw)

    def cost_function(
        self, 
        u: np.ndarray, 
        heat_demand_prediction: np.ndarray, 
        current_state: Dict[str, Any]
    ) -> float:
        """
        Objective function evaluating cost and constraints over the horizon.
        """
        ph = self.prediction_horizon
        u_hwk = u[0 : ph]
        u_bhkw = u[ph : 2 * ph]
        u_gt = u[2 * ph : 3 * ph]
        
        total_cost: float = 0.0
        prev_gt: float = current_state['gt_on']
        prev_bhkw: float = current_state['bhkw_on']
        
        # 1. Evaluate Gas Turbine minimum runtime constraints
        gt_start_times: List[int] = []
        for i in range(ph):
            # Check if GT was turned on in this step
            if u_gt[i] == 1 and (i == 0 or u_gt[i - 1] == 0):
                gt_start_times.append(i)
                
        for start in gt_start_times:
            min_end = min(start + self.min_gt_runtime_h, ph)
            for j in range(start, min_end):
                if u_gt[j] == 0:
                    total_cost += 1e6  # Massive penalty for violating min runtime
        
        # 2. Evaluate operational and switch costs
        for i in range(ph):
            heat_prod = self.calculate_heat_production(u_hwk[i], u_bhkw[i], u_gt[i])
            heat_diff = heat_demand_prediction[i] - heat_prod
            
            # Penalize mismatch (quadratic for deficit, linear for surplus)
            if heat_diff > 0:
                total_cost += self.cost_underproduction * (heat_diff ** 2)
            else:
                total_cost += self.cost_overproduction * abs(heat_diff)
            
            # Operational costs
            total_cost += u_gt[i] * self.cost_gt_operation
            total_cost += u_bhkw[i] * self.cost_bhkw_operation
            
            # Switch costs (using absolute differences)
            if i > 0:
                total_cost += self.cost_gt_switch * abs(u_gt[i] - u_gt[i - 1])
                total_cost += self.cost_bhkw_switch * abs(u_bhkw[i] - u_bhkw[i - 1])
            else:
                total_cost += self.cost_gt_switch * abs(u_gt[i] - prev_gt)
                total_cost += self.cost_bhkw_switch * abs(u_bhkw[i] - prev_bhkw)
                
        return total_cost

    def optimize_control(
        self, 
        heat_demand_prediction: np.ndarray, 
        current_state: Dict[str, Any]
    ) -> np.ndarray:
        """Runs the Scipy optimizer to find the best control signals."""
        ph = self.prediction_horizon
        u0 = np.zeros(3 * ph)
        
        u0[0 : ph] = current_state['hwk_load'] / self.hwk_max_mw
        u0[ph : 2 * ph] = current_state['bhkw_on']
        u0[2 * ph : 3 * ph] = current_state['gt_on']
        
        bounds: List[Tuple[float, float]] = []
        for i in range(3 * ph):
            if i < ph:
                bounds.append((0.0, 1.0))  # HWK: 0% to 100%
            elif i < 2 * ph:
                bounds.append((0.0, float(self.anzahl_bhkw)))  # BHKW: 0 to Max Units
            else:
                bounds.append((0.0, 1.0))  # GT: Binary (0 or 1)
        
        res = minimize(
            self.cost_function, 
            u0, 
            args=(heat_demand_prediction, current_state),
            bounds=bounds, 
            method='Powell', 
            options={'maxiter': 2000, 'disp': False}
        )
        
        optimal_u = res.x
        
        # Round discrete variables
        optimal_u[ph : 2 * ph] = np.round(optimal_u[ph : 2 * ph])
        optimal_u[2 * ph : 3 * ph] = np.round(optimal_u[2 * ph : 3 * ph])
        
        return optimal_u

    def control_step(
        self, 
        current_heat_demand: float, 
        electricity_price: float, 
        history: List[float], 
        current_time: int
    ) -> Dict[str, Any]:
        """Executes a single MPC iteration."""
        heat_demand_pred = self.predict_heat_demand(current_time, history)
        
        current_state = {
            'gt_on': self.current_gt_on,
            'bhkw_on': self.current_bhkw_on,
            'hwk_load': self.current_hwk_load,
            'gt_runtime': max(0, self.gt_runtime_remaining)
        }
        
        optimal_u = self.optimize_control(heat_demand_pred, current_state)
        
        ph = self.prediction_horizon
        u_hwk = optimal_u[0]
        u_bhkw = int(np.round(optimal_u[ph]))
        u_gt = int(np.round(optimal_u[2 * ph]))
        
        # Update minimum runtime tracker
        if u_gt == 1 and self.current_gt_on == 0:
            self.gt_runtime_remaining = self.min_gt_runtime_h
        elif u_gt == 1:
            self.gt_runtime_remaining = max(0, self.gt_runtime_remaining - 1)
            
        # Hard override: Turn off GT if electricity price is not favorable 
        # Note: This override breaks the min_runtime constraint if triggered!
        if electricity_price <= 0 and u_gt == 1:
            u_gt = 0
            self.gt_runtime_remaining = 0
            
        self.current_gt_on = u_gt
        self.current_bhkw_on = min(u_bhkw, self.anzahl_bhkw)
        self.current_hwk_load = u_hwk * self.hwk_max_mw
        
        return {
            'hwk_load': self.current_hwk_load,
            'bhkw_on': self.current_bhkw_on,
            'gt_on': u_gt,
            'total_heat': self.calculate_heat_production(u_hwk, self.current_bhkw_on, u_gt)
        }


def optimized_mpc_steuere_anlagen(
    waermebedarf_vektor: np.ndarray, 
    strompreis_vektor: np.ndarray, 
    latex_output: bool = False
) -> Tuple[List[List[Any]], List[float], OptimizedMPCHeatingController]:
    """Runs the MPC simulation over the provided time vectors."""
    controller = OptimizedMPCHeatingController()
    
    schaltmatrix: List[List[Any]] = []
    erzeugte_waerme: List[float] = []
    history: List[float] = []
    
    print("Running optimized MPC simulation...")
    
    for i in range(len(waermebedarf_vektor)):
        current_demand = waermebedarf_vektor[i]
        current_price = strompreis_vektor[i]
        history.append(current_demand)
        
        control = controller.control_step(current_demand, current_price, history, i)
        
        erzeugte_waerme.append(control['total_heat'])
        
        # Generate boolean array for active BHKWs
        bhkw_count = int(control['bhkw_on'])
        bhkw_states = [1 if j < bhkw_count else 0 for j in range(controller.anzahl_bhkw)]
        
        hwk_ratio = control['hwk_load'] / controller.hwk_max_mw
        schaltmatrix.append([f"{hwk_ratio:.2f}"] + bhkw_states + [control['gt_on']])
        
    return schaltmatrix, erzeugte_waerme, controller


if __name__ == "__main__":
    # Realistic test parameters
    HOURS = 96
    t = np.linspace(0, 4 * np.pi, HOURS)
    
    waermebedarf = 8 + 6 * np.sin(t) + 2 * np.sin(2 * t + 1) + np.random.normal(0, 0.8, HOURS)
    waermebedarf_vektor = np.maximum(3.0, waermebedarf)  # Minimum demand of 3 MW
    
    # Electricity price: 1 during day, 0 at night, random spikes to 2
    strompreis = np.array([1 if 6 <= i % 24 <= 22 else 0 for i in range(HOURS)])
    strompreis = np.where(np.random.random(HOURS) > 0.9, 2, strompreis)

    # Run simulation
    schaltmatrix, erzeugte_waerme, controller = optimized_mpc_steuere_anlagen(
        waermebedarf_vektor, strompreis
    )
    
    # --- Plotting ---
    fig = plt.figure(figsize=(10, 5))
    time_axis = range(len(waermebedarf_vektor))
    
    # Extract capacities
    hwk_leistung = [float(row[0]) * controller.hwk_max_mw for row in schaltmatrix]
    bhkw_leistung = [sum(int(x) for x in row[1:5]) * controller.bhkw_heat_mw for row in schaltmatrix]
    gt_leistung = [int(row[-1]) * controller.gt_heat_mw for row in schaltmatrix]
    
    # Calculate means for legends
    mean_demand = float(np.mean(waermebedarf_vektor))
    mean_production = float(np.mean(erzeugte_waerme))
    
    plt.stackplot(
        time_axis, hwk_leistung, bhkw_leistung, gt_leistung,
        labels=[
            f'HWK ({np.mean(hwk_leistung):.1f} MW avg)',
            f'BHKW ({np.mean(bhkw_leistung):.1f} MW avg)',
            f'GT ({np.mean(gt_leistung):.1f} MW avg)'
        ],
        colors=['#2ca02c', '#ff7f0e', '#9467bd'], 
        alpha=0.7
    )
    
    plt.plot(
        time_axis, waermebedarf_vektor, 
        label=f'Benötigte Wärme ({mean_demand:.1f} MW avg)', 
        linestyle='--', color='red', linewidth=2
    )
    plt.plot(
        time_axis, erzeugte_waerme, 
        label=f'Erzeugte Wärme ({mean_production:.1f} MW avg)',
        linestyle='-', color='blue', linewidth=2
    )
    
    # Add horizontal lines for means
    plt.axhline(mean_demand, color='red', linestyle=':', alpha=0.6)
    plt.axhline(mean_production, color='blue', linestyle=':', alpha=0.6)
    
    plt.ylim(0, max(waermebedarf_vektor) * 1.4)  # Dynamic y-limit instead of hardcoded 30
    plt.xlabel("Time (Hours)")
    plt.ylabel("Thermal output (MW)")
    plt.title("Optimized MPC Heat Generation Schedule")
    plt.legend(loc='upper left', framealpha=0.9)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()