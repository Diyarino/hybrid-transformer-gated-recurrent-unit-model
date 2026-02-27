# -*- coding: utf-8 -*-
"""
Heuristic control module for heating plant scheduling.

This module provides a rule-based approach to schedule a Wood Heating Plant (HWK),
Block Heating Power Plants (BHKW), and a Gas Turbine (GT) based on heat demand
and electricity pricing.
"""

from typing import Union, List, Any
import numpy as np
import matplotlib.pyplot as plt


def steuere_anlagen(
    waermebedarf_vektor: Union[np.ndarray, List[float]], 
    strompreis_positiv: Union[bool, np.ndarray, List[bool], List[int]], 
    latex_output: bool = False
) -> plt.Figure:
    """
    Simulates the heuristic control of combined heating power plants.

    Parameters
    ----------
    waermebedarf_vektor : Union[np.ndarray, List[float]]
        Time series array of the required heat demand in MW.
    strompreis_positiv : Union[bool, np.ndarray, List[bool], List[int]]
        Indicates if the electricity price is positive. Can be a single boolean 
        applied to all steps, or a time series array matching the demand vector.
    latex_output : bool, optional
        If True, prints the scheduling matrix as a LaTeX table. Defaults to False.

    Returns
    -------
    plt.Figure
        The generated matplotlib figure showing the demand vs. production.
    """
    # --- Plant Parameters ---
    hwk_max_mw: float = 3.0       # Max heat capacity of the wood heating plant (HWK)
    bhkw_heat_mw: float = 1.1     # Heat capacity of a single block heating plant (BHKW)
    gt_heat_mw: float = 9.0       # Heat capacity of the gas turbine (GT)
    min_gt_runtime_h: int = 4     # Minimum required runtime for the GT in hours
    anzahl_bhkw: int = 4          # Total number of available BHKWs

    # --- Initialization ---
    schaltmatrix: List[List[Any]] = []
    hwk_prozent: List[float] = []
    erzeugte_waerme: List[float] = []
    laufzeit_gt: int = 0
    
    # Ensure inputs are numpy arrays for consistent handling
    demand_arr = np.asarray(waermebedarf_vektor)
    
    # Handle single boolean vs. array for the electricity price
    if isinstance(strompreis_positiv, (bool, int, float)):
        price_arr = np.full(len(demand_arr), bool(strompreis_positiv))
    else:
        price_arr = np.asarray(strompreis_positiv, dtype=bool)
        if len(price_arr) != len(demand_arr):
            raise ValueError("Length of strompreis_positiv must match waermebedarf_vektor if passed as an array.")

    # --- Main Control Loop ---
    for waermebedarf, is_price_positive in zip(demand_arr, price_arr):
        
        # 1. HWK Control: Flexible modulation up to 100%
        if waermebedarf <= hwk_max_mw:
            hwk_an = waermebedarf / hwk_max_mw
            verbleibender_bedarf = 0.0
        else:
            hwk_an = 1.0
            verbleibender_bedarf = waermebedarf - hwk_max_mw

        hwk_prozent.append(hwk_an * 100.0)

        # 2. BHKW Control: Discrete step activation
        aktive_bhkw = min(anzahl_bhkw, int(np.ceil(verbleibender_bedarf / bhkw_heat_mw)))
        verbleibender_bedarf -= aktive_bhkw * bhkw_heat_mw

        # 3. GT Control: Check threshold, minimum runtime, and electricity price
        if laufzeit_gt > 0:
            gt_an = 1
            laufzeit_gt -= 1
        elif verbleibender_bedarf > 3.6 and is_price_positive:
            gt_an = 1
            laufzeit_gt = min_gt_runtime_h - 1  # Set remaining minimum runtime
        else:
            gt_an = 0

        # Calculate total generated heat
        gesamtwaerme = (hwk_an * hwk_max_mw) + (aktive_bhkw * bhkw_heat_mw) + (gt_an * gt_heat_mw)
        erzeugte_waerme.append(gesamtwaerme)

        # Append state to the scheduling matrix
        bhkw_states = [1 if i < aktive_bhkw else 0 for i in range(anzahl_bhkw)]
        schaltmatrix.append([f"{hwk_an:.2f}"] + bhkw_states + [gt_an])

    # --- Console / LaTeX Output ---
    if latex_output:
        print("\n\\begin{table}[h]")
        print("\\centering")
        print("\\begin{tabular}{c c c c c c | c}")
        print("HWK (\\%) & BHKW 1 & BHKW 2 & BHKW 3 & BHKW 4 & GT & Gesamtwärme (MW) \\\\")
        print("\\hline")
        for i, row in enumerate(schaltmatrix):
            print(" & ".join(map(str, row)) + f" & {erzeugte_waerme[i]:.2f} \\\\")
        print("\\end{tabular}")
        print("\\caption{Heuristische Kraftwerkssteuerung}")
        print("\\end{table}\n")
    else:
        print(f"\n{'HWK(%)':<8} | {'B1':<2} | {'B2':<2} | {'B3':<2} | {'B4':<2} | {'GT':<2} | {'Gesamtwärme'}")
        print("-" * 48)
        for i, row in enumerate(schaltmatrix):
            print(f"{row[0]:<8} | {row[1]:<2} | {row[2]:<2} | {row[3]:<2} | {row[4]:<2} | {row[5]:<2} | {erzeugte_waerme[i]:.2f} MW")

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(8, 4))
    time_axis = range(len(demand_arr))
    
    ax.plot(
        time_axis, demand_arr, 
        label="Benötigte Wärme (MW)", 
        linestyle='--', marker='o', color='red', markersize=3
    )
    ax.plot(
        time_axis, erzeugte_waerme, 
        label="Erzeugte Wärme (MW)", 
        linestyle='-', marker='s', color='blue', markersize=2
    )
    
    # Safe mean plotting using axhline (Bugfix for the hardcoded 96)
    mean_produced = float(np.mean(erzeugte_waerme))
    mean_demand = float(np.mean(demand_arr))
    
    ax.axhline(mean_produced, linestyle='dashed', color='blue', alpha=0.75, label=f"Ø Erzeugt ({mean_produced:.1f} MW)")
    ax.axhline(mean_demand, linestyle='dashed', color='red', alpha=0.75, label=f"Ø Bedarf ({mean_demand:.1f} MW)")
    
    ax.set_xlabel("Zeit (Stunden)")
    ax.set_ylabel("Wärmeleistung (MW)")
    ax.set_ylim(2, max(max(demand_arr), max(erzeugte_waerme)) * 1.2)  # Dynamic y-limit
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, linestyle=':', alpha=0.7)
    
    fig.tight_layout()
    plt.show()
    
    return fig


if __name__ == "__main__":
    # Example Data Generation
    HOURS = 96
    np.random.seed(42)
    
    # Base load + sine wave + noise
    waermebedarf = 5 + 3 * np.sin(np.linspace(0, 4 * np.pi, HOURS)) + np.random.normal(0, 0.5, HOURS)
    
    # Simulated electricity price vector (Boolean array: True if price is high enough)
    # Using the vector directly instead of ignoring it!
    strompreis_vektor = np.where(waermebedarf > 8, True, False) 
    
    # Execute heuristic control
    steuere_anlagen(waermebedarf, strompreis_vektor, latex_output=False)