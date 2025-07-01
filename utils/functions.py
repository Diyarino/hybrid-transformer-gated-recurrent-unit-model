# -*- coding: utf-8 -*-%
"""
Created on %(date)s

@author: Diyar Altinses, M.Sc.

to-do:
    - 
"""

# %% imports

import numpy

# %% application

def piecewise_linear(x, breakpoint = 17.32456140350877):
    """
    Defines a piecewise linear function with two linear segments,
    handling NumPy arrays for x.

    Args:
        x: A NumPy array or a single value representing the x-value(s).
        breakpoint: The x-value where the two linear functions meet.

    Returns:
        A NumPy array or a single value representing the y-value(s)
        of the piecewise linear function at x.
    """

    if isinstance(x, numpy.ndarray):  # Check if x is a NumPy array
        y = numpy.where(x <= breakpoint, -0.52 * x + 11.4, -0.064 * x + 3.5)
        return y
    
    
def reverse_piecewise_linear(y, breakpoint=17.32456140350877):
    """
    Calculates the inverse of the piecewise linear function.

    Args:
        y: A NumPy array or a single value representing the y-value(s).
        breakpoint: The x-value where the two linear functions meet.

    Returns:
        A NumPy array or a single value representing the x-value(s)
        corresponding to the given y-value(s).  Returns None if y is outside the range.
    """
    x = numpy.zeros_like(y, dtype=float)  # Initialize x with correct dtype
    for i, y_val in enumerate(y):
        if y_val > -0.52 * breakpoint + 11.4: #y is in the second segment
            x[i] = (y_val - 11.4) / -0.52
        else: #y is in the first segment
            x[i] = (y_val - 3.5) / -0.064
    return x
