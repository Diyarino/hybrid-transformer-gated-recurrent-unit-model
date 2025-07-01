# -*- coding: utf-8 -*-%
"""
Created on %(date)s

@author: Diyar Altinses, M.Sc.

to-do:
    - 
"""

# %% imports

import torch

# %% application

# %%

# thermo = train_set.dataset_clean.flatten(end_dim = -2)[:,7] + 1e-5
# temp = train_set.dataset_clean.flatten(end_dim = -2)[:,4] + 1e-5
# func = numpy.polyfit(temp, numpy.log(thermo), 1)

# thermo = train_set.dataset_clean.mean(dim = 1)[:,7] + 1e-5
# temp = train_set.dataset_clean.mean(dim = 1)[:,4] + 1e-5
# func = numpy.polyfit(temp, numpy.log(thermo), 1)
# coeffs = numpy.polyfit(thermo, temp, 6)
# poly_func = numpy.poly1d(coeffs)


# %% test

if __name__ == '__main__':
	test = torch.rand(1)