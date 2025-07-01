# -*- coding: utf-8 -*-
'''
Created on Thu Mar  3 12:53:33 2022

@author: Diyar Altinses, M.Sc.

to-do:
    - split to more files
'''

# %% imports

import torch

# %% No change


class NoChange(torch.nn.Module):
    '''
    Do nothing.

    Returns
    -------
    The unmodified sequence.

    '''

    def __init__(self, **args):
        super().__init__()

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        '''
        Do nothing.

        Parameters
        ----------
        inp : torch.Tensor 
            The input to not modify.

        Returns
        -------
        inp : torch.Tensor
            The input.

        '''
        return inp

# %%


class Prints(torch.nn.Module):
    '''
    Do nothing.. just print.

    Returns
    -------
    The unmodified sequence.

    '''

    def __init__(self, **args):
        super().__init__()

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        '''
        Do nothing... just print.

        Parameters
        ----------
        inp : torch.Tensor 
            The input to print.

        Returns
        -------
        inp : torch.Tensor
            The modified input.

        '''
        print(inp.shape)
        return inp


