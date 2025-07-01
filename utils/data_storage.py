# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 12:33:43 2022

@author: Diyar Altinses, M.Sc.

to-do:
    - 
"""

# %% imports

import time
import torch

# %% data storage class


class DataStorage():
    """
    Stores training data while also offering customizable prints of the data.

    Usage
    ----------
    Create an instance.\n
    Call the Store function in every Batch with a given list of the values to store.

    Parameters
    ----------
    names : list of str
        List of str of values to store. Automatically creates and computed a moving average\n
        for the names 'Loss' and 'Acc' if those are given in this list.
    average_window : int, optional
        Window size (in Batches) for the moving average calculation. The default is 100.
    show : int, optional
        Number of Batches between each new print. The default is 25.
    line : int, optional
        Number of Batches to show values in a new line. The default is 500.
    header : int, optional
        Number of Batches to reprint the names for the columns. The default is 5000.
    step : int, optional
        Step size of data storage in Batches. Data gets stored every step Batches.\n
        The default is 1. step = 2 reduces memory consumption by 50\%.
    precision : int, optional
        Number of decimal digits shown.
    auto_show : bool, optional
        Enable/Disable automatic value display. The default is True.

    Returns
    -------
    None.

    """
    def __init__(self, names, average_window=100, show=2, line=50, header=500, step=1, precision=3, name = "", auto_show = True):
        self.Name = name
        self.Names = ["Time"]
        for name in names:
            self.Names.append(name)
        self.AverageWindow = average_window
        self.Show = show
        self.Line = line
        self.Header = header
        self.Step = step
        self.Precision = precision
        self.Batch = 0
        self.autoshow = auto_show

        if "Loss" in self.Names:
            self.Names.append("avg. Loss")
        if "Acc" in self.Names:
            self.Names.append("avg. Acc")

        self.Lens = [len(self.Names[idx])+5 for idx in range(len(self.Names))]
        self.StoredValues = {}

        for name in self.Names:
            self.StoredValues[name] = []

        self.Columns = len(self.Names)
        self.DumpValues = {}


    def Store(self, vals, force = False):
        """
        Stores data in internal StoredValues-dictionary.

        Parameters
        ----------
        vals : list of values
            List of values to be stored in the internal 'StoredValues'-dictionary.\n
            Order has to be the same as given during initialization. Best used with \n
            int, float or torch.tensor.
        force : int
            If given an integer it appends the values with the given batch number.

        Returns
        -------
        None.

        """
        # save time when first storing
        if self.Batch == 0:
            self.DumpValues["TimeStart"] = time.time()
        if self.Batch%self.Step == 0 or force > 0:
            if len(self.StoredValues["Time"]) == 0:
                self.StoredValues["Time"] = [(time.time() - self.DumpValues["TimeStart"])/60]
            else:
                self.StoredValues["Time"].append((time.time() - self.DumpValues["TimeStart"])/60.0)
            for col in range(1,self.Columns):
                name = self.Names[col]
                if name == "avg. Loss":
                    self.StoredValues[name].append(torch.sum(torch.tensor(self.StoredValues["Loss"][-self.AverageWindow:]))/self.AverageWindow)
                elif name == "avg. Acc":
                    self.StoredValues[name].append(torch.sum(torch.tensor(self.StoredValues["Acc"][-self.AverageWindow:]))/self.AverageWindow)
                else:
                    if type(vals[col-1]) == torch.Tensor:
                        self.StoredValues[name].append(vals[col-1].cpu().detach().item())
                    else:
                        self.StoredValues[name].append(vals[col-1])
            
            if self.autoshow:
                if self.Batch == 0:
                    self._GetHead()
                    self._Display()
                    print("")
                else:
                    if self.Batch%self.Show == 0 or force > 0:
                        self._Display()
                    if self.Batch%self.Line == 0:
                        print("")
                    if self.Batch%self.Header == 0:
                        self._GetHead()
        self.Batch+=1

    def _Display(self):
        outstr = "\r"
        args = []
        for col in range(self.Columns):
            val = self.StoredValues[self.Names[col]][-1]
            outstr += "{:s}"

            if type(val) == float:
                val = str(round(val, self.Precision))
            elif type(val) == torch.Tensor:
                val = str(round(val.item(), self.Precision))
            else:
                val = str(val)
            args.append(val+(self.Lens[col]-len(val))*" ")
        print(outstr.format(*args), end="")

    def _GetHead(self):
        print("")
        string = ""
        for col in range(self.Columns):
            name = self.Names[col]
            string += name+(self.Lens[col]-len(name))*" "
        print(string)
