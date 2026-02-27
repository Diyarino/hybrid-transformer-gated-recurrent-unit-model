import torch
from torch.utils.data import Dataset as TorchDataset
from typing import Callable, Tuple, Optional, Any

class Dataset(TorchDataset):
    """
    A custom iterable dataset.

    Parameters
    ----------
    data : torch.Tensor
        Input data tensor.
    label : torch.Tensor
        Output data (target) tensor.
    transforms : Callable, optional
        A function or transform to apply to the data points. Defaults to None.
    """

    def __init__(
        self, 
        data: torch.Tensor, 
        label: torch.Tensor, 
        transforms: Optional[Callable] = None
    ) -> None:
        self.data = data
        self.label = label
        self.transforms = transforms

    def __str__(self) -> str:
        """Return a brief description of the dataset."""
        return "Simple Dataset"

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the data point and its corresponding label at the given index.

        Parameters
        ----------
        idx : int
            Index of the item to retrieve.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the (optionally transformed) data point and its label.
        """
        sample = self.data[idx]
        
        # Apply transformation with a 50% probability
        if self.transforms is not None and torch.rand(1).item() > 0.5:
            sample = self.transforms(sample)
            
        return sample, self.label[idx]


class DataLoader:
    """
    A simple custom data loader for iterating over a dataset in batches.

    Parameters
    ----------
    data : Any
        The dataset to load data from. Must support slicing/multiple indices.
    batch_size : int, optional
        The number of samples per batch. Defaults to 1.
    shuffle : bool, optional
        Whether to shuffle the data at the start of each iteration. Defaults to False.
    """

    def __init__(self, data: Any, batch_size: int = 1, shuffle: bool = False) -> None:
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.len = len(data)

    def __str__(self) -> str:
        """Return a brief description of the dataloader."""
        return "Simple DataLoader"

    def __iter__(self) -> 'DataLoader':
        """
        Initialize the iterator and optionally shuffle the data indices.

        Returns
        -------
        DataLoader
            The iterator object itself.
        """
        if self.shuffle:
            indices = torch.randperm(self.len)
        else:
            indices = torch.arange(self.len)
            
        self.input_tensors, self.targets_tensors = self.data[indices]
        self.i = 0
        
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fetch the next batch of data.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A batch of inputs and their corresponding targets.

        Raises
        ------
        StopIteration
            When all batches have been yielded.
        """
        if self.i >= self.len:
            raise StopIteration
            
        batch_input = self.input_tensors[self.i : self.i + self.batch_size]
        batch_target = self.targets_tensors[self.i : self.i + self.batch_size]
        
        self.i += self.batch_size
        
        return batch_input, batch_target

    def __len__(self) -> int:
        """Return the total number of samples processed by the dataloader."""
        return self.len