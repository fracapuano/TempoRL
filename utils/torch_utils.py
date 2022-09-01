import torch
from typing import Iterable

def iterable_to_cuda(input:Iterable[torch.tensor]) -> Iterable[torch.tensor]: 
    """This function returns an iterable containing all the tensors in the input Iterable in which the various tensors
    are sent to CUDA (if applicable). 

    Args:
        input (Iterable[torch.tensor]): Iterable of tensors to be sent to CUDA.

    Returns:
        Iterable[torch.tensor]: Iterable of tensors sent to CUDA. 
    """
    return [tensor.to("cuda") if torch.cuda.is_available() else tensor for tensor in input]