#!/usr/bin/env python3
"""
part1.py

UNSW COMP9444 Neural Networks and Deep Learning

ONLY COMPLETE METHODS AND CLASSES MARKED "TODO".

DO NOT MODIFY IMPORTS. DO NOT ADD EXTRA FUNCTIONS.
DO NOT MODIFY EXISTING FUNCTION SIGNATURES.
DO NOT IMPORT ADDITIONAL LIBRARIES.
DOING SO MAY CAUSE YOUR CODE TO FAIL AUTOMATED TESTING.
"""
import torch

# Simple addition operation

def simple_addition(x, y):
    """
    TODO: Implement a simple addition function that accepts two tensors and returns the result.
    """
    return x.add(y)


# Resize tensors
# Use view() to implement the following functions ( flatten() and reshape() are not allowed )

def simple_reshape(x, shape):
    """
    TODO: Implement a function that reshapes the given tensor as the given shape and returns the result.
    """
    # if x.shape.numel() != shape.numel():
    #    return x
    return x.view(shape)
   


def simple_flat(x):
    """
    TODO: Implement a function that flattens the given tensor and returns the result.
    """
    flat = x.view(1, -1)
    return flat.squeeze()


# Transpose and Permutation

def simple_transpose(x):
    """
    TODO: Implement a function that swaps the first dimension and
        the second dimension of the given matrix x and returns the result.
    """
    return x.view(x.shape[1], x.shape[0])

def simple_permute(x, order):
    """
    TODO: Implement a function that permute the dimensions of the given tensor
        x according to the given order and returns the result.
    """
    permuted = []
    for i in range(0, x.ndim):
        permuted.append(x.shape[order[i]])
    return x.view(permuted)
        
# Matrix multiplication (with broadcasting).

def simple_dot_product(x, y):
    """
    TODO: Implement a function that computes the dot product of
        two rank 1 tensors and returns the result.
    """
    # TODO: Check dimensions are equal
    if x.ndim != y.ndim:
        raise ValueError("x and y must have the same dimensions.")
    output = 0
    for i in range(0,x.numel()):
        output += x[i]*y[i]
    # or return x.dot(y)
    return output


def simple_matrix_mul(x, y):
    """
    TODO: Implement a function that performs a matrix multiplication
        of two given rank 2 tensors and returns the result.
    """
    return x.matmul(y)


def broadcastable_matrix_mul(x, y):
    """x
    TODO: Implement a function that computes the matrix product of two tensors and returns the result.
        The function needs to be broadcastable.
    """
    if x.numel() == 1:
        return y * x[0]
    elif y.numel() == 1:
        return x * y[0]
    # Check inner matrix dimensions to be multiplied are valid
    # i.e. Given a matrix m1 dimensions are [A,B] and
    # m2 dimensions are [C, D] then B == C to be valid.
    # In PyTorch the last 2 dimensions are the one to be multiplied.
    elif x.shape[-1] == y.shape[-2]:
        # Check that every element after that is valid (from the tail)
        # i.e. every element must either be equal at the corresponding indice
        # from the tail or equal to 1.
        x_pointer = int(x.ndim - 3)
        y_pointer = int(y.ndim - 3)
        loop_start = 0
        if x_pointer <= y_pointer:
            loop_start = x_pointer
        else:
            loop_start = y_pointer
        for i in range(loop_start, -1, -1):
            if (x.shape[x_pointer] == y.shape[y_pointer] or x.shape[x_pointer] == 1 or y.shape[y_pointer] == 1):
                x_pointer -= 1
                y_pointer -= 1
                continue
            else:
                raise ValueError("Input violates PyTorch broadcasting rules")
        return x.matmul(y)
    else:
        raise ValueError("Input violates PyTorch broadcasting rules.")

# Concatenate and stack.
def simple_concatenate(tensors):
    """
    TODO: Implement a function that concatenates the given sequence of tensors
        in the first dimension and returns the result
    """
    # TODO: Check if the first dimension is 0 or 1
    return torch.cat(tensors, 0)


def simple_stack(tensors, dim):
    """
    TODO: Implement a function that concatenates the given sequence of tensors
        along a new dimension(dim) and returns the result.
    """
    return torch.stack(tensors, dim)


