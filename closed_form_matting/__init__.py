#!/usr/bin/env python
"""Init script when importing closed-form-matting package"""

from closed_form_matting.closed_form_matting import (
    compute_laplacian,
    closed_form_matting_with_prior,
    closed_form_matting_with_trimap,
    closed_form_matting_with_scribbles,
)
from closed_form_matting.solve_foreground_background import (
    solve_foreground_background
)

__version__ = '1.0.0'
__all__ = [
    'compute_laplacian',
    'closed_form_matting_with_prior',
    'closed_form_matting_with_trimap',
    'closed_form_matting_with_scribbles',
    'solve_foreground_background',
]
