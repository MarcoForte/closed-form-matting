"""Computes foreground and background images given source image and transparency map.

This module implements foreground and background reconstruction method described in Section 7 of:
    Levin, Anat, Dani Lischinski, and Yair Weiss. "A closed-form solution to natural image
    matting." IEEE Transactions on Pattern Analysis and Machine Intelligence 30.2 (2008): 228-242.

Please note, that the cost-function optimized by this code doesn't perfectly match Eq. 19 of the
paper, since our implementation mimics `solveFB.m` Matlab function provided by the authors of the
original paper (this implementation is
availale at http://people.csail.mit.edu/alevin/matting.tar.gz).

The code can be used in two ways:
    1. By importing solve_foregound_background in your code:
        ```
            from solve_foregound_background import solve_foregound_background
            ...
            foreground, background = solve_foregound_background(image, alpha)
        ```
    2. From command line:
        ```
            ./solve_foregound_background.py image.png alpha.png foreground.png background.png
        ```

Authors: Mikhail Erofeev, Yury Gitman.
"""

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

CONST_ALPHA_MARGIN = 0.02


def __spdiagonal(diag):
    """Produces sparse matrix with given vector on its main diagonal."""
    return scipy.sparse.spdiags(diag, (0,), len(diag), len(diag))


def get_grad_operator(mask):
    """Returns sparse matrix computing horizontal, vertical, and two diagonal gradients."""
    horizontal_left = np.ravel_multi_index(np.nonzero(mask[:, :-1] | mask[:, 1:]), mask.shape)
    horizontal_right = horizontal_left + 1

    vertical_top = np.ravel_multi_index(np.nonzero(mask[:-1, :] | mask[1:, :]), mask.shape)
    vertical_bottom = vertical_top + mask.shape[1]

    diag_main_1 = np.ravel_multi_index(np.nonzero(mask[:-1, :-1] | mask[1:, 1:]), mask.shape)
    diag_main_2 = diag_main_1 + mask.shape[1] + 1

    diag_sub_1 = np.ravel_multi_index(np.nonzero(mask[:-1, 1:] | mask[1:, :-1]), mask.shape) + 1
    diag_sub_2 = diag_sub_1 + mask.shape[1] - 1

    indices = np.stack((
        np.concatenate((horizontal_left, vertical_top, diag_main_1, diag_sub_1)),
        np.concatenate((horizontal_right, vertical_bottom, diag_main_2, diag_sub_2))
    ), axis=-1)
    return scipy.sparse.coo_matrix(
        (np.tile([-1, 1], len(indices)), (np.arange(indices.size) // 2, indices.flatten())),
        shape=(len(indices), mask.size))


def get_const_conditions(image, alpha):
    """Returns sparse diagonal matrix and vector encoding color prior conditions."""
    falpha = alpha.flatten()
    weights = (
        (falpha < CONST_ALPHA_MARGIN) * 100.0 +
        0.03 * (1.0 - falpha) * (falpha < 0.3) +
        0.01 * (falpha > 1.0 - CONST_ALPHA_MARGIN)
    )
    conditions = __spdiagonal(weights)

    mask = falpha < 1.0 - CONST_ALPHA_MARGIN
    right_hand = (weights * mask)[:, np.newaxis] * image.reshape((alpha.size, -1))
    return conditions, right_hand


def solve_foreground_background(image, alpha):
    """Compute foreground and background image given source image and transparency map."""

    consts = (alpha < CONST_ALPHA_MARGIN) | (alpha > 1.0 - CONST_ALPHA_MARGIN)
    grad = get_grad_operator(~consts)
    grad_weights = np.power(np.abs(grad * alpha.flatten()), 0.5)

    grad_only_positive = grad.maximum(0)
    grad_weights_f = grad_weights + 0.003 * grad_only_positive * (1.0 - alpha.flatten())
    grad_weights_b = grad_weights + 0.003 * grad_only_positive * alpha.flatten()

    grad_pad = scipy.sparse.coo_matrix(grad.shape)

    smoothness_conditions = scipy.sparse.vstack((
        scipy.sparse.hstack((__spdiagonal(grad_weights_f) * grad, grad_pad)),
        scipy.sparse.hstack((grad_pad, __spdiagonal(grad_weights_b) * grad))
    ))

    composite_conditions = scipy.sparse.hstack((
        __spdiagonal(alpha.flatten()),
        __spdiagonal(1.0 - alpha.flatten())
    ))

    const_conditions_f, b_const_f = get_const_conditions(image, 1.0 - alpha)
    const_conditions_b, b_const_b = get_const_conditions(image, alpha)

    non_zero_conditions = scipy.sparse.vstack((
        composite_conditions,
        scipy.sparse.hstack((
            const_conditions_f,
            scipy.sparse.coo_matrix(const_conditions_f.shape)
        )),
        scipy.sparse.hstack((
            scipy.sparse.coo_matrix(const_conditions_b.shape),
            const_conditions_b
        ))
    ))

    b_composite = image.reshape(alpha.size, -1)

    right_hand = non_zero_conditions.transpose() * np.concatenate((b_composite,
                                                                   b_const_f,
                                                                   b_const_b))

    conditons = scipy.sparse.vstack((
        non_zero_conditions,
        smoothness_conditions
    ))
    left_hand = conditons.transpose() * conditons

    solution = scipy.sparse.linalg.spsolve(left_hand, right_hand).reshape(2, *image.shape)
    foreground = solution[0, :, :, :].reshape(*image.shape)
    background = solution[1, :, :, :].reshape(*image.shape)
    return foreground, background


def main():
    """Parse command line arguments and apply solve_foregound_background."""

    import argparse
    import cv2
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument('image', type=str)
    arg_parser.add_argument('alpha', type=str)
    arg_parser.add_argument('foreground', type=str)
    arg_parser.add_argument('background', type=str, default=None, nargs='?')
    args = arg_parser.parse_args()

    image = cv2.imread(args.image) / 255.0
    alpha = cv2.imread(args.alpha, 0) / 255.0
    foreground, background = solve_foreground_background(image, alpha)
    cv2.imwrite(args.foreground, foreground * 255.0)
    if args.background:
        cv2.imwrite(args.background, background * 255.0)


if __name__ == "__main__":
    main()
