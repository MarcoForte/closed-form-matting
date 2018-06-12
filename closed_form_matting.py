"""Implementation of Closed-Form Matting.

This module implements natural image matting method described in:
    Levin, Anat, Dani Lischinski, and Yair Weiss. "A closed-form solution to natural image matting."
    IEEE Transactions on Pattern Analysis and Machine Intelligence 30.2 (2008): 228-242.

The code can be used in two ways:
    1. By importing solve_foregound_background in your code:
        ```
            import closed_form_matting
            ...
            # For scribles input
            alpha = closed_form_matting.closed_form_matting_with_scribbles(image, scribbles)

            # For trimap input
            alpha = closed_form_matting.closed_form_matting_with_trimap(image, trimap)

            # For prior with confidence
            alpha = closed_form_matting.closed_form_matting_with_prior(
                image, prior, prior_confidence, optional_const_mask)

            # To get Matting Laplacian for image
            laplacian = compute_laplacian(image, optional_const_mask)
        ```
    2. From command line:
        ```
            # Scribbles input
            ./closed_form_matting.py input_image.png -s scribbles_image.png  -o output_alpha.png

            # Trimap input
            ./closed_form_matting.py input_image.png -t scribbles_image.png  -o output_alpha.png

            # Add flag --solve-fg to compute foreground color and output RGBA image instead
            # of alpha.
        ```
"""

from __future__ import division

import logging

import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided
import scipy.sparse
import scipy.sparse.linalg


def _rolling_block(A, block=(3, 3)):
    """Applies sliding window to given matrix."""
    shape = (A.shape[0] - block[0] + 1, A.shape[1] - block[1] + 1) + block
    strides = (A.strides[0], A.strides[1]) + A.strides
    return as_strided(A, shape=shape, strides=strides)


def compute_laplacian(img, mask=None, eps=10**(-7), win_rad=1):
    """Computes Matting Laplacian for a given image.

    Args:
        img: 3-dim numpy matrix with input image
        mask: mask of pixels for which Laplacian will be computed.
            If not set Laplacian will be computed for all pixels.
        eps: regularization parameter controlling alpha smoothness
            from Eq. 12 of the original paper. Defaults to 1e-7.
        win_rad: radius of window used to build Matting Laplacian (i.e.
            radius of omega_k in Eq. 12).
    Returns: sparse matrix holding Matting Laplacian.
    """

    win_size = (win_rad * 2 + 1) ** 2
    h, w, d = img.shape
    # Number of window centre indices in h, w axes
    c_h, c_w = h - 2 * win_rad, w - 2 * win_rad
    win_diam = win_rad * 2 + 1

    indsM = np.arange(h * w).reshape((h, w))
    ravelImg = img.reshape(h * w, d)
    win_inds = _rolling_block(indsM, block=(win_diam, win_diam))

    win_inds = win_inds.reshape(c_h, c_w, win_size)
    if mask is not None:
        mask = cv2.dilate(
            mask.astype(np.uint8),
            np.ones((win_diam, win_diam), np.uint8)
        ).astype(np.bool)
        win_mask = np.sum(mask.ravel()[win_inds], axis=2)
        win_inds = win_inds[win_mask > 0, :]
    else:
        win_inds = win_inds.reshape(-1, win_size)

    
    winI = ravelImg[win_inds]

    win_mu = np.mean(winI, axis=1, keepdims=True)
    win_var = np.einsum('...ji,...jk ->...ik', winI, winI) / win_size - np.einsum('...ji,...jk ->...ik', win_mu, win_mu)

    inv = np.linalg.inv(win_var + (eps/win_size)*np.eye(3))

    X = np.einsum('...ij,...jk->...ik', winI - win_mu, inv)
    vals = np.eye(win_size) - (1.0/win_size)*(1 + np.einsum('...ij,...kj->...ik', X, winI - win_mu))

    nz_indsCol = np.tile(win_inds, win_size).ravel()
    nz_indsRow = np.repeat(win_inds, win_size).ravel()
    nz_indsVal = vals.ravel()
    L = scipy.sparse.coo_matrix((nz_indsVal, (nz_indsRow, nz_indsCol)), shape=(h*w, h*w))
    return L


def closed_form_matting_with_prior(image, prior, prior_confidence, consts_map=None):
    """Applies closed form matting with prior alpha map to image.

    Args:
        image: 3-dim numpy matrix with input image.
        prior: matrix of same width and height as input image holding apriori alpha map.
        prior_confidence: matrix of the same shape as prior hodling confidence of prior alpha.
        consts_map: binary mask of pixels that aren't expected to change due to high
            prior confidence.

    Returns: 2-dim matrix holding computed alpha map.
    """

    assert image.shape[:2] == prior.shape, ('prior must be 2D matrix with height and width equal '
                                            'to image.')
    assert image.shape[:2] == prior_confidence.shape, ('prior_confidence must be 2D matrix with '
                                                       'height and width equal to image.')
    assert (consts_map is not None) or image.shape[:2] == consts_map.shape, (
        'consts_map must be 2D matrix with height and width equal to image.')

    logging.info('Computing Matting Laplacian.')
    laplacian = compute_laplacian(image, ~consts_map if consts_map is not None else None)

    confidence = scipy.sparse.diags(prior_confidence.ravel())
    logging.info('Solving for alpha.')
    solution = scipy.sparse.linalg.spsolve(
        laplacian + confidence,
        prior.ravel() * prior_confidence.ravel()
    )
    alpha = np.minimum(np.maximum(solution.reshape(prior.shape), 0), 1)
    return alpha


def closed_form_matting_with_trimap(image, trimap, trimap_confidence=100.0):
    """Apply Closed-Form matting to given image using trimap."""

    assert image.shape[:2] == trimap.shape, ('trimap must be 2D matrix with height and width equal '
                                             'to image.')
    consts_map = (trimap < 0.1) | (trimap > 0.9)
    return closed_form_matting_with_prior(image, trimap, trimap_confidence * consts_map, consts_map)


def closed_form_matting_with_scribbles(image, scribbles, scribbles_confidence=100.0):
    """Apply Closed-Form matting to given image using scribbles image."""

    assert image.shape == scribbles.shape, 'scribbles must have exactly same shape as image.'
    consts_map = np.sum(abs(image - scribbles), axis=-1) > 0.001
    return closed_form_matting_with_prior(
        image,
        scribbles[:, :, 0],
        scribbles_confidence * consts_map,
        consts_map
    )


closed_form_matting = closed_form_matting_with_trimap

def main():
    import argparse

    logging.basicConfig(level=logging.INFO)
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument('image', type=str, help='input image')

    arg_parser.add_argument('-t', '--trimap', type=str, help='input trimap')
    arg_parser.add_argument('-s', '--scribbles', type=str, help='input scribbles')
    arg_parser.add_argument('-o', '--output', type=str, required=True, help='output image')
    arg_parser.add_argument(
        '--solve-fg', dest='solve_fg', action='store_true',
        help='compute foreground color and output RGBA image'
    )
    args = arg_parser.parse_args()

    image = cv2.imread(args.image, cv2.IMREAD_COLOR) / 255.0

    if args.scribbles:
        scribbles = cv2.imread(args.scribbles, cv2.IMREAD_COLOR) / 255.0
        alpha = closed_form_matting_with_scribbles(image, scribbles)
    elif args.trimap:
        trimap = cv2.imread(args.trimap, cv2.IMREAD_GRAYSCALE) / 255.0
        alpha = closed_form_matting_with_trimap(image, trimap)
    else:
        logging.error('Either trimap or scribbles must be specified.')
        arg_parser.print_help()
        exit(-1)

    if args.solve_fg:
        from solve_foreground_background import solve_foreground_background
        foreground, _ = solve_foreground_background(image, alpha)
        output = np.concatenate((foreground, alpha[:, :, np.newaxis]), axis=2)
    else:
        output = alpha

    cv2.imwrite(args.output, output * 255.0)


if __name__ == "__main__":
    main()
