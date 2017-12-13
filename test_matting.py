"""Tests for Closed-Form matting and foreground/background solver."""
import unittest

import cv2
import numpy as np

import closed_form_matting
from solve_foreground_background import solve_foreground_background

class TestMatting(unittest.TestCase):
    def test_solution_close_to_original_implementation(self):
        image = cv2.imread('testdata/source.png', cv2.IMREAD_COLOR) / 255.0
        scribles = cv2.imread('testdata/scribbles.png', cv2.IMREAD_COLOR) / 255.0

        alpha = closed_form_matting.closed_form_matting_with_scribbles(image, scribles)
        foreground, background = solve_foreground_background(image, alpha)

        matlab_alpha = cv2.imread('testdata/matlab_alpha.png', cv2.IMREAD_GRAYSCALE) / 255.0
        matlab_foreground = cv2.imread('testdata/matlab_foreground.png', cv2.IMREAD_COLOR) / 255.0
        matlab_background = cv2.imread('testdata/matlab_background.png', cv2.IMREAD_COLOR) / 255.0

        sad_alpha = np.mean(np.abs(alpha - matlab_alpha))
        sad_foreground = np.mean(np.abs(foreground - matlab_foreground))
        sad_background = np.mean(np.abs(background - matlab_background))

        self.assertLess(sad_alpha, 1e-2)
        self.assertLess(sad_foreground, 1e-2)
        self.assertLess(sad_background, 1e-2)

    def test_matting_with_trimap(self):
        image = cv2.imread('testdata/source.png', cv2.IMREAD_COLOR) / 255.0
        trimap = cv2.imread('testdata/trimap.png', cv2.IMREAD_GRAYSCALE) / 255.0

        alpha = closed_form_matting.closed_form_matting_with_trimap(image, trimap)

        reference_alpha = cv2.imread('testdata/output_alpha.png', cv2.IMREAD_GRAYSCALE) / 255.0

        sad_alpha = np.mean(np.abs(alpha - reference_alpha))
        self.assertLess(sad_alpha, 1e-3)
