#!/usr/bin/env python
"""Setting up closed-form-matting package during pip installation"""

import os
import re

import setuptools

# Project root directory
root_dir = os.path.dirname(__file__)

# Get version string from __init__.py in the package
with open(os.path.join(root_dir, 'closed_form_matting', '__init__.py')) as f:
    version = re.search(r'__version__ = \'(.*?)\'', f.read()).group(1)

# Get dependency list from requirements.txt
with open(os.path.join(root_dir, 'requirements.txt')) as f:
    requirements = f.read().split()

setuptools.setup(
    name='closed-form-matting',
    version=version,
    author='Marco Forte',
    author_email='fortemarco.irl@gmail.com',
    maintainer='Marco Forte',
    maintainer_email='fortemarco.irl@gmail.com',
    url='https://github.com/MarcoForte/closed-form-matting',
    description='A closed-form solution to natural image matting',
    long_description=open(os.path.join(root_dir, 'README.md')).read(),
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Image Processing',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    keywords=['closed-form matting', 'image matting', 'image processing'],
    license='MIT',
    python_requires='>=3.5',
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'closed-form-matting=closed_form_matting.closed_form_matting:main',
            'solve-foreground-background=closed_form_matting.solve_foreground_background:main',
        ],
    },
)
