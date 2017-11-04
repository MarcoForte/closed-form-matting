# Closed-Form Matting
Python implementation of image matting method proposed in A. Levin D. Lischinski and Y. Weiss. A Closed Form Solution to Natural Image Matting. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), June 2006, New York

The repository also contains implementation of background/foreground reconstruction method proposed in Levin, Anat, Dani Lischinski, and Yair Weiss. "A closed-form solution to natural image matting." IEEE Transactions on Pattern Analysis and Machine Intelligence 30.2 (2008): 228-242.

## Requirements
- python 3.5+ (Though it should run on 2.7)
- scipy
- numpy
- matplotlib
- sklearn
- opencv-python (for CLI of `solve_foregound_background.py`)


## Usage

### Closed-Form matting
- mylambda (λ) is a constant controlling the users confidence in the constraints.
- eps (ε) is a constant controlling the smoothness of alpha.

Running the demo:

```bash
python closed_form_matting.py
```

Python interface:

```python
from closed_form_matting import closed_form_matte
...
alpha = closed_form_matte(image, scribbled_img)
```

### Foreground and Background Reconstruction
CLI interface (requires opencv-python):

```bash
./solve_foregound_background.py image.png alpha.png foreground.png background.png
```

Python interface:

```python
from solve_foregound_background import solve_foregound_background
...
foreground, background = solve_foregound_background(image, alpha)
```

## Results
![Original image](https://github.com/MarcoForte/closed-form-matting/blob/master/dandelion_clipped.bmp)
![Scribbled image](https://github.com/MarcoForte/closed-form-matting/blob/master/dandelion_clipped_m.bmp)
![Result](https://github.com/MarcoForte/closed-form-matting/blob/master/dandelion_clipped_alpha.bmp)


## More Information

This version of matting laplacian does not support computation over only unknown regions. The computation is generally faster than the matlab version regardless thanks to more vectorisation.
Note. The computed laplacian is slightly different due to array ordering in numpy being different than in matlab. To get same laplacian as in matlab change,

`indsM = np.arange(h*w).reshape((h, w))`
`ravelImg = img.reshape(h*w, d)`
to
`indsM = np.arange(h*w).reshape((h, w), order='F')`
`ravelImg = img.reshape(h*w, d, , order='F')`.
Again note that this will result in incorrect alpha if the `D_s, b_s` orderings are not also changed to `order='F'F`.

For more information see the orginal paper  http://www.wisdom.weizmann.ac.il/~levina/papers/Matting-Levin-Lischinski-Weiss-CVPR06.pdf
The original matlab code is here http://www.wisdom.weizmann.ac.il/~levina/matting.tar.gz

## Disclaimer

The code is free for academic/research purpose. Use at your own risk and we are not responsible for any loss resulting from this code. Feel free to submit pull request for bug fixes.
