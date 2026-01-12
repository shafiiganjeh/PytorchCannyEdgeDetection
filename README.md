# PyTorch Canny Edge Detection

Fully vectorized PyTorch implementation of the Canny edge detection algorithm with hysteresis edge tracking, packaged as a PyTorch module. This project is designed to run efficiently on a GPU and scale to high-resolution images and image batches.

## Usage

You can import and use the module like a regular PyTorch module:

```python
import torch
import edge_detector as ce

# create detector (note: original parameter names preserved)
getedge = ce.c_edge(upper_threshold=40, lower_threshold=20, max_iterations=10)

```

Input — grayscale image (UTF-8) tensor with shape (Batch, 1, Height, Width).

Output — torch.int32 tensor of shape (Batch, 1, Height, Width) containing detected edges.

Parameters
- kernel_size_gauss: int — kernel size for the Gaussian blur.
- sigma_gauss: float — standard deviation for the Gaussian blur.
- kernel_size_sobel: int — kernel size for the Sobel operator.
- upper_threshold: float — high threshold for double thresholding.
- lower_threshold: float — low threshold for double thresholding.
- max_iterations: int — maximum iterations for the hysteresis / propagation step.
- padding_mode: one of 'zeros', 'reflect', 'replicate', 'circular' — padding mode used for convolutions.
- precision: either torch.float32 or torch.float16 — numeric precision used for intermediate calculations.

There are two examples:
- `test.py` — simple example for processing an image batch.
- `webcam_test.py` — real-time video conversion using a webcam.

## Steps and implementation details

<img src="resources/example.jpeg" alt="original" width="500" style="display: block; margin: auto;" />
<p style="text-align: center;">original</p>

1. Blur the grayscale image using a Gaussian filter to reduce noise.

<img src="resources/blur.jpg" alt="blur" width="500" style="display: block; margin: auto;" />

2. Compute image gradients using a Sobel filter.

<img src="resources/gradients.jpg" alt="gradients" width="500" style="display: block; margin: auto;" />

3. Apply non-maximum suppression to thin edges, then apply double thresholding to classify strong and weak edge pixels.

<img src="resources/threshold.jpg" alt="threshold" width="500" style="display: block; margin: auto;" />

4. Follows chains of pixels to connect weak edge pixels to strong ones via edge tracking by hysteresis, If there is no connection to a strong pixel the weak pixel is discarted.

<img src="resources/final.jpg" alt="final" width="500" style="display: block; margin: auto;" />

The Gaussian blur and Sobel kernels are implemented via separable convolution and their kernel sizes are configurable. Edge tracking by hysteresis is implemented using a parallel connected-components labeling algorithm to run efficinetly on a GPU, the algorithm is iterative. Usually a maximum of 5-8 iterations are more than enough.
