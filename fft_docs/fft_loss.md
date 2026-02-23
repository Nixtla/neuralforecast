# PR Description: Integration of Fourier-Domain Loss Functions

## Overview
This PR introduces a suite of frequency-domain loss functions and a flexible `MixedFFTLoss` utility. These additions allow the model to optimize for spectral density and periodic structures, which provides a more robust loss function to trend and seasonality vs direct pointwise prediction errors.

### Fourier Loss Suite
I have implemented three core frequency-domain losses based on the magnitude spectrum of the Real Discrete Fourier Transform (RFFT):
- `FFTMAELoss`: Mean Absolute Error in frequency space.
- `FFTMSELoss`: Mean Squared Error in frequency space.
- `FFTRMSELoss`: Root Mean Squared Error in frequency space.

These losses operate on the magnitude spectrum $|F(y)|$ which ensures the loss remains real-valued, focuses on the power distribution of seasonal and trend components rather than exact point-in-time alignment.

### Hybrid Optimization (`MixedFFTLoss`)
To balance point-wise accuracy with structural frequency alignment, I added `MixedFFTLoss`. This allows for a composite objective function. This function provides the best balance for real-world use cases:

$$\mathcal{L}_{total} = \mathcal{L}_{time} + \lambda \cdot \mathcal{L}_{freq}$$

### Key Features:
- **Zero-Masking:** Masks are applied in the time-domain prior to FFT to prevent padding or missing values from introducing high-frequency noise into the spectrum.
- **Normalization:** Supports magnitude normalization via the `norm` parameter to ensure loss stability across varying sequence lengths ($H$).
- **Architectural Compatibility:** The implementation respects the `BasePointLoss` interface to seamlessly integrate with existing loss functions in the repository. 

### Testing
Tests were performed locally via PyTest. The baseline formulas were computed in pure numpy and compared against the pytorch functions implemented. The loss functions are numerically stable and work within the NeuralForecast repository.

## Testing

Tests are located in `tests/test_losses/test_fft_losses.py` and run via PyTest. Each of the four loss classes (`FFTMAELoss`, `FFTMSELoss`, `FFTRMSELoss`, `MixedFFTLoss`) has four dedicated test cases:

| Test | Description |
|---|---|
| **Univariate correctness** | Passes identical tensors as `y` and `y_hat`; asserts loss is exactly zero. |
| **Multivariate correctness** | Runs the loss on random `[B, H, N]` tensors with `N > 1` outputs; asserts the result is positive, finite, and — for RMSE — equal to the square root of the corresponding MSE loss. |
| **Autoregressive masking** | Zeros out a trailing or leading slice of the time dimension (simulating causal / AR-style masking) and asserts the masked loss is finite and non-negative. Also verifies that `MixedFFTLoss` raises `TypeError` for invalid loss type pairings. |
| **Numerical stability** | Runs each loss at extreme input scales (`1e-8` and `1e8`) with `norm=True` and asserts no `NaN` or `Inf` values are produced. |

All tests pass locally; note `torch` and `pytest` are required to run tests.