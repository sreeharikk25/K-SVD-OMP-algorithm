# Parallel Image Denoising Using K-SVD with CUDA

This code implements the K-SVD  algorithm for denoising images. It leverages CUDA for GPU acceleration, enabling efficient processing of high-resolution images.

---

## Features

- Utilizes CUDA for parallelizing sparse coding and dictionary learning in K-SVD.
- Adds synthetic noise to an input image for testing.
- Produces both noisy and denoised images as output in PNG format.

---

## Requirements

### Software
- **CUDA Toolkit** (version 10.0 or higher recommended)
- **Compiler**: `nvcc` (part of the CUDA Toolkit)
- **Libraries**:
  - `libpng` for handling PNG image files

### Hardware
- NVIDIA GPU with CUDA support

---

## Build Instructions

1. Install the CUDA Toolkit from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads).

2. Install `libpng` for PNG file handling:
   ```bash
   sudo apt-get install libpng-dev
   ```
2. Compile the program using `nvcc`:
    ```bash
   nvcc -o denoise_cuda cuda_main.cu -lpng -lm
    ```

---

## How to Run
1. Place `input.raw` in the same directory as the compiled executable.

2. Run the program:
    ```bash
   ./denoise_cuda
    ```
3. The program will:
    - Add noise to the input image and save it as `noisy_image.png`.
    - Perform denoising using K-SVD and save the result as `denoised_image.png`.

---

## Dependencies
1. CUDA Toolkit: Provides the compiler (nvcc) and runtime libraries for running CUDA programs.

2. libpng: For reading and writing PNG images. Install using:
    ```
    sudo apt-get install libpng-dev
    ```
