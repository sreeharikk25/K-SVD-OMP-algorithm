# Parallel Image Denoising Using K-SVD with OpenMP

This code implements the K-SVD  algorithm for denoising images. It leverages OpenMP for parallelization to improve performance during sparse coding and dictionary updates.

---

## Features

- Denoises the image using K-SVD with dictionary learning and sparse coding.
- Utilizes OpenMP for parallel processing, enabling faster computations.
- Outputs noisy and denoised images in PNG format.

---

## Requirements

- **Compiler**: GCC with OpenMP support (`-fopenmp`).
- **Libraries**:
  - `libpng` (for handling PNG files).
- **Input File**: `input.raw` (800x800 image).
- **Output Files**:
  - `noisy_image.png`: Noisy image.
  - `denoised_image.png`: Denoised image.

---

## Build Instructions

1. Install the required library:
   ```bash
   sudo apt-get install libpng-dev
    ```
2. Compile the program:
    ```bash
   gcc -o denoise_openmp openmp_main.c -lpng -lm -fopenmp
    ```

---

## How to Run
1. Place `input.raw` in the same directory as the compiled executable.

2. Run the program:
    ```bash
   ./denoise_openmp
    ```
3. The program will:
    - Add noise to the input image and save it as `noisy_image.png`.
    - Perform denoising using K-SVD and save the result as `denoised_image.png`.

---

## Dependencies
1. Ensure libpng is installed on your system. For Debian-based systems:

    ```
    sudo apt-get install libpng-dev
    ```
