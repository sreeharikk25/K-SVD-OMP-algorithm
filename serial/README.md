# Serial Code for Image Denoising Using K-SVD

This program performs image denoising using the K-SVD algorithm. The program is implemented in C and leverages the `libpng` library for handling image files.

---

## Features

- Adds synthetic noise to an input image.
- Applies K-SVD for denoising using dictionary learning and sparse coding.
- Outputs noisy and denoised images in PNG format.

---

## Requirements

- **Compiler**: GCC
- **Libraries**:
  - `libpng` (for handling PNG files)
- **Input File**: `input.raw` (800x800 image data)
- **Output Files**:
  - `noisy_image.png`: Noisy image
  - `denoised_image.png`: Denoised image

---

## Build Instructions

1. Install the required library:
   ```bash
   sudo apt-get install libpng-dev
    ```
2. Compile the program:
    ```bash
   gcc -o denoise serial_main.c -lpng -lm
    ```

---

## How to Run
1. Place `input.raw` in the same directory as the compiled executable.

2. Run the program:
    ```bash
   ./denoise
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
