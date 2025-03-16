#include "mnist_loader.hpp"
#include <fstream>
#include <iostream>
#include <vector>

// Function to read MNIST images from the binary file
std::vector<Matrix> loadMNISTImages(const std::string &filename) {
    // Open the file in binary mode
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        return {};  // Return empty vector if file is not found
    }

    // MNIST file headers contain metadata that describes the dataset
    int magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;

    // char magicNumber[4];  // loading data
    // char numOfImages[4];
    // char numOfRows[4];
    // char numOfCols[4];
    // file.read(magicNumber, 4);
    // file.read(numOfImages, 4);
    // file.read(numOfRows, 4);
    // file.read(numOfCols, 4);
    
    // Read first 16 bytes (metadata)
    file.read((char*)&magic_number, sizeof(magic_number));  // Magic number (4 bytes) 0-3
    file.read((char*)&num_images, sizeof(num_images));      // Number of images (4 bytes) 4-7
    file.read((char*)&num_rows, sizeof(num_rows));          // Image height (4 bytes) 8-11
    file.read((char*)&num_cols, sizeof(num_cols));          // Image width (4 bytes) 12-15

    // The numbers in the MNIST file are stored in Big-Endian format, (Most significant byte first)
    // but most systems use Little-Endian. We need to swap byte order. (LSB first)
    magic_number = __builtin_bswap32(magic_number);
    num_images = __builtin_bswap32(num_images);
    num_rows = __builtin_bswap32(num_rows);
    num_cols = __builtin_bswap32(num_cols);

    // Debugging Output
    std::cout << "Loading MNIST Images...\n";
    std::cout << "Magic Number: " << magic_number << "\n";
    std::cout << "Number of Images: " << num_images << "\n";
    std::cout << "Image Size: " << num_rows << "x" << num_cols << "\n";

    // Vector to store image matrices
    std::vector<Matrix> images;

    // Read each image, one by one
    for (int i = 0; i < num_images; i++) {
        Matrix img(num_rows, num_cols);  // Create a new Matrix for the image

        // Read pixel values row by row
        for (int r = 0; r < num_rows; r++) {
            for (int c = 0; c < num_cols; c++) {
                unsigned char pixel = 0;
                file.read((char*)&pixel, sizeof(pixel));  // Read 1 pixel (1 byte)
                img.data[r][c] = pixel / 255.0;  // Normalize pixel value to [0, 1]
            }
        }

        images.push_back(img);  // Store the image in the vector
    }

    // Close the file after reading
    file.close();

    // Return the vector of images
    return images;
}

// Function to read MNIST labels from the binary file
std::vector<int> loadMNISTLabels(const std::string &filename) {
    // Open the file in binary mode
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        return {};  // Return empty vector if file is not found
    }

    // MNIST label files contain a header describing the dataset
    int magic_number = 0, num_items = 0;

    // Read first 8 bytes (metadata)
    file.read((char*)&magic_number, sizeof(magic_number));  // Magic number (4 bytes)
    file.read((char*)&num_items, sizeof(num_items));        // Number of labels (4 bytes)

    // Convert Big-Endian to Little-Endian
    magic_number = __builtin_bswap32(magic_number);
    num_items = __builtin_bswap32(num_items);

    // Debugging Output (Optional)
    std::cout << "Loading MNIST Labels...\n";
    std::cout << "Magic Number: " << magic_number << "\n";
    std::cout << "Number of Labels: " << num_items << "\n";

    // Vector to store labels
    std::vector<int> labels(num_items);

    // Read labels one by one
    for (int i = 0; i < num_items; i++) {
        unsigned char label = 0;
        file.read((char*)&label, sizeof(label));  // Read 1 label (1 byte)
        labels[i] = static_cast<int>(label);  // Store as integer
    }

    // Close the file after reading
    file.close();

    // Return the vector of labels
    return labels;
}
