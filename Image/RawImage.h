#pragma once
#include <string>
#include <vector>

#include "Pixel.hpp"


/// <summary>
/// Representation of a image in a raw format.
/// </summary>
class RawImage {
private:
    std::vector<std::vector<int>> m_PixelGrid;  // 2D pixel grid with grayscale pixel values [0-255].

public:
    /// <summary>
    /// Main constructor of the raw image.
    /// </summary>
    RawImage() = default;

    /// <summary>
    /// Constructor of the raw image.
    /// </summary>
    /// <param name="pixelGrid">2D vector of integers [0-255]</param>
    RawImage(const std::vector<std::vector<int>>& pixelGrid);

    /// <summary>
    /// Constructor of the raw image with the given X and Y dimensions.
    /// </summary>
    /// <param name="x">Number of columns</param>
    /// <param name="y">Number of rows</param>
    /// <param name="value">Value of each pixel in the newly created image</param>
    RawImage(const int x, const int y, const int value = 0);


    /// <summary>
    /// Check whether the given image is the same as the current one..
    /// </summary>
    /// <param name="image">Given image</param>
    /// <returns>True if the two images are the same, false otherwise</returns>
    bool operator==(const RawImage& image) const;

    /// <summary>
    /// Calculation of differences between the current and the given image.
    /// </summary>
    /// <param name="image">Second image</param>
    /// <returns>Raw image of differences</returns>
    RawImage operator-(const RawImage& image) const;


    /// <summary>
    /// Image width getter.
    /// </summary>
    /// <returns>Width of the image in pixels</returns>
    int width() const;

    /// <summary>
    /// Image height getter.
    /// </summary>
    /// <returns>Height of the image in pixels</returns>
    int height() const;

    /// <summary>
    /// Image size getter.
    /// </summary>
    /// <returns>Width * height</returns>
    int size() const;

    /// <summary>
    /// Calculation of a byte count in a raw image.
    /// </summary>
    /// <returns>Number of bytes</returns>
    int byteCount() const;

    /// <summary>
    /// Calculation of a bit count in a raw image.
    /// </summary>
    /// <returns>Number of bits</returns>
    int bitCount() const;

    /// <summary>
    /// Calculation of a histogram with pixel values, where n-th bin denotes the grayscale pixel with the value n.
    /// </summary>
    /// <returns>Vector of appereances of each value</returns>
    std::vector<int> histogram() const;

    /// <summary>
    /// Calculation of the image entropy value.
    /// </summary>
    /// <returns>Entropy</returns>
    double entropy() const;

    /// <summary>
    /// Accessing the image pixel value at coordinates X and Y.
    /// </summary>
    /// <param name="x">X coordinate</param>
    /// <param name="y">Y coordinate</param>
    /// <returns>Pixel value</returns>
    int at(const int x, const int y) const;

    /// <summary>
    /// Accessing the image pixels values at the coordinates, given with a list of pixels.
    /// </summary>
    /// <param name="pixels">List of pixels</param>
    /// <returns>Pixel values</returns>
    std::vector<int> at(const std::vector<Pixel>& pixels) const;

    /// <summary>
    /// Setting the image pixel at coordinates X and Y with the given value.
    /// </summary>
    /// <param name="x">X coordinate</param>
    /// <param name="y">Y coordinate</param>
    /// <param name="value">Given value [0-255]</param>
    void setAt(const int x, const int y, const int value);

    /// <summary>
    /// Transformation of the image into an 1D sequence.
    /// </summary>
    /// <param name="switchPixelOrderInConsecutiveLines">If true, pixels of each even row are ordered left to right while pixels of each odd row are ordered right to left</param>
    /// <returns>1D image sequence of pixels</returns>
    std::vector<int> imageTo1DSequence(const bool switchPixelOrderInConsecutiveLines = false) const;

    /// <summary>
    /// Transformation of pixels into an 1D sequence with their coordinates.
    /// </summary>
    /// <returns>1D sequence of pixel with coordinates</returns>
    std::vector<Pixel> imageTo1DPixelSequence() const;

    /// <summary>
    /// Extraction of minimum and maximum values of pixels inside the image.
    /// </summary>
    /// <returns>Minimum and maximum pixel value</returns>
    std::pair<int, int> minAndMaxPixelValues() const;
};