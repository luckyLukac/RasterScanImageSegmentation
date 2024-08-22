#pragma once


/// <summary>
/// Pixel structure.
/// </summary>
struct Pixel {
    int x;  // X coordinate
    int y;  // Y coordinate


    /// <summary>
    /// Default constructor.
    /// </summary>
    Pixel() = default;

    /// <summary>
    /// Main constructor for pixel.
    /// </summary>
    /// <param name="x">X coordinate</param>
    /// <param name="y">Y coordinate</param>
    Pixel(const int x, const int y) {
        this->x = x;
        this->y = y;
    }


    /// <summary>
    /// Equality operator for two pixels.
    /// </summary>
    /// <param name="pixel">Pixel to be compared to</param>
    /// <returns>True if the two pixels are the same</returns>
    bool operator==(const Pixel& pixel) const {
        return x == pixel.x && y == pixel.y;
    }

    /// <summary>
    /// Comparison operator for two pixels.
    /// </summary>
    /// <param name="pixel">Pixel to be compared to</param>
    /// <returns>True if the pixel is smaller than the other</returns>
    bool operator<(const Pixel& pixel) const {
        return x < pixel.x || (x == pixel.x && y < pixel.y);
    }
};


/// <summary>
/// Hash function for the Pixel struct.
/// </summary>
template <>
struct std::hash<Pixel> {
    std::size_t operator()(const Pixel& pixel) const {
        return std::hash<int>()(pixel.x) ^ (std::hash<int>()(pixel.y) << 1);
    }
};