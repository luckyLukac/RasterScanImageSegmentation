#include <cmath>
#include <numeric>

#include "RawImage.h"


RawImage::RawImage(const std::vector<std::vector<int>>& pixelGrid) {
	m_PixelGrid = pixelGrid;
}

RawImage::RawImage(const int x, const int y, const int value) {
	m_PixelGrid = std::vector<std::vector<int>>(y, std::vector<int>(x, value));
}


bool RawImage::operator==(const RawImage& image) const {
	// Assertion of the same size of images.
	const int imageWidth = width();
	const int imageHeight = height();
	if (width() != image.width() || height() != image.height()) {
		return false;
	}

	// Pixel check.
	for (int y = 0; y < imageHeight; y++) {
		for (int x = 0; x < imageWidth; x++) {
			if (this->at(x, y) != image.at(x, y)) {
				return false;
			}
		}
	}

	return true;
}

RawImage RawImage::operator-(const RawImage& image) const {
	// Assertion of the same size of images.
	const int imageWidth = width();
	const int imageHeight = height();
	if (width() != image.width() || height() != image.height()) {
		throw "Images should be the same size when calculating differences.";
	}

	// Difference calculation.
	std::vector<std::vector<int>> pixelGrid(imageHeight, std::vector<int>(imageWidth));
	for (int y = 0; y < imageHeight; y++) {
		for (int x = 0; x < imageWidth; x++) {
			pixelGrid[y][x] = this->at(x, y) - image.at(x, y);
		}
	}

	return RawImage(pixelGrid);
}


int RawImage::width() const {
	if (m_PixelGrid.empty()) {
		return 0;
	}
	
	return static_cast<int>(m_PixelGrid[0].size());
}

int RawImage::height() const {
	return static_cast<int>(m_PixelGrid.size());
}

int RawImage::size() const {
	return width() * height();
}

int RawImage::byteCount() const {
	return width() * height();
}

int RawImage::bitCount() const {
	return 8 * byteCount();
}

std::vector<int> RawImage::histogram() const {
	const auto [min, max] = minAndMaxPixelValues();  // Extraction of minimum and maximum pixel values.

	// Creating a histogram of pixels.
	std::vector<int> histogram(max - min + 1, 0);
	for (int y = 0; y < height(); y++) {
		for (int x = 0; x < width(); x++) {
			histogram[m_PixelGrid[y][x] - min]++;
		}
	}

	return histogram;
}

double RawImage::entropy() const {
	// Histogram calculation.
	const std::vector<int> histogram = this->histogram();
	const double sum = std::reduce(histogram.begin(), histogram.end());

	// Entropy calculation.
	double entropy = 0.0;
	for (const int value : histogram) {
		if (value > 0) {
			entropy += (value / sum) * std::log2(value / sum);
		}
	}

	return -entropy;
}

int RawImage::at(const int x, const int y) const {
	return m_PixelGrid[y][x];
}

std::vector<int> RawImage::at(const std::vector<Pixel>& pixels) const {
	std::vector<int> values;

	// Obtaining values of pixels.
	for (const Pixel& pixel : pixels) {
		values.push_back(at(pixel.x, pixel.y));
	}

	return values;
}

void RawImage::setAt(const int x, const int y, const int value) {
	m_PixelGrid[y][x] = value;
}

std::vector<int> RawImage::imageTo1DSequence(const bool switchPixelOrderInConsecutiveLines) const {
	std::vector<int> sequence;
	const int imageHeight = height();
	const int imageWidth = width();

	// Building a sequence.
	for (int y = 0; y < imageHeight; y++) {
		if (y % 2 == 0 || !switchPixelOrderInConsecutiveLines) {
			for (int x = 0; x < imageWidth; x++) {
				sequence.push_back(this->at(x, y));
			}
		}
		else {
			for (int x = imageWidth - 1; x >= 0; x--) {
				sequence.push_back(this->at(x, y));
			}
		}
	}

	return sequence;
}

std::vector<Pixel> RawImage::imageTo1DPixelSequence() const {
	std::vector<Pixel> sequence;
	const int imageHeight = height();
	const int imageWidth = width();

	// Building a sequence.
	for (int y = 0; y < imageHeight; y++) {
		for (int x = 0; x < imageWidth; x++) {
			sequence.push_back(Pixel(x, y));
		}
	}

	return sequence;
}

std::pair<int, int> RawImage::minAndMaxPixelValues() const {
	// Initial values are set to extreme.
	int minValue = std::numeric_limits<int>::max();
	int maxValue = std::numeric_limits<int>::min();

	// Iteration through an image.
	for (const auto& row : m_PixelGrid) {
		for (const int value : row) {
			minValue = std::min(minValue, value);
			maxValue = std::max(maxValue, value);
		}
	}

	return { minValue, maxValue };
}
