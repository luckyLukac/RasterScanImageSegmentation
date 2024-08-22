#include "ImageConverter.h"


std::vector<std::vector<int>> ImageConverter::QImageToRaw(const QImage& image) {
	// Initialization of the raw image.
	std::vector<std::vector<int>> pixelGrid(image.height(), std::vector<int>(image.width()));
	for (int y = 0; y < static_cast<int>(image.height()); y++) {
		for (int x = 0; x < static_cast<int>(image.width()); x++) {
			pixelGrid[y][x] = qGray(image.pixel(x, y));
		}
	}

	return pixelGrid;
}

QImage ImageConverter::rawImageToQImage(const RawImage& rawImage) {
	const int numRows = static_cast<int>(rawImage.height());
	const int numCols = static_cast<int>(rawImage.width());

	QImage result(numCols, numRows, QImage::Format_RGB32);

	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			const int pixelValue = rawImage.at(x, y);
			const QRgb color = qRgb(pixelValue, pixelValue, pixelValue); // Grayscale image
			result.setPixel(x, y, color);
		}
	}

	return result;
}

std::vector<std::vector<int>> ImageConverter::cvToRaw(const cv::Mat& image) {
	cv::Mat cvImage;
	image.convertTo(cvImage, CV_8U);

	// Initialization of the raw image.
	std::vector<std::vector<int>> pixelGrid(cvImage.rows, std::vector<int>(cvImage.cols));
	for (int y = 0; y < static_cast<int>(cvImage.rows); y++) {
		for (int x = 0; x < static_cast<int>(cvImage.cols); x++) {
			pixelGrid[y][x] = cvImage.at<uchar>(y, x);
		}
	}

	return pixelGrid;
}

cv::Mat ImageConverter::rawImageToCv(const RawImage& rawImage) {
	std::vector<int> data = rawImage.imageTo1DSequence();

	cv::Mat cvImage(rawImage.height(), rawImage.width(), CV_8UC1);
	for (int y = 0; y < rawImage.height(); y++) {
		for (int x = 0; x < rawImage.width(); x++) {
			cvImage.at<uchar>(y, x) = static_cast<uchar>(data[y * rawImage.width() + x]);
		}
	}
	return cvImage;
}
