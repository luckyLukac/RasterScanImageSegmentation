#include <QImage>
#include "Segmentation/ImageSegmentation.h"


// Sweep-line image segmentation.
std::vector<std::vector<Pixel>> segments;
QImage segmentImage;
const double epsilon = 10.0;
const int maxDeviation = 50;
const ImageSegmentation::Distance distanceMetric = ImageSegmentation::Distance::oneNeighbor;

std::tie(segments, segmentImage) = ImageSegmentation::segmentizeImageWithSweepLine(ImageConverter::rawImageToCv(m_OriginalImage), epsilon, maxDeviation, distanceMetric, seed);
