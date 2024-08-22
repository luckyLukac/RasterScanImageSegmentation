#include "Segmentation/ImageSegmentation.h"


// Sweep-line image segmentation.
std::vector<std::vector<Pixel>> segments;
RawImage segmentImage;
const double epsilon = 10.0;

std::tie(segments, segmentImage) = ImageSegmentation::segmentizeImageWithSweepLine(ImageConverter::rawImageToCv(m_OriginalImage), epsilon);