#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "vector"

#define ATD at<double>
#define vector vector<Mat>
#define i32 int
#define i64 long long
#define db double

using namespace cv;
using namespace std;

i32 get_id(i32 i, i32 j, i32 width) {
	return i * width + j;
}
// 求系数矩阵A
Mat getA(i32 height, i32 width) {
	Mat A = Mat::eye(height * width, height * width, CV_64FC1);
	A *= -4;
	for (i32 i = 0; i < height; i++) {
		for (i32 j = 0; j < width; j++) {
			i32 id = get_id(i, j, width);
			if (i != 0) A.ATD(id, get_id(i - 1, j, width)) = 1;
			if (i != height - 1) A.ATD(id, get_id(i + 1, j, width)) = 1;
			if (j != 0) A.ATD(id, get_id(i, j - 1, width)) = 1;
			if (j != width - 1) A.ATD(id, get_id(i, j + 1, width)) = 1;
		}
	}
	return A;
}
// 求源图像散度
i32 get_div(Mat img, i32 i, i32 j, i32 roi_height, i32 roi_width) {
	i32 x1, x2, y1, y2;
	if (j != 0) x1 = img.ATD(i, j - 1);
	else x1 = img.ATD(i, j);
	if (j != roi_width - 1) x2 = img.ATD(i, j + 1);
	else x2 = img.ATD(i, j);
	if (i != 0) y1 = img.ATD(i - 1, j);
	else y1 = img.ATD(i, j);
	if (i != roi_height - 1) y2 = img.ATD(i + 1, j);
	else y2 = img.ATD(i, j);
	return x1 + x2 + y1 + y2 - 4 * img.ATD(i, j);
}
// 求b
Mat getb(Mat img1, Mat img2, i32 roix, i32 roiy, i32 roi_height, i32 roi_width) {
	Mat B = Mat::zeros(roi_height * roi_width, 1, CV_64FC1);
	for (i32 i = 0; i < roi_height; i++) {
		for (i32 j = 0; j < roi_width; j++) {
			i32 div = get_div(img1, i, j, roi_height, roi_width);
			if (i == 0) div -= img2.ATD(roiy + i - 1, roix + j);
			if (i == roi_height - 1) div -= img2.ATD(roiy + i + 1, roix + j);
			if (j == 0) div -= img2.ATD(roiy + i, roix + j - 1);
			if (j == roi_width - 1) div -= img2.ATD(roiy + i, roix + j + 1);
			i32 id = get_id(i, j, roi_width);
			B.ATD(id, 0) = div;
		}
	}
	return B;
}

Mat get_result(Mat A, Mat b, i32 height) {
	Mat result;
	solve(A, b, result);
	result = result.reshape(0, height);
	return result;
}

Mat poisson_blending(i32 roix, i32 roiy, Mat img1, Mat img2, i32 roi_height, i32 roi_width, Mat A) {
	vector rgb1, rgb2, result;
	// 三个通道分别求解，最后再叠加
	split(img1, rgb1);
	split(img2, rgb2);

	Mat b, res;
	b = getb(rgb1[0], rgb2[0], roix, roiy, roi_height, roi_width);
	res = get_result(A, b, roi_height);
	result.push_back(res);

	b = getb(rgb1[1], rgb2[1], roix, roiy, roi_height, roi_width);
	res = get_result(A, b, roi_height);
	result.push_back(res);

	b = getb(rgb1[2], rgb2[2], roix, roiy, roi_height, roi_width);
	res = get_result(A, b, roi_height);
	result.push_back(res);

	Mat v;
	merge(result, v);

	return v;
}

i32 main() {
	Mat img1, img2;
	Mat i1 = imread("C:/Users/lynn/Desktop/moon.png");
	Mat i2 = imread("C:/Users/lynn/Desktop/cuhk.jfif");
	imshow("image1", i1);
	imshow("image2", i2);
	i1.convertTo(img1, CV_64FC3);
	i2.convertTo(img2, CV_64FC3);

	i32 roix, roiy, roi_height, roi_width;
	roix = 200;
	roiy = 50;

	roi_height = img1.rows;
	roi_width = img1.cols;

	Mat A = getA(roi_height, roi_width);
	Mat Result = poisson_blending(roix, roiy, img1, img2, roi_height, roi_width, A);

	Rect rc = Rect(roix, roiy, roi_width, roi_height);
	Mat T = img2(rc);
	Result.copyTo(T);
	img2.convertTo(img2, CV_8UC1);
	imshow("result", img2);

	imwrite("C:/Users/lynn/Desktop/result.jpg", img2);

	waitKey(0);
}
