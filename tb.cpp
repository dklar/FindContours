#include "top_level.hpp"
#include <hls_opencv.h>
#include <iostream>
#include <fstream>


using namespace std;


void testContours(){
	IplImage* src_image = new IplImage;

	AXI_STREAM src_stream;
	src_image = cvLoadImage("foto.jpg");
	IplImage2AXIvideo(src_image, src_stream);
	int8_t imageBuffer[MAX_HEIGHT*MAX_WIDTH];
	static Shape *shapes = (Shape*)malloc(sizeof(Shape)*1024);
	for(int i=0;i<1024;i++){
		Point s(0,0);
		Point e(0,0);

		shapes[i].start = s;
		shapes[i].end = e;
	}

	image_Preprocessing(src_stream,imageBuffer);

	FILE *file = fopen("data", "wb");
	fwrite(imageBuffer, sizeof(int8_t), MAX_HEIGHT * MAX_WIDTH, file);
	fclose(file);
	findContours_HLS<MAX_WIDTH,MAX_HEIGHT>(imageBuffer,shapes);
	cv::Mat mat = cv::imread("foto.jpg");
	for(int i=0;i<1024;i++){
		std::cout << "start x = " << shapes[i].start.col<< " y = " << shapes[i].start.row;
		std::cout << " end x = " << shapes[i].end.col  << " y = " << shapes[i].end.row  <<"\n";
		if (shapes[i].start.col !=0 || shapes[i].end.col != 0){
			cv::Point start(shapes[i].start.col,shapes[i].start.row);
			cv::Point end(shapes[i].end.col,shapes[i].end.row);
			cv::rectangle(mat, start, end, cv::Scalar(0, 255, 0));
		}
	}
	cv::imwrite("resultdraw.jpg",mat);
}


int main() {
	testContours();
}
