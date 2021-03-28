#include "hls_video.h"
const int MAX_HEIGHT = 480;//1080;
const int MAX_WIDTH  = 640;//1920;

typedef hls::stream<ap_axiu<24, 1, 1, 1> > AXI_STREAM; //32 bit data stream
typedef hls::Mat<MAX_HEIGHT, MAX_WIDTH, HLS_8UC3> RGB_IMAGE; //RGB image from type HLS::Mat
typedef hls::Mat<MAX_HEIGHT, MAX_WIDTH, HLS_8UC1> GRAY_IMAGE; //Gray image from type HLS::Mat
typedef hls::Scalar<3, uint8_t> RGBPIXEL;
typedef hls::Scalar<1, uint8_t> PIXELGRAY;


struct Border {
	int seq_num;
	int border_type;
};

struct Point {
	uint16_t row;
	uint16_t col;

	Point() {
		row=0;
		col=0;
	}

	Point(uint16_t r, int c) {
		row = r;
		col = c;
	}

	void setPoint(uint16_t r, uint16_t c) {
		row = r;
		col = c;
	}

	bool samePoint(Point p) {
		return row == p.row && col == p.col;
	}
};

struct Pixel {
	unsigned char red;
	unsigned char blue;
	unsigned char green;

	Pixel(unsigned char r, unsigned char g, unsigned char b) {
		red = r;
		green = g;
		blue = b;
	}
	void setPixel(unsigned char r, unsigned char g, unsigned char b) {
		red = r;
		green = g;
		blue = b;
	}
};

struct Node {
	int parent;
	int first_child;
	int next_sibling;
	Border border;
	Node(int p, int fc, int ns) {
		parent = p;
		first_child = fc;
		next_sibling = ns;
	}
	Node(){
		parent = -1;
		first_child = -1;
		next_sibling = -1;
	}
	void reset() {
		parent = -1;
		first_child = -1;
		next_sibling = -1;
	}
};

struct Shape{
	Point start;
	Point end;
	uint16_t size;
	Shape(){
		start = Point(0,0);
		end = Point(0,0);
		size = 0;
	}
};



#define HOLE_BORDER 1
#define OUTER_BORDER 2


void inRange(RGB_IMAGE &in, GRAY_IMAGE &out) {
	for (int y = 0; y < MAX_HEIGHT; y++) {
		for (int x = 0; x < MAX_WIDTH; x++) {
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE II=1

			RGBPIXEL pixel;
			PIXELGRAY pxout;
			in >> pixel;
			int h = pixel.val[0];
			int s = pixel.val[1];
			int v = pixel.val[2];

			bool c1 = false;
			bool c2 = false;
			bool c3 = false;
			if (h > 21 && h < 126) {
				c1 = true;
			}
			if (s > 109 && s < 255) {
				c2 = true;
			}
			if (v > 57 && v < 255) {
				c3 = true;
			}
			if (c1 && c2 && c3) {
				pxout.val[0] = 255;
			} else {
				pxout.val[0] = 0;
			}
			out << pxout;
		}
	}
}

template<int WIDTH, int HEIGHT>
void Mat2Master(uint8_t *in,int8_t *picture){
	for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE II=1
			int p = in[y * WIDTH + x];
			if (p==255)
				picture[y*MAX_WIDTH+x] = 1;
			else
				picture[y*MAX_WIDTH+x] = 0;

		}
	}
}

template<int WIDTH, int HEIGHT>
void Gauss5(uint8_t *imageIn, uint8_t *imageOut) {

	const int K_SIZE = 5;
	uint8_t line_buf1[K_SIZE][WIDTH];
	uint8_t line_buf2[K_SIZE][WIDTH];
	uint8_t line_buf3[K_SIZE][WIDTH];
	uint8_t window_buf1[K_SIZE][K_SIZE];
	uint8_t window_buf2[K_SIZE][K_SIZE];
	uint8_t window_buf3[K_SIZE][K_SIZE];
	const int KERNEL[K_SIZE][K_SIZE] = { { 1, 4, 6, 4, 1 }, { 4, 16, 24, 16, 4 }, { 6,
			24, 36, 24, 6 }, { 4, 16, 24, 16, 4 }, { 1, 4, 6, 4, 1 } };

#pragma HLS ARRAY_PARTITION variable=window_buf1 complete dim=0
#pragma HLS ARRAY_PARTITION variable=window_buf2 complete dim=0
#pragma HLS ARRAY_PARTITION variable=window_buf3 complete dim=0
#pragma HLS ARRAY_PARTITION variable=KERNEL complete dim=0
#pragma HLS ARRAY_RESHAPE variable=line_buf1 complete dim=1
#pragma HLS ARRAY_RESHAPE variable=line_buf2 complete dim=1
#pragma HLS ARRAY_RESHAPE variable=line_buf3 complete dim=1


	gaussLoop:
	for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {
		#pragma HLS PIPELINE II=1
		#pragma HLS LOOP_FLATTEN off

			for (int yl = 0; yl < K_SIZE - 1; yl++) {
				line_buf1[yl][x] = line_buf1[yl + 1][x];
				line_buf2[yl][x] = line_buf2[yl + 1][x];
				line_buf3[yl][x] = line_buf3[yl + 1][x];
			}

			line_buf1[K_SIZE - 1][x] = imageIn[x * 3 + y * WIDTH * 3 + 0];
			line_buf2[K_SIZE - 1][x] = imageIn[x * 3 + y * WIDTH * 3 + 1];
			line_buf3[K_SIZE - 1][x] = imageIn[x * 3 + y * WIDTH * 3 + 2];

			for (int yw = 0; yw < K_SIZE; yw++) {
				for (int xw = 0; xw < K_SIZE - 1; xw++) {
					window_buf1[yw][xw] = window_buf1[yw][xw + 1];
					window_buf2[yw][xw] = window_buf2[yw][xw + 1];
					window_buf3[yw][xw] = window_buf3[yw][xw + 1];
				}
			}

			// write to window buffer
			for (int yw = 0; yw < K_SIZE; yw++) {
				window_buf1[yw][K_SIZE - 1] = line_buf1[yw][x];
				window_buf2[yw][K_SIZE - 1] = line_buf2[yw][x];
				window_buf3[yw][K_SIZE - 1] = line_buf3[yw][x];
			}

			int px_ch1 = 0;
			int px_ch2 = 0;
			int px_ch3 = 0;

			for (int yw = 0; yw < K_SIZE; yw++) {
				for (int xw = 0; xw < K_SIZE; xw++) {
					px_ch1 += window_buf1[yw][xw] * KERNEL[yw][xw];
					px_ch2 += window_buf2[yw][xw] * KERNEL[yw][xw];
					px_ch3 += window_buf3[yw][xw] * KERNEL[yw][xw];
				}
			}

			px_ch1 >>= 8;
			px_ch2 >>= 8;
			px_ch3 >>= 8;
			imageOut[x * 3 + y * WIDTH * 3 + 0] = px_ch1;
			imageOut[x * 3 + y * WIDTH * 3 + 1] = px_ch2;
			imageOut[x * 3 + y * WIDTH * 3 + 2] = px_ch3;
		}
	}
}

/**
* Morphological erode operation
* see https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html
* @tparam WIDTH width of the images
* @tparam HEIGHT height of the images
* @param src Source image
* @param dst destination image
**/
template<int WIDTH, int HEIGHT>
void erode(uint8_t *src, uint8_t *dst) {
	const int WINDOW_SIZE = 3;

	uint8_t line_buf[WINDOW_SIZE][WIDTH];
	uint8_t window_buf[WINDOW_SIZE][WINDOW_SIZE];

#pragma HLS ARRAY_RESHAPE variable=line_buf complete dim=1
#pragma HLS ARRAY_PARTITION variable=window_buf complete dim=0

	for (int i = 0; i < WINDOW_SIZE; i++) {
		for (int j = 0; j < WIDTH; j++) {
			line_buf[i][j] = 255;
		}
	}
	for (int i = 0; i < WINDOW_SIZE; i++) {
		for (int j = 0; j < WINDOW_SIZE; j++) {
			window_buf[i][j] = 255;
		}
	}

	erodeLoop: for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN off

			for (int i = 0; i < WINDOW_SIZE - 1; i++)
				line_buf[i][x] = line_buf[i + 1][x];

			line_buf[WINDOW_SIZE - 1][x] = src[x + y * WIDTH];

			for (int i = 0; i < WINDOW_SIZE; i++) {
				for (int j = 0; j < WINDOW_SIZE - 1; j++) {
					window_buf[i][j] = window_buf[i][j + 1];
				}
			}

			for (int i = 0; i < WINDOW_SIZE; i++)
				window_buf[i][WINDOW_SIZE - 1] = line_buf[i][x];

			uint8_t min = 255;

			for (int i = 0; i < WINDOW_SIZE; i++) {
				for (int j = 0; j < WINDOW_SIZE; j++) {
					min = min > window_buf[i][j] ? window_buf[i][j] : min;
				}
			}
			dst[x + y * WIDTH] = min;
		}
	}
}

/**
* Morphological dilate operation
* see https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html
* @tparam WIDTH width of the images
* @tparam HEIGHT height of the images
* @param src Source image
* @param dst destination image
**/
template<int WIDTH, int HEIGHT>
void dilate(uint8_t *src, uint8_t *dst) {
	const int WINDOW_SIZE = 3;

	uint8_t line_buf[WINDOW_SIZE][WIDTH];
	uint8_t window_buf[WINDOW_SIZE][WINDOW_SIZE];

#pragma HLS ARRAY_RESHAPE variable=line_buf complete dim=1
#pragma HLS ARRAY_PARTITION variable=window_buf complete dim=0

	for (int i=0;i<WINDOW_SIZE;i++){
		for (int j=0;j<WIDTH;j++){
			line_buf[i][j] = 0;
		}
	}
	for (int i=0;i<WINDOW_SIZE;i++){
		for (int j=0;j<WINDOW_SIZE;j++){
			window_buf[i][j] = 0;
		}
	}

	dilateLoop:
	for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN off


			for (int i = 0; i < WINDOW_SIZE - 1; i++)
				line_buf[i][x] = line_buf[i + 1][x];

			line_buf[WINDOW_SIZE - 1][x] = src[x + y * WIDTH];


			for (int i = 0; i < WINDOW_SIZE; i++) {
				for (int j = 0; j < WINDOW_SIZE - 1; j++) {
					window_buf[i][j] = window_buf[i][j + 1];
				}
			}

			for (int i = 0; i < WINDOW_SIZE; i++)
				window_buf[i][WINDOW_SIZE - 1] = line_buf[i][x];

			uint8_t max = 0;
			for (int i = 0; i < WINDOW_SIZE; i++) {
				for (int j = 0; j < WINDOW_SIZE; j++) {
					max = max < window_buf[i][j] ? window_buf[i][j] : max;
				}
			}
			dst[x + y * WIDTH] = max;
		}
	}
}

/**
* Close small holes in a picture by executing
* first erosion then dilation.
* Used here to minimize the probability of a missdetection
* of a circle. (the algorithm to detect circles is very
* sensitive to holes.
* @tparam WIDTH width of the images
* @tparam HEIGHT height of the images
* @param src Source image
* @param dst destination image
**/
template<int WIDTH, int HEIGHT>
void morphOpening(uint8_t *src, uint8_t *dst){
#pragma HLS DATAFLOW
	uint8_t fifo1[WIDTH * HEIGHT];
#pragma HLS STREAM variable=fifo1 depth=1 dim=1
	erode<WIDTH, HEIGHT>(src, fifo1);
	dilate<WIDTH, HEIGHT>(fifo1,dst);
}

template<int WIDTH,int HEIGHT>
void inRange(uint8_t *in, uint8_t *out,uint8_t threshold[6]) {
	for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE II=1

			uint8_t ch[]={0,0,0};
			for(int i=0;i<3;i++){
				ch[i] = in[y * WIDTH * 3 + x * 3 + i];
			}

			int h = ch[0];
			int s = ch[1];
			int v = ch[2];

			uint8_t pxout;

			bool c1 = false;
			bool c2 = false;
			bool c3 = false;
			if (h > threshold[0] && h < threshold[1])
				c1 = true;
			if (s > threshold[2] && s < threshold[3])
				c2 = true;
			if (v > threshold[4] && v < threshold[5])
				c3 = true;
			if (c1 && c2 && c3)
				pxout = 255;
			else
				pxout = 0;
			out[y*WIDTH+x] = pxout;
		}
	}
}

template<int WIDTH, int HEIGHT>
void MatToArray(RGB_IMAGE &in, uint8_t* out) {
	RGBPIXEL pixel_value;
	loopPixel: for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {
#pragma HLS loop_flatten off
#pragma HLS pipeline II=1
			in >> pixel_value;

			for (int i=0;i<3;i++){
#pragma HLS unroll
				out[y * WIDTH * 3 + x * 3 + i] = pixel_value.val[i];
			}

		}
	}
}

template<int WIDTH, int HEIGHT>
void convertColor(uint8_t *in, uint8_t *out) {
	RGBPIXEL pixel_value;
	loopPixel: for (int y = 0; y < HEIGHT; y++) {
		for (int x = 0; x < WIDTH; x++) {
#pragma HLS loop_flatten off
#pragma HLS pipeline II=1

			int ch[3] = { 0, 0, 0 };

			int max = 0;
			int min = 255;
			for (int i = 0; i < 3; i++) {
#pragma HLS unroll
				ch[i] = in[y * WIDTH * 3 + x * 3 + i];
				max = ch[i] > max ? ch[i] : max;
				min = ch[i] < min ? ch[i] : min;
			}
			float H = 0;
			float S = 0;
			float V = max;
			if (V != 0)
				S = (V - min) / V;

			int R = ch[2];
			int G = ch[1];
			int B = ch[0];
			if (V == R)
				H = 60 * (G - B) / (V - min);
			else if (V == G)
				H = 120 + 60 * (B - R) / (V - min);
			else if (V == B)
				H = 240 + 60 * (R - G) / (V - min);
			if (R == G && G == B && R == 0)
				H = 0;

			if (H < 0)
				H = H + 360;
			uint8_t h_out = (uint8_t)(H/2);
			uint8_t s_out = (uint8_t)(S*255);
			uint8_t v_out = (uint8_t)(V*255);
			out[y * WIDTH * 3 + x * 3 + 0] = h_out;
			out[y * WIDTH * 3 + x * 3 + 1] = s_out;
			out[y * WIDTH * 3 + x * 3 + 2] = v_out;
		}
	}
}

template<int width, int height>
void duplicate(uint8_t *in, uint8_t *out1, uint8_t* out2) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN off
			out1[x + y * width] = in[x + y * width];
			out2[x + y * width] = in[x + y * width];
		}
	}
}

template<int width, int height>
void toSigendArray(uint8_t *in, int8_t *out){
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN off
			out[x + y * width] = (int8_t)in[x + y * width];
		}
	}
}

template<int width, int height>
void zeroBorder(uint8_t *in,uint8_t *out){
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN off
			if (y<5 || y> height-5 || x<5 ||x>width-5){
				out[x + y * width] = 0;
			}else{
				out[x + y * width] = in[x + y * width];
			}

		}
	}
}

void stepCW(Point &current, Point pivot) {
	if (current.col > pivot.col)
		current.setPoint(pivot.row + 1, pivot.col);
	else if (current.col < pivot.col)
		current.setPoint(pivot.row - 1, pivot.col);
	else if (current.row > pivot.row)
		current.setPoint(pivot.row, pivot.col - 1);
	else if (current.row < pivot.row)
		current.setPoint(pivot.row, pivot.col + 1);
}

void stepCCW(Point &current, Point pivot) {
	if (current.col > pivot.col)
		current.setPoint(pivot.row - 1, pivot.col);
	else if (current.col < pivot.col)
		current.setPoint(pivot.row + 1, pivot.col);
	else if (current.row > pivot.row)
		current.setPoint(pivot.row, pivot.col + 1);
	else if (current.row < pivot.row)
		current.setPoint(pivot.row, pivot.col - 1);
}

bool pixelOutOfBounds(Point p, int numrows, int numcols) {
	return (p.col >= numcols || p.row >= numrows || p.col < 0 || p.row < 0);
}

void markExamined(Point mark, Point center, bool checked[4]) {
	//p3.row, p3.col + 1
	int loc = -1;
	//    3
	//  2 x 0
	//    1
	if (mark.col > center.col)
		loc = 0;
	else if (mark.col < center.col)
		loc = 2;
	else if (mark.row > center.row)
		loc = 1;
	else if (mark.row < center.row)
		loc = 3;

	checked[loc] = true;
	return;
}

bool isExamined(bool checked[4]) {
	return checked[0];
}

template<int WIDTH, int HEIGHT>
void followBorder(int8_t* image, Point p2, int row, int col, Border NBD, Shape& shape) {
	Point current(p2.row, p2.col);
	Point start(row, col);

	const int h = HEIGHT;
	const int w = WIDTH;

	do {
		stepCW(current, start);
		if (current.samePoint(p2)) {
			image[start.row * WIDTH + start.col] = -NBD.seq_num;
			return;
		}
	} while (pixelOutOfBounds(current, h, w) || image[current.row * w + current.col] == 0);
	Point p1 = current;


	Point p3 = start;
	Point p4;
	p2 = p1;
	bool checked[4];

	while (true) {

		current = p2;

		for (int i = 0; i < 4; i++)
			checked[i] = false;

		do {
			markExamined(current, p3, checked);
			stepCCW(current, p3);
		} while (pixelOutOfBounds(current, h, w) || image[current.row * w + current.col] == 0);
		p4 = current;


		if ((p3.col + 1 >= WIDTH || image[p3.col + 1 + p3.row * WIDTH] == 0) && isExamined(checked)) {
			image[p3.row * WIDTH + p3.col] = -NBD.seq_num;
		}
		else if (p3.col + 1 < WIDTH && image[p3.row * WIDTH + p3.col] == 1) {
			image[p3.row * WIDTH + p3.col] = NBD.seq_num;
		}



		if (p3.col < shape.start.col)
			shape.start.col = p3.col;
		if (p3.row < shape.start.row)
			shape.start.row = p3.row;

		if (p3.row > shape.end.row)
			shape.end.row = p3.row;
		if (p3.col > shape.end.col)
			shape.end.col = p3.col;


		if (p4.samePoint(start) && p3.samePoint(p1)) {

			return;
		}

		p2 = p3;
		p3 = p4;
	}
}

template<int WIDTH, int HEIGHT >
void findContours_HLS(int8_t* image, Shape shapes[1024]) {
	uint8_t h_count = 1;
	Border NBD, LNBD;
	Node h[1024];

	LNBD.border_type = HOLE_BORDER;
	NBD.border_type = HOLE_BORDER;
	NBD.seq_num = 1;

	Node temp_node(-1, -1, -1);
	temp_node.border = NBD;
	h[0] = temp_node;

	Point p2;

	int shapeCount = 0;

	bool border_start_found;

 for (int y = 0; y < HEIGHT; y++) {
	LNBD.seq_num = 1;
	LNBD.border_type = HOLE_BORDER;
	for (int x = 0; x < WIDTH; x++) {
		if (shapeCount < 1024 && h_count < 1024) {
			border_start_found = false;
			auto currP = image[x + 1 + y * WIDTH];
			auto curr = image[x + 0 + y * WIDTH];
			auto currM = image[x - 1 + y * WIDTH];

			if ((curr == 1 && x - 1 < 0) || (curr == 1 && currM == 0)) {
				NBD.border_type = OUTER_BORDER;
				NBD.seq_num += 1;
				p2.setPoint(y, x - 1);
				border_start_found = true;
			}
			else if (x + 1 < WIDTH && (curr >= 1 && currP == 0)) {
				NBD.border_type = HOLE_BORDER;
				NBD.seq_num += 1;
				if (curr > 1) {
					LNBD.seq_num = image[x + y * WIDTH];
					LNBD.border_type =
						h[LNBD.seq_num - 1].border.border_type;
				}
				p2.setPoint(y, x + 1);
				border_start_found = true;
			}

			if (border_start_found) {

				temp_node.reset();
				if (NBD.border_type == LNBD.border_type) {
					temp_node.parent = h[LNBD.seq_num - 1].parent;
					temp_node.next_sibling =
						h[LNBD.seq_num - 1].first_child;
					h[temp_node.parent - 1].first_child = NBD.seq_num;

					temp_node.border = NBD;
					//hierarchy.push_back(temp_node);
					h[h_count] = temp_node;
					h_count++;
				}
				else {
					if (h[LNBD.seq_num - 1].first_child != -1) {
						temp_node.next_sibling =
							h[LNBD.seq_num - 1].first_child;
					}
					temp_node.parent = LNBD.seq_num;
					h[LNBD.seq_num - 1].first_child = NBD.seq_num;
					temp_node.border = NBD;
					h[h_count] = temp_node;
					h_count++;
				}

				Shape temp;
				temp.start = p2;
				followBorder<WIDTH, HEIGHT>(image, p2, y, x, NBD, temp);
				shapes[shapeCount] = temp;
				shapeCount++;

			}

			if (image[x + y * WIDTH] != 0) {
				if (image[x + y * WIDTH] < 0)
					LNBD.seq_num = -image[x + y * WIDTH];
				else
					LNBD.seq_num = image[x + y * WIDTH];

				LNBD.border_type = h[LNBD.seq_num - 1].border.border_type;
			}
		}
	}
}
}

/*
 * Save AXI Stream to AXI Master interface
 */
void image_Preprocessing(AXI_STREAM &stream1,int8_t *picture) {
#pragma HLS INTERFACE axis port=outputStream
#pragma HLS INTERFACE m_axi port=picture offset=slave bundle=MASTER_BUS
#pragma HLS INTERFACE ap_ctrl_none port=return

#pragma HLS dataflow

	static uint8_t fifo1[MAX_WIDTH * MAX_HEIGHT * 3];
	static uint8_t fifo2[MAX_WIDTH * MAX_HEIGHT * 3];
	static uint8_t fifo3[MAX_WIDTH * MAX_HEIGHT * 3];
	static uint8_t fifo4[MAX_WIDTH * MAX_HEIGHT * 1];
	static uint8_t fifo5[MAX_WIDTH * MAX_HEIGHT * 1];
	static uint8_t fifo6[MAX_WIDTH * MAX_HEIGHT * 1];
	static uint8_t fifo7[MAX_WIDTH * MAX_HEIGHT * 1];

	uint8_t threshold[6] = {21,126,109,255,57,255};
	RGB_IMAGE img1(MAX_HEIGHT, MAX_WIDTH);

	hls::AXIvideo2Mat(stream1, img1);
	MatToArray<MAX_WIDTH,MAX_HEIGHT>(img1,fifo1);
	Gauss5<MAX_WIDTH,MAX_HEIGHT>(fifo1,fifo2);
	convertColor<MAX_WIDTH,MAX_HEIGHT>(fifo2,fifo3);
	inRange<MAX_WIDTH,MAX_HEIGHT>(fifo3,fifo4,threshold);
	dilate<MAX_WIDTH,MAX_HEIGHT>(fifo4,fifo5);
	erode<MAX_WIDTH,MAX_HEIGHT>(fifo5,fifo6);
	zeroBorder<MAX_WIDTH,MAX_HEIGHT>(fifo6,fifo7);
	Mat2Master<MAX_WIDTH,MAX_HEIGHT>(fifo7,picture);
}


