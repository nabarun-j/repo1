#include <iostream>
#include <sstream>
#include <time.h>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#ifndef _CRT_SECURE_NO_WARNINGS
# define _CRT_SECURE_NO_WARNINGS
#endif

using namespace cv;
using namespace std;


int main(int argc, char** argv){
int MAX_KERNEL_LENGTH = 31;
Mat img;
if (argc>=2)
	img=imread(argv[1],1);
else{
	printf("error\n");
	return -1;
}
imwrite("original.png",img);
Mat gray_img;
int threshval = atoi(argv[2]);
//conversion to grayscale image
cvtColor(img, gray_img, CV_BGR2GRAY);
cout << "gray_img size " << gray_img.cols << " x " << gray_img.rows << std::endl;
imwrite("grayscale.png", gray_img);
Mat gray_thresh;
int threshold_type;
  /* threshold_type=
  	 0: Binary
     1: Binary Inverted
     2: Threshold Truncated
     3: Threshold to Zero
     4: Threshold to Zero Inverted
   */
threshold_type=0;
int flag=0;
int count=0; 
// insert appropriate threshold value here (should be 30% of maximum intensity value in the image or something similar)
cv::threshold(gray_img, gray_thresh, threshval, 255, threshold_type );
cout << "gray_thresh size " << gray_thresh.cols << 
" x " << gray_thresh.rows << std::endl;
for(int i=0; i<gray_thresh.rows; i++){
	for(int j=0; j<gray_thresh.cols; j++){ 
		if(gray_thresh.at<uchar>(Point(j,i))!=0){
			count++;
		}
		if(count >100){
			printf("number of pixels above threshold is greater than 100. Adjustment of gain required\n");
			flag=1;
			break;
		}
	}
	if(flag==1){
		//do something to reduce gain
		break;
	}
}

// Now, we know that the number of pixels above the threshold is less than 100, and so we can now find the required contours which fit the description for red hues
vector<vector<Point> > contours;
vector<Vec4i> hierarchy;
findContours(gray_thresh, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

vector<vector<Point> > smallContours;

//insert all contours of relevant size into the collection of small-contours
count=0;
for (int i =0; i<contours.size(); i++){
	double conArea= contourArea(contours[i], false);
	if(conArea>25 && conArea<1000){
		smallContours.push_back(contours[i]);
		count++;
		//printf("one more\n");
	}
}

//collecting the contours inside rectangles
vector<Rect> rects;
for (int i=0; i<smallContours.size();i++){
	rects.push_back(boundingRect(smallContours[i]));
}

//implement blur on image
/*
for( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
         { medianBlur ( img, img, i );
         	printf("blur implemented\n");
         }
*/

// fragmenting image into different channels (BGR)
std::vector<cv::Mat> BGR(3);
cv::split(img, BGR);

imwrite("blue.png",BGR[0]); //blue channel
imwrite("green.png",BGR[1]); //green channel
imwrite("red.png",BGR[2]); //red channel

//removing areas in the image which are common to all channels (to get regions which are specifically red)
add(BGR[0], BGR[1], BGR[0], noArray(), -1 );
imwrite("blue_green.png",BGR[0]);
subtract(BGR[2], BGR[0], BGR[2], noArray(), -1 );

namedWindow("window",WINDOW_AUTOSIZE);
imwrite("red_sub_img.jpg", BGR[2]);
imshow("window", BGR[2]);
waitKey(0);

//checking for pixels inside the rectangles binding the found contours of interest
int x,y;
int flag1=0,count1;
flag=0;
int count_of_spots=0;
for(int i=0; i<rects.size(); i++){
	count1=0;
	flag=0;
	for(x=rects[i].x; x<rects[i].x+rects[i].width; x++){
		for(y=rects[i].y; y<rects[i].y+rects[i].height; y++){
			if(BGR[2].at<uchar>(Point(x,y))!=0){
				count1++;
				flag1=1;
			}	
			if(count1>50){
				count_of_spots++;
				flag=1;
				break;
			}
		}
		if(flag==1){
			printf("region of interest found\n");
			count_of_spots++;
			break;
		}
	}
}

if(flag1==1){
	printf("atleast one rectangle binds a region of interest\n");
}
else{
	printf("no rectangle binds a region of interest\n");
}

printf("total no. of spots found: %d\n", count_of_spots );

printf("%d contours found in range\n", count);
imwrite("gray_thresh.jpg", gray_thresh);
imshow("window", gray_thresh);
waitKey(0);
}
