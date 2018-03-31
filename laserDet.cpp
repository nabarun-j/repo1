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

struct str{
	bool operator() ( Point2f a, Point2f b ){
		if ( a.x != b.x ) 
			return a.x < b.x;
	}
} comp;

Point2f* mySwap(Point2f* points, int i, int j){
	Point2f temp;
	temp=points[i];
	points[i]=points[j];
	points[j]=temp;
	return points;
}

void mySort(Point2f * points){
	//based on x values
	sort(points, points+4, comp);
	points=mySwap(points, 1,3); 
	//now points0 and points3 are the leftmost points (assumptions involved here)
	if(points[0].y>points[3].y){points=mySwap(points,0,3);}
	if(points[1].y>points[2].y){points=mySwap(points,1,2);}
}



Mat myAdaptiveThreshold(Mat gray_img){
	// adaptive thresholding where we are using a sliding(in steps of 100x100) window 
	// in which we find the average value (of grayscale image) in each window 
	// and pixels below the average by a limit (den * 30) will be thresholded out
	int X,Y;
	float greatestVal=0;
	Scalar averageVal;
	for(X=0; X<gray_img.cols; X+=100){
		for(Y=0; Y<gray_img.rows; Y+=100){
			int minX, minY;
			//check for reaching end of the image
			minX=(X+100<gray_img.cols)?(X+100):gray_img.cols;
			minY=(Y+100<gray_img.rows)?(Y+100):gray_img.rows;
			//forming the mask on the window
			cv::Mat mask = cv::Mat::zeros(gray_img.rows,gray_img.cols, CV_8UC1); // all 0
			//only region inside this rect is now set to 255 (ROI)
			mask(Rect(X,Y,minX-X,minY-Y)) = 255;
			//finding the average value in the ROI with respect to the grayscale image
			averageVal= mean(gray_img ,mask=mask);
			//checking greatest average intesity (of any window) in image
			if(averageVal.val[0]>greatestVal){greatestVal=averageVal.val[0];}
			int i,j;
			for(i=X;i<minX;i++){
				for(j=Y;j<minY;j++){
					// If the ROI is bright the average value would be high
					// and the difference in the intensities of the laser 
					// and the background would be low. So we need a factor which is
					// inversely proportional to the avg. intensity but is >1

					// the multiplicative factor was found by experimentation
					// TODO: Needs better approach.
					float factor=(1/averageVal.val[0])*200*30;
					if(gray_img.at<uchar>(Point(i,j))>averageVal.val[0]+factor){
						gray_img.at<uchar>(Point(i,j))=255;
					}
					else{
						gray_img.at<uchar>(Point(i,j))=0;
					}
				}
			}
		}
	}
	printf("highest average val: %f\n",greatestVal);
return gray_img;
}



vector<Rect > detectContours(Mat gray_img){
	// We can now find the contours from the thresholded image and filter them based on some criteria
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(gray_img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	printf("number of contours: %d \n", contours.size());

	// insert all contours of relevant size into the collection of small-contours (filter by size)
	vector<vector<Point> > smallContours;
	for (int i =0; i<contours.size(); i++){
		double conArea= contourArea(contours[i], false);
		// Limits on size found by experimentation and need a review
		if(conArea>30 && conArea<1000){
			smallContours.push_back(contours[i]);
		}
	}
	printf("number of rects/smallContours: %d\n", smallContours.size());

	// Forming a bounding rectangle on the contour
	vector<Rect> rectsProposed;
	for (int i=0; i<smallContours.size();i++){
		rectsProposed.push_back(boundingRect(smallContours[i]));
	}

	//the ones selected as the real contours
	vector <Rect > rects;
	// adding the new constraint for windowSize, where we check the number of white pixels 
	// in a window around each contour and filtering out the rest.
	// Done to differentiate solitary bright spots from continuous white regions
	int x,y, countWindowPixels, countContourPixels, countTotalPixels=0;
	Mat contourImg;
	for(int i=0; i<rectsProposed.size(); i++){
		countTotalPixels=0;
		contourImg=gray_img(rectsProposed[i]);
		// No of white pixels inside the contour
		countContourPixels=countNonZero(contourImg);
		for(x=rectsProposed[i].x-20; x<rectsProposed[i].x+rectsProposed[i].width+20; x++){
			for(y=rectsProposed[i].y-20; y<rectsProposed[i].y+rectsProposed[i].height+20; y++){
				if(gray_img.at<uchar>(Point(x,y))!=0){
					countTotalPixels++;
				}
			}
		}
		// No of white pixels outside the contour but within the window
		countWindowPixels=countTotalPixels-countContourPixels;
		// TODO: Value found by experimentation need to review
		if(countWindowPixels<=13){
			rects.push_back(rectsProposed[i]);
		}
	}
	printf("Number of filtered contours: %d\n", rects.size()); 
return rects;
}


vector<Rect > checkRedHue(Mat red_img_hsv, vector <Rect > rects){
	vector<Rect > detectedRect;
	//checking for red pixels inside the rectangles binding the found contours of interest
	int flag1=0,count1;
	int count_of_spots=0;
	int x,y;
	for(int i=0; i<rects.size(); i++){
		int count1=0;
		int flag=0;
		for(x=rects[i].x; x<rects[i].x+rects[i].width; x++){
			for(y=rects[i].y; y<rects[i].y+rects[i].height; y++){
				if(red_img_hsv.at<uchar>(Point(x,y))!=0){
					count1++;
					flag1=1;
				}	
				//int pixels=countNonZero(rects[i]);
				//printf("Number of non zero pixels in rect%d: %d\n",i, pixels);
				if(count1>20){
					detectedRect.push_back(rects[i]);
					count_of_spots++;
					flag=1;	
					break;
				}
			}
			if(flag==1){
				printf("region of interest found\n");
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
	printf("total no. of spots found: %d\n", count_of_spots);
return detectedRect;
}

void myCrop(vector<Rect > detectedRect, Mat img, int fileNum){
		// Lambda Matrix
		Mat lambda( 2, 4, CV_32FC1 );

		// Output Quadilateral or World plane coordinates
		Point2f outputQuad[4];

		// output mat
		Mat output;

		// Set the lambda matrix the same type and size as input
		lambda = Mat::zeros( img.rows, img.cols, img.type());

		//points
		Point2f points[4];

		//unsorted order of points    
		points[0]=Point2f(detectedRect[0].x+(detectedRect[0].width)/2,detectedRect[0].y+(detectedRect[0].height)/2);
		points[1]=Point2f(detectedRect[1].x+(detectedRect[1].width)/2,detectedRect[1].y+(detectedRect[1].height)/2);
		points[2]=Point2f(detectedRect[2].x+(detectedRect[2].width)/2,detectedRect[2].y+(detectedRect[2].height)/2);
		points[3]=Point2f(detectedRect[3].x+(detectedRect[3].width)/2,detectedRect[3].y+(detectedRect[3].height)/2);

		//sort in clockwise order starting from the top left at point[0]
		mySort(points);

		// The 4 points where the mapping is to be done , from top-left in clockwise order
		outputQuad[0] = Point2f( 0,0 );
		outputQuad[1] = Point2f( img.cols,0);
		outputQuad[2] = Point2f( img.cols,img.rows);
		outputQuad[3] = Point2f( 0,img.rows);

		// Get the Perspective Transform Matrix i.e. lambda 
		lambda = getPerspectiveTransform( points, outputQuad );

		// forming a custom region of interest on which to apply the warp on 
		//masking involved
		Mat mask;
		mask.create(img.rows, img.cols, CV_8UC1);
		// create black image with the same size as the original
		int i,j;
		for(i=0;i<mask.cols;i++){
			for(j=0; j<mask.rows; j++){
				mask.at<uchar>(Point(i,j))=0;
			}
		} 

		//forming a region of interest for example
		vector<Point > ROI_Vertices;
		ROI_Vertices.push_back(points[0]);
		ROI_Vertices.push_back(points[1]);
		ROI_Vertices.push_back(points[2]);
		ROI_Vertices.push_back(points[3]);

		//form the shape of the region of interest
		vector<Point > ROI_Poly;
		approxPolyDP(ROI_Vertices, ROI_Poly, 1.0, true);

		//filling white in polygon to create mask
		fillConvexPoly(mask, &ROI_Poly[0], ROI_Poly.size(), 255, 8, 0); 

		// Create new image for result storage
		Mat imageDest;
		imageDest.create(img.rows, img.cols, CV_8UC3);

		// Cut out ROI and store it in imageDest
		img.copyTo(imageDest, mask);  

		// Apply the Perspective Transform just found to the src image
		warpPerspective(img,output,lambda,output.size() );
		char bufferName[100];
		int t=sprintf(bufferName, "/home/nabarun/laserDet/build/img_%d/ROI.png",fileNum);
		imwrite(bufferName, imageDest);
		t=sprintf(bufferName, "/home/nabarun/laserDet/build/img_%d/outputMasked.png",fileNum);
		imwrite(bufferName, output);
}


int main(int argc, char** argv){
	int MAX_KERNEL_LENGTH = 31;
	Mat img;
	if (argc>=2)
		img=imread(argv[1],1);
	else{
		printf("error\n");
		return -1;
	}
	Mat gray_img;
	int threshval = atoi(argv[2]);
	//for testing 
	int fileNum=atoi(argv[3]);
	char bufferName[100];
	int t;
	t=sprintf(bufferName, "/home/nabarun/laserDet/build/img_%d/original.jpg",fileNum);
	imwrite(bufferName,img);

	//conversion to grayscale image
	cvtColor(img, gray_img, CV_BGR2GRAY);
	cout << "gray_img size " << gray_img.cols << " x " << gray_img.rows << std::endl;
	t=sprintf(bufferName, "/home/nabarun/laserDet/build/img_%d/grayscale.png",fileNum);
	imwrite(bufferName, gray_img);

	//threshold the grayscale image
	gray_img=myAdaptiveThreshold(gray_img);
	t=sprintf(bufferName, "/home/nabarun/laserDet/build/img_%d/manipulatedGray.png",fileNum);
	// Dump the thresholded image
	imwrite(bufferName, gray_img);
 
	//detect the contours in the thresholded image, form bounding Rect around each and collect them
	Mat img1;
	vector <Rect > rects=detectContours(gray_img); 
	// Marking the filtered contours on the image and dumping
	for(int i=0;i<rects.size(); i++){
		img1=img.clone();
		cv::rectangle(img1, Point(rects[i].x, rects[i].y), Point(rects[i].x + rects[i].width, rects[i].y + rects[i].height), Scalar(0, 255, 0), 5, LINE_8, 0);
		char buffer[100];
		int l=sprintf(buffer, "/home/nabarun/laserDet/build/img_%d/spots%d.png",fileNum,i);
		imwrite(buffer, img1);	
	}

	//converting to hsv
	Mat img_hsv;
	cvtColor(img, img_hsv, COLOR_BGR2HSV);
	t=sprintf(bufferName, "/home/nabarun/laserDet/build/img_%d/img_hsv.png",fileNum);
	imwrite(bufferName, img_hsv);

	//selecting red hue
	Mat lower_red_hsv; 
	Mat upper_red_hsv;
	Mat red_img_hsv;
	// Lower part of red hue
	inRange(img_hsv,Scalar(8, 80, 0), Scalar(10, 255, 255), lower_red_hsv);
	// Upper part of red hue
	inRange(img_hsv,Scalar(160, 80, 0), Scalar(180, 255, 255), upper_red_hsv);
	// Club both upper and lower limit
	add(upper_red_hsv,lower_red_hsv, red_img_hsv, noArray(),-1);
	t=sprintf(bufferName, "/home/nabarun/laserDet/build/img_%d/red_img_hsv(0-3,160-180 & S 80-255).png",fileNum);
	imwrite(bufferName, red_img_hsv);

	/* TESTING BEGIN */
	// For testing
	// finding the pixels of light in the upper limit which may be a source of light, 
	// subtracting it from the red range will help nullify them
	Mat upper_img_hsv;
	Mat higher_upper_img_hsv;
	inRange(img_hsv, Scalar(160,0,0),Scalar(180, 80,255), higher_upper_img_hsv);
	inRange(img_hsv,Scalar(0, 0, 0), Scalar(20, 80, 255), upper_img_hsv);

	Mat total_img_hsv=upper_img_hsv;
	t=sprintf(bufferName, "/home/nabarun/laserDet/build/img_%d/lower_img_hsv(0-20, S- 0-80).png",fileNum);
	imwrite(bufferName, upper_img_hsv);

	t=sprintf(bufferName, "/home/nabarun/laserDet/build/img_%d/upper_img_hsv(160-180, S- 0-80).png",fileNum);
	imwrite(bufferName, higher_upper_img_hsv);


	// finding contours in the red-range hsv image, to compare with ones from the thresholded grayscale image
	vector<vector<Point> > contoursHsv;
	vector<Vec4i> hierarchyHsv;
	findContours(red_img_hsv, contoursHsv, hierarchyHsv, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	printf("number of contours in red_img_hsv: %d \n", contoursHsv.size());
	vector<vector<Point> > smallContoursHsv;
	//insert all contours of relevant size into the collection of small-contours
	for (int i =0; i<contoursHsv.size(); i++){
		double conArea= contourArea(contoursHsv[i], false);
		if(conArea>40 && conArea<300){
			smallContoursHsv.push_back(contoursHsv[i]);
			//printf("one more\n");
		}
	}
	printf("number of rects/smallContoursHsv: %d\n", smallContoursHsv.size());
	/* TESTING END */

	std::vector<Rect > detectedRect=checkRedHue(red_img_hsv, rects);
	//here
	for(int i;i<detectedRect.size();i++){
			img1=img.clone();
			cv::rectangle(img1, Point(detectedRect[i].x, detectedRect[i].y),Point(detectedRect[i].x + detectedRect[i].width, 
						detectedRect[i].y + detectedRect[i].height), 
						Scalar(0, 255, 0), 5, LINE_8, 0);
			char buffer[100];
			t=sprintf(buffer, "/home/nabarun/laserDet/build/img_%d/detected_spots%d.png",fileNum,i);
			imwrite(buffer, img1);		
	}

	// TODO: Put in a separate function
	// cropping
	if(detectedRect.size()==4){
		printf("cropping\n");
		myCrop(detectedRect,img,fileNum);

	}
	else{
		printf("Number of spots detected is not 4. Cropping not possible.\n");
	}
}
