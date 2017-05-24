## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[hog_car]: ./output_images/hog_car.png
[hog_noncar]: ./output_images/hog_noncar.png
[image_pipeline]: ./output_images/image_pipeline.png
[video]: ./output_images/project_video_result.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in  in lines 17 through 28 of the file called `lesson_functions.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  I explored different color spaces.  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is some example of one of each of the `vehicle` and `non-vehicle` classes using the `RGB`, `HSV`, `YUV`, `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`::

![alt text][hog_car]

![alt text][hog_noncar]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and use the parameters to training a classifier. The parameters with the highest accuracy on test dataset are set as final choice of HOG parameters.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for training is in `training.py`. The `HSV` color space is used to generate HOG feature. The parameters used to generate HOG feature are `param_spatial_size = (32, 32)`, `param_hist_bins = 32`, 
`param_orient = 8`, `param_pix_per_cell = 8`, `param_cell_per_block = 2`, `param_hog_channel = 'ALL'`. Besides HOG feature, spatial feature and histogram features are also used just as different combinations suggest. The length of the feature vector is 7892. The feature vector is scaled before training. Then a linear SVM classifier is used to train a classifier.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Firstly, combinations of different positions and window size are defined in line 12 through 16 in `image_pipeline.py`. The position and window size are set by the possible vehicle position and vehicle size in the image. Then, I iterate to generate all the sliding windows with different settings in line 146 through line 196 in `lesson_functions.py`.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The pipeline to find car is in line 232 through 273 in `lesson_functions.py`. The detailed pipeline is as follows:
1. All the sliding windows are generated;
2. Feature vecotr of each patch of the image genrated by sliding window is calculated and scaled;
3. Each patch is fed into the trained classifier. If the classifier suggests that the patch is a car, the window is saved. 
4. All the patches that may be a car is saved and overlapped to generate a heatmap.
5. The pixels has more than one patch are identified as vehicle to avoid false positive detection.

Here is one example of the pipeline.
![alt text][image_pipeline]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
The video file is in output_images.


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The filter used to avoid false positive detection is in line 42 through line 60 in `lesson_functions.py`. The algorithm used for video pipeline is the same as the fiter used in image pipeline. I recorded all the detected windows that are identified as vehicles. A heatmap is generated based on all these windows. The pixels that has more than 15 windows identified as vehicles are set as final detection of vehicle. The parameters 20 and 15 are carefully tuned by testing the pipeline in project video.

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline may fail where the color or the texture (HOG feature) is the same as a vehicle. For example, the shadow of trees maybe identified as vehicles.  To avoid false positive detection, a filter based on rolling window is implemented. Deep learning based vehicle detection may also solve this problem if large training dataset is provided.