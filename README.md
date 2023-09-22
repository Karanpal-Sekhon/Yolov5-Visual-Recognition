# Yolov5-Visual-Recognition

In this repo, you will find the files best.pt, classifier.py, log.txt, and main.py.
<br>
<a href = "best.py">best.pt</a>
The best trained model from previous training. This model predicts with an accuracy greater than 95% on my given dataset. Unfortunately the dataset is too large to be posted on to github, in the future it is likely I will cut the data in half and retriain and repost.
<br>

## classifier.py
This python file loads the pre-trained model and the images, then proceeds to use best.pt and test the model on the images using a for loop. The for loop then ends up returning the predicted classes and boxes for the digit classification. Which will be later used to for 
calculating the classification accuracy and IOU score in main.py
<br>
## main.py
This is the file that calculates all of the classification and IOU scores, also keeps track of all of the time taken to run, as well as creating log.txt, which shows the accuracies and iou scores of the model.
