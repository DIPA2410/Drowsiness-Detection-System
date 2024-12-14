# Real Time Drowsiness Detection System

Drowsiness detection is a safety technology that can prevent accidents that are caused by drivers who fell asleep while driving. The objective of this project is to build a drowsiness detection system that will detect drowsiness through the implementation of computer vision system that automatically detects drowsiness in real-time from a live video stream and then alert the user with an alarm notification.

## Built With

* [OpenCV Library](https://opencv.org/) - Most used computer vision library. Highly efficient. Facilitates real-time image processing.
* [imutils library](https://github.com/jrosebr1/imutils) -  A collection of helper functions and utilities to make working with OpenCV easier.
* [Dlib library](http://dlib.net/) - Implementations of state-of-the-art CV and ML algorithms (including face recognition).
* [scikit-learn library](https://scikit-learn.org/stable/) - Machine learning in Python. Simple. Efficient. Beautiful, easy to use API.
* [Numpy](http://www.numpy.org/) - NumPy is the fundamental package for scientific computing with Python. 

## Alogorithm

1. Capture the image of the driver from the camera.
2. Send the captured image to haarcascade file for face detection.
3. If the face is detected then crop the image consisting of the face only. If the driver is distracted then a face might not be detected, so play the buzzer.
4. Send the face image to haarcascade file for eye detection.
5. If the eyes are detected then crop only the eyes and extract the left and right eye from that image. If both eyes are not found, then the driver is looking sideways, so sound the buzzer.
6. The cropped eye images are sent to the hough transformations for detecting pupils, which will determine whether they are open or closed.
7. If they are found to be closed for five continuous frames, then the driver should be alerted by playing the buzzer.

For a more detailed explanation of this project check [Drowsiness-Detection_Report_2024.pdf](https://github.com/DIPA2410/Drowsiness-Detection-System/blob/main/Real-Time-Drowsiness-Detection-System-main/Drowsiness-Detection_Report_2024.pdf).***************

## Testing and Results in Real-World Scenario:

Test case 1: Open Eyes 

<p align="center">
<img width="600" height="350" src="https://github.com/user-attachments/assets/ee2bb23f-2614-4590-9bd3-4fc0dfe0e261">
</p>
                                  
Test case 2: Closed Eyes

<p align="center">
<img width="600" height="350" src="https://github.com/user-attachments/assets/4b332d85-c02a-4674-b52e-0a13bc75024b">
</p>
                                                               
Test case 3: Yawning     
                                   
<p align="center">
<img width="600" height="350" src="https://github.com/user-attachments/assets/7cf2457f-00bc-47d7-863b-60f246c12c45">
</p>

The system was extensively tested even in real world scenarios, this was achieved by placing the camera on the visor of the car, focusing on the automobile driver. It was found that the system gave positive output unless there was any direct light falling on the camera.       

## Future Scope

Smart phone application: It can be implemented as a smart phone application, which can be installed on smart phones. And the automobile driver can start the application after placing it at a position where the camera is focused on the driver.

## References

IEEE standard Journal Paper,

[1] A. Y. e. al., ""Driver Drowsiness Detection Techniques: A Review," Proceedings of the IEEE International Conference on Smart Technologies for Smart Nation (SmartTechCon)," (https://ieeexplore.ieee.org/document/8358464), p. pp. 1101–1106, 2017. 

[2] K. Roy, S. Bhattacharjee, and A. Ghosh, "Real-Time Driver Drowsiness Detection Using Facial Features and Machine Learning," IEEE Access, vol. 9, pp. 88739–88753, 2021. (https://ieeexplore.ieee.org/document/9456325). 

[3] Facial Features Monitoring for Real Time Drowsiness Detection by Manu B.N, 2016 12th International Conference on Innovations in Information Technology (IIT) [Pg. 78-81] (https://ieeexplore.ieee.org/document/7880030). 

[4] Real Time Drowsiness Detection using Eye Blink Monitoring by Amna Rahman Department of Software Engineering Fatima Jinnah Women University 2015 National Software Engineering Conference (NSEC 2015) (https://ieeexplore.ieee.org/document/7396336). 

[5] R. K Yadav, Richa Gupta, Srikar Sundram, Sakshi Awasthi, Harsh Anand Real Time Driver Drowsiness Detection System using Facial Expression (2023) . (https://ijarsct.co.in/Paper10895.pdf)

Websites referred:

1.	https://www.codeproject.com/Articles/26897/TrackEye-Real-Time-Tracking-Of-Human-Eyes-
2.	https://realpython.com/face-recognition-with-python/
3.	https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv
4.	https://www.pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv
5.	https://www.codeproject.com/Articles/26897/TrackEye-Real-Time-Tracking-Of-HumanEyesUsing-a
6.	https://docs.opencv.org/3.4/d7/d8b/tutorial_py_face_detection.html
7.	https://www.learnopencv.com/training-better-haar-lbp-cascade-eye-detector-opencv/

## Authors

<p> Smita Jha </p> 
<p> Dipa Roy </p>
<p> Rajdip Barman </p>
<p> Joy Hazra </p>
<p> Bhagyashree Singh </p>
