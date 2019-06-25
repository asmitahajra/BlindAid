# BlindAid
### Overview
BlindAid is an app that performs real time object detection by continuously detecting the objects (bounding boxes and classes) in the frames seen by your device's back camera, using a quantized [MobileNet SSD](https://github.com/tensorflow/models/tree/master/research/object_detection) model trained on the [COCO dataset](http://cocodataset.org/).

Additional features:
+ Speech promopted service. Trigger object detection by saying 'guide me/ start guiding me'
+ The app then performs position detection of different objects in a frame (left, centre and right) with speech
+ 'Stop/ Stop guiding me' to stop the detection
+ Speech prompted text detection. Trigger OCR with the words 'what's written in front of me/read'
+ Performs barcode detection

The model files are downloaded via Gradle scripts when you build and run. You don't need to perform any steps to download TFLite models into the project explicitly.

Application can run either on device or emulator.

![Banana on the left, Orange on the right](https://user-images.githubusercontent.com/30727326/60101900-3905da80-977a-11e9-90f3-05ae911a1065.jpeg)
"Banana on the left, Orange on the right"

![barcode](https://user-images.githubusercontent.com/30727326/60102317-0f997e80-977b-11e9-999f-b74a072ddfd5.jpeg)

Barcode detection

![dental](https://user-images.githubusercontent.com/30727326/60104180-8edc8180-977e-11e9-98fa-cafbbea975c8.jpeg)

Text detection example 1

![salt](https://user-images.githubusercontent.com/30727326/60102562-9189a780-977b-11e9-82b3-b805a127313c.jpeg)

Text detection example 2

The model used currently recognizes 80 different classes of objects, and will be trained on extended datasets to further increase the efficiency of the app.

<!-- TODO(b/124116863): Add app screenshot. -->

## Build using Android Studio

### Prerequisites

* If you don't have it already, install **[Android Studio](https://developer.android.com/studio/index.html)**, by following the instructions on the website.
* You need an Android device and Android development environment with minimum API 21.
* Android Studio 3.2 or later.

### Building
* Clone the repository 
* Open Android Studio, and from the Welcome screen, select Open an existing Android Studio project
* From the Open File or Project window that appears, navigate to and select the directory from wherever you cloned the project
* If it asks you to do a Gradle Sync, click OK.
* You may also need to install various platforms and tools, if you get errors like "Failed to find target with hash string 'android-21'" and similar.
* Run the app.
* Also, you need to have an Android device plugged in with developer options enabled at this point. See **[here](https://developer.android.com/studio/run/device)** for more details on setting up developer devices.

### Model used
Downloading, extracting and placing it in assets folder has been managed automatically by download.gradle.

If you explicitly want to download the model, you can download it from **[here](http://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip)**. Extract the zip to get the .tflite and label file.

### Additional Note
_Please do not delete the assets folder content_. If you explicitly deleted the files, then please choose *Build*->*Rebuild* from menu to re-download the deleted model files into assets folder.
