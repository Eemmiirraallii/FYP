# README.md

## Drowsiness Detection

This is a Python script that uses machine learning and computer vision to detect drowsiness and yawning in a person. If drowsiness or yawning is detected, it plays an alert to the user. 

The drowsiness detection is based on the state of the user's eyes, while the yawning detection is based on the size of the user's mouth. This program uses OpenCV and dlib libraries for the computer vision part and a Support Vector Machine (SVM) model for the machine learning part. 

## Getting Started

These instructions will guide you on how to run the program on your local machine for development and testing purposes.

### Prerequisites

Make sure you have Python 3.x installed on your system. If not, you can download it [here](https://www.python.org/downloads/).

### Clone the Repository

Firstly, clone the repository using the following command:

```bash
git clone https://github.com/{username}/FYP-main.git
cd FYP-main
```

Replace `{username}` with your GitHub username.

### Install Dependencies

This project requires several Python libraries. Install them with the following command:

```bash
pip install numpy opencv-python dlib imutils matplotlib python-vlc joblib
```

### Download Additional Files

You need to download the following files and place them in the project's directory:

- `shape_predictor_68_face_landmarks.dat`: You can download it from [here](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2). This file is used to detect the facial landmarks in a face.

- `model3.pkl`: This is the pre-trained SVM model that is used for drowsiness detection. You should have received this file along with the project files.

- `focus.mp4`: This is the audio that is played when drowsiness is detected. You should have received this file along with the project files.

- `take_a_break.mp4`: This is the audio that is played when yawning is detected multiple times. You should have received this file along with the project files.

### Running the Program

After installing the prerequisites and downloading the required files, you can run the program with the following command:

```bash
python final-integration.py
```

The script will start the webcam or regular camera and begin to detect for signs of drowsiness and yawning. 

To stop the script, press the `ESC` key.

## License

This project is licensed under the MIT License. 
