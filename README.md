# Exercise Tracker: AI-Powered Gym Reps Counter

## Overview
Exercise Tracker is a Flask-based web application that uses computer vision to track body movements and count exercise repetitions in real-time. It leverages OpenCV and MediaPipe Pose for accurate pose estimation, providing a seamless experience for fitness enthusiasts, trainers, and developers looking to explore AI-powered workout tracking.

<p align="center">
  <img src="https://github.com/mohyd2233/Exercise-Tracker-Application/blob/main/github_assets/gym-2.jpg" />
</p>

<p align="center">
  <img src="https://github.com/mohyd2233/Exercise-Tracker-Application/blob/main/github_assets/gym-app-prototype.jpg" />
</p>

## Features
- Real-time Pose Detection: Tracks body landmarks and joint movements using MediaPipe Pose.
- Reps Counter: Automatically counts exercise repetitions (e.g., bicep curls) based on body angles.
Stage Detection: Provides feedback on exercise stages (e.g., "up" or "down").
- Web Interface: Stream video feed directly to a browser using Flask and display live metrics.
- Customizable UI: Dark-themed, gym-styled interface with dynamic updates for reps and stage.
- Modular Design: Easy to extend for different exercises and user preferences.

## Technologies Used
- Flask: For backend and web app deployment.
- OpenCV: For capturing and processing video frames.
- MediaPipe: For real-time pose estimation.
- HTML/CSS: For an attractive, gym-themed user interface.

## How It Works
1. The user opens the web app and allows camera access.
2. The app captures video feed, processes it with MediaPipe Pose, and calculates joint angles.
3. Based on the detected angles, the app tracks the number of completed repetitions and current exercise stage.
4. All data is rendered in real-time on the web interface, including live video feed, reps count, and exercise stage.

## Installation and Usage
1. Clone the repository:
git clone https://github.com/mohyd2233/Exercise-Tracker-Application
2. Install dependencies:
pip install -r requirements.txt
3. Run the Flask app:
python app.py
4. Open your browser and navigate to http://127.0.0.1:5000/.

## Future Enhancements
- Support for multiple exercises (e.g., squats, push-ups).
- Exercise form correction and feedback.
- Integration with user accounts and workout history tracking.
- Deployment on cloud platforms for wider accessibility.