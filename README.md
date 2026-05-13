# Attendify AI

An AI-based smart attendance system built with Flask, Python, OpenCV facial recognition, and MySQL through XAMPP.

## Features

- Student enrollment with webcam face capture
- Dataset registration with multiple face samples per student
- OpenCV LBPH face model training after enrollment and dataset updates
- Browser-camera attendance scanning
- One attendance record per student per day
- Attendance reports with date filtering
- MySQL schema for XAMPP/phpMyAdmin

## Requirements

- Python 3.10 or newer
- XAMPP with MySQL running
- A webcam

## Setup

1. Start Apache and MySQL in XAMPP.
2. The app can create the database tables automatically. You can also import `database.sql` in phpMyAdmin, or run:

   ```bash
   mysql -u root < database.sql
   ```

3. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Copy `.env.example` to `.env` and update the values if your XAMPP MySQL user has a password.

6. Run the app:

   ```bash
   python app.py
   ```

7. Open `http://127.0.0.1:5000`.

## Notes

- `opencv-contrib-python` is required because the app uses `cv2.face.LBPHFaceRecognizer_create`.
- For best results with the current OpenCV model, register 10 to 20 dataset samples per student in good lighting, with small angle and expression changes.
- The trained model and face samples are stored in the local `data/` folder.

## Accuracy Recommendation

This project currently uses OpenCV LBPH because it is lightweight, runs locally, and is easy to install for a Flask/XAMPP student project.

For a more production-grade AI model, use ArcFace or FaceNet embeddings through a library such as InsightFace or DeepFace. Those models are usually more accurate than LBPH, especially with many students, different lighting, and different camera quality.
