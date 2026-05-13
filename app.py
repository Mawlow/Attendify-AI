import base64
import os
import shutil
from datetime import date, datetime, time, timedelta
from pathlib import Path

import cv2
import mysql.connector
import numpy as np
from dotenv import load_dotenv
from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from mysql.connector import IntegrityError


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FACES_DIR = DATA_DIR / "faces"
MODEL_PATH = DATA_DIR / "trainer.yml"
FACE_SIZE = (200, 200)
# LBPH returns a Chi-square histogram distance; lower is a better match. Values
# above this are treated as "unknown" (not enrolled). Default 75 was too loose
# and let different people match. Tune with env FACE_RECOGNITION_THRESHOLD: raise
# (e.g. 55–65) if the real student is often rejected; lower for stricter checks.
RECOGNITION_THRESHOLD = float(os.getenv("FACE_RECOGNITION_THRESHOLD", "50"))
DB_NAME = os.getenv("DB_NAME", "attendify_ai").strip()

# Order of face captures during student enrollment (must match the form and UI).
ENROLLMENT_POSE_ORDER = ("front", "left", "right", "down", "up")


def enrollment_pose_label(pose_key: str) -> str:
    return {
        "front": "front (straight at the camera)",
        "left": "left side (turn head so your left cheek faces the camera)",
        "right": "right side (turn head so your right cheek faces the camera)",
        "down": "looking down",
        "up": "looking up",
    }.get(pose_key, pose_key)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-this-secret-key")
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def ensure_storage() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    FACES_DIR.mkdir(exist_ok=True)


ensure_storage()


def get_mysql_config(include_database=True):
    config = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "3306")),
        "user": os.getenv("DB_USER", "root"),
        "password": os.getenv("DB_PASSWORD", ""),
    }

    if include_database:
        config["database"] = DB_NAME

    return config


def validate_database_name():
    if not DB_NAME or not DB_NAME.replace("_", "").isalnum():
        raise RuntimeError("DB_NAME must contain only letters, numbers, and underscores.")


def initialize_database():
    validate_database_name()
    connection = mysql.connector.connect(**get_mysql_config(include_database=False))
    cursor = connection.cursor()
    try:
        cursor.execute(
            f"""
            CREATE DATABASE IF NOT EXISTS `{DB_NAME}`
            CHARACTER SET utf8mb4
            COLLATE utf8mb4_unicode_ci
            """
        )
        cursor.execute(f"USE `{DB_NAME}`")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS students (
                id INT AUTO_INCREMENT PRIMARY KEY,
                student_number VARCHAR(50) NOT NULL UNIQUE,
                full_name VARCHAR(150) NOT NULL,
                course VARCHAR(100),
                year_section VARCHAR(100),
                email VARCHAR(150),
                face_image_path VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance (
                id INT AUTO_INCREMENT PRIMARY KEY,
                student_id INT NOT NULL,
                attendance_date DATE NOT NULL,
                time_in TIME NOT NULL,
                status ENUM('Present', 'Late', 'Absent') NOT NULL DEFAULT 'Present',
                confidence DECIMAL(8, 2),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT fk_attendance_student
                    FOREIGN KEY (student_id) REFERENCES students(id)
                    ON DELETE CASCADE,
                CONSTRAINT unique_student_attendance_day
                    UNIQUE (student_id, attendance_date)
            )
            """
        )
        connection.commit()
    finally:
        cursor.close()
        connection.close()


def get_db_connection():
    return mysql.connector.connect(**get_mysql_config())


initialize_database()


def fetch_all(query, params=None):
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute(query, params or ())
        return cursor.fetchall()
    finally:
        cursor.close()
        connection.close()


def fetch_one(query, params=None):
    rows = fetch_all(query, params)
    return rows[0] if rows else None


def execute_query(query, params=None, return_id=False):
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute(query, params or ())
        connection.commit()
        return cursor.lastrowid if return_id else None
    finally:
        cursor.close()
        connection.close()


def time_value_to_input_str(value):
    """Format MySQL TIME (time or timedelta) for HTML time inputs (HH:MM)."""
    if value is None:
        return ""
    if isinstance(value, time):
        return value.strftime("%H:%M")
    if isinstance(value, timedelta):
        total = int(value.total_seconds()) % (24 * 3600)
        hours, remainder = divmod(total, 3600)
        minutes, _ = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}"
    if isinstance(value, str) and len(value) >= 5:
        return value[:5]
    return str(value)[:8]


def parse_time_in_value(raw: str):
    stripped = (raw or "").strip()
    if not stripped:
        raise ValueError("Time in is required.")
    parts = stripped.split(":")
    hour = int(parts[0])
    minute = int(parts[1]) if len(parts) > 1 else 0
    second = int(parts[2]) if len(parts) > 2 else 0
    return time(hour, minute, second)


def decode_camera_image(image_data):
    if not image_data:
        raise ValueError("No camera image was received.")

    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(image_data)
    except ValueError as exc:
        raise ValueError("The submitted camera image is invalid.") from exc

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The submitted camera image could not be read.")

    return image


def extract_face(image, min_neighbors=6, scale_factor=1.08):
    if FACE_CASCADE.empty():
        raise RuntimeError("OpenCV could not load the Haar cascade face detector.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(80, 80),
    )
    if len(faces) == 0:
        return None

    x, y, width, height = max(faces, key=lambda face: face[2] * face[3])
    face = gray[y : y + height, x : x + width]
    face = cv2.resize(face, FACE_SIZE)
    return cv2.equalizeHist(face)


def create_recognizer():
    if not hasattr(cv2, "face"):
        raise RuntimeError(
            "OpenCV face recognizer is unavailable. Install opencv-contrib-python."
        )
    return cv2.face.LBPHFaceRecognizer_create(
        1,
        10,
        8,
        8,
        RECOGNITION_THRESHOLD,
    )


def face_sample_count(student_id):
    student_dir = FACES_DIR / str(student_id)
    if not student_dir.exists():
        return 0

    return len(list(student_dir.glob("*.png")))


def attach_dataset_counts(students):
    for student in students:
        student["dataset_samples"] = face_sample_count(student["id"])
        student["dataset_ready"] = student["dataset_samples"] >= 10
    return students


def save_face_sample(student_id, face):
    student_dir = FACES_DIR / str(student_id)
    student_dir.mkdir(exist_ok=True)
    existing_numbers = [
        int(path.stem.replace("sample_", ""))
        for path in student_dir.glob("sample_*.png")
        if path.stem.replace("sample_", "").isdigit()
    ]
    next_number = max(existing_numbers, default=0) + 1
    sample_path = student_dir / f"sample_{next_number}.png"
    cv2.imwrite(str(sample_path), face)
    return sample_path


def train_model():
    images = []
    labels = []

    for student_dir in FACES_DIR.iterdir():
        if not student_dir.is_dir() or not student_dir.name.isdigit():
            continue

        student_id = int(student_dir.name)
        for image_path in student_dir.glob("*.png"):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                images.append(image)
                labels.append(student_id)

    if not images:
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        return False

    recognizer = create_recognizer()
    recognizer.train(images, np.array(labels, dtype=np.int32))
    recognizer.setThreshold(RECOGNITION_THRESHOLD)
    recognizer.write(str(MODEL_PATH))
    return True


def recognize_student(image):
    if not MODEL_PATH.exists():
        return None, None, "No trained face model exists yet."

    face = extract_face(image)
    if face is None:
        return None, None, "No face was detected. Please face the camera clearly."

    recognizer = create_recognizer()
    recognizer.read(str(MODEL_PATH))
    recognizer.setThreshold(RECOGNITION_THRESHOLD)
    student_id, distance = recognizer.predict(face)

    if student_id < 0 or not np.isfinite(distance) or distance > RECOGNITION_THRESHOLD:
        safe = float(distance) if np.isfinite(distance) else None
        return None, safe, "Face was detected but did not match any student."

    return student_id, float(distance), None


@app.route("/")
def dashboard():
    today = date.today()
    stats = {
        "students": fetch_one("SELECT COUNT(*) AS total FROM students")["total"],
        "today_attendance": fetch_one(
            "SELECT COUNT(*) AS total FROM attendance WHERE attendance_date = %s",
            (today,),
        )["total"],
    }
    recent_attendance = fetch_all(
        """
        SELECT a.attendance_date, a.time_in, a.status, a.confidence,
               s.student_number, s.full_name, s.course, s.year_section
        FROM attendance a
        JOIN students s ON s.id = a.student_id
        ORDER BY a.attendance_date DESC, a.time_in DESC
        LIMIT 10
        """
    )
    return render_template(
        "dashboard.html",
        stats=stats,
        today=today,
        recent_attendance=recent_attendance,
    )


@app.route("/students", methods=["GET", "POST"])
def students():
    if request.method == "POST":
        student_number = request.form.get("student_number", "").strip()
        full_name = request.form.get("full_name", "").strip()
        course = request.form.get("course", "").strip()
        year_section = request.form.get("year_section", "").strip()
        email = request.form.get("email", "").strip()
        if not student_number or not full_name:
            flash("Student number and full name are required.", "error")
            return redirect(url_for("students"))

        try:
            pose_faces = []
            for pose_key in ENROLLMENT_POSE_ORDER:
                field_name = f"image_pose_{pose_key}"
                raw = request.form.get(field_name, "").strip()
                if not raw:
                    flash(
                        "Enrollment requires all five face poses. Capture front, left, "
                        "right, look down, and look up before saving.",
                        "error",
                    )
                    return redirect(url_for("students"))

                image = decode_camera_image(raw)
                face = extract_face(image)
                if face is None and pose_key != "front":
                    face = extract_face(image, min_neighbors=4, scale_factor=1.05)
                if face is None:
                    flash(
                        f"No face was detected for the {enrollment_pose_label(pose_key)}. "
                        "Retake that pose with clearer lighting and face the camera.",
                        "error",
                    )
                    return redirect(url_for("students"))
                pose_faces.append(face)

            student_id = execute_query(
                """
                INSERT INTO students
                    (student_number, full_name, course, year_section, email)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (student_number, full_name, course, year_section, email),
                return_id=True,
            )

            first_sample_path = None
            for face in pose_faces:
                sample_path = save_face_sample(student_id, face)
                if first_sample_path is None:
                    first_sample_path = sample_path

            execute_query(
                "UPDATE students SET face_image_path = %s WHERE id = %s",
                (str(first_sample_path.relative_to(BASE_DIR)), student_id),
            )
            train_model()
            flash(f"{full_name} was enrolled successfully with five face samples.", "success")
        except IntegrityError:
            flash("That student number is already enrolled.", "error")
        except (RuntimeError, ValueError, mysql.connector.Error) as exc:
            flash(str(exc), "error")

        return redirect(url_for("students"))

    enrolled_students = fetch_all(
        """
        SELECT id, student_number, full_name, course, year_section, email, created_at
        FROM students
        ORDER BY created_at DESC
        """
    )
    enrolled_students = attach_dataset_counts(enrolled_students)
    return render_template("students.html", students=enrolled_students)


@app.route("/students/<int:student_id>/delete", methods=["POST"])
def delete_student(student_id):
    student = fetch_one("SELECT full_name FROM students WHERE id = %s", (student_id,))
    if not student:
        flash("Student not found.", "error")
        return redirect(url_for("students"))

    execute_query("DELETE FROM students WHERE id = %s", (student_id,))
    shutil.rmtree(FACES_DIR / str(student_id), ignore_errors=True)
    train_model()
    flash(f"{student['full_name']} was removed.", "success")
    return redirect(url_for("students"))


@app.route("/dataset", methods=["GET", "POST"])
def dataset():
    if request.method == "POST":
        student_id = request.form.get("student_id", "").strip()
        image_data = request.form.get("image_data", "")

        if not student_id.isdigit():
            flash("Please choose a student before saving a dataset sample.", "error")
            return redirect(url_for("dataset"))

        student = fetch_one(
            "SELECT id, full_name, face_image_path FROM students WHERE id = %s",
            (int(student_id),),
        )
        if not student:
            flash("Student not found.", "error")
            return redirect(url_for("dataset"))

        try:
            image = decode_camera_image(image_data)
            face = extract_face(image)
            if face is None:
                flash("No face was detected. Capture another sample.", "error")
                return redirect(url_for("dataset"))

            sample_path = save_face_sample(student["id"], face)
            if not student["face_image_path"]:
                execute_query(
                    "UPDATE students SET face_image_path = %s WHERE id = %s",
                    (str(sample_path.relative_to(BASE_DIR)), student["id"]),
                )

            train_model()
            total_samples = face_sample_count(student["id"])
            flash(
                f"Dataset sample saved for {student['full_name']}. "
                f"Total samples: {total_samples}.",
                "success",
            )
        except (RuntimeError, ValueError, mysql.connector.Error) as exc:
            flash(str(exc), "error")

        return redirect(url_for("dataset"))

    students_for_dataset = fetch_all(
        """
        SELECT id, student_number, full_name, course, year_section
        FROM students
        ORDER BY full_name
        """
    )
    students_for_dataset = attach_dataset_counts(students_for_dataset)
    total_samples = sum(student["dataset_samples"] for student in students_for_dataset)

    return render_template(
        "dataset.html",
        students=students_for_dataset,
        total_samples=total_samples,
        model_exists=MODEL_PATH.exists(),
    )


@app.route("/dataset/train", methods=["POST"])
def train_dataset():
    try:
        trained = train_model()
        if trained:
            flash("Face recognition model retrained successfully.", "success")
        else:
            flash("No dataset samples were found to train the model.", "error")
    except RuntimeError as exc:
        flash(str(exc), "error")

    return redirect(url_for("dataset"))


@app.route("/attendance")
def attendance():
    return render_template("attendance.html")


@app.route("/api/attendance/recognize", methods=["POST"])
def api_recognize_attendance():
    payload = request.get_json(silent=True) or {}

    try:
        image = decode_camera_image(payload.get("image_data", ""))
        student_id, confidence, error = recognize_student(image)
        if error:
            return jsonify({"ok": False, "message": error, "confidence": confidence}), 400

        student = fetch_one(
            """
            SELECT id, student_number, full_name, course, year_section
            FROM students
            WHERE id = %s
            """,
            (student_id,),
        )
        if not student:
            return jsonify({"ok": False, "message": "Matched student no longer exists."}), 404

        today = date.today()
        existing = fetch_one(
            """
            SELECT id, time_in
            FROM attendance
            WHERE student_id = %s AND attendance_date = %s
            """,
            (student_id, today),
        )

        if existing:
            return jsonify(
                {
                    "ok": True,
                    "already_marked": True,
                    "message": "Attendance was already recorded today.",
                    "student": student,
                    "time_in": str(existing["time_in"]),
                    "confidence": round(float(confidence), 2),
                }
            )

        now = datetime.now().time().replace(microsecond=0)
        execute_query(
            """
            INSERT INTO attendance
                (student_id, attendance_date, time_in, status, confidence)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (student_id, today, now, "Present", float(confidence)),
        )

        return jsonify(
            {
                "ok": True,
                "already_marked": False,
                "message": "Attendance recorded successfully.",
                "student": student,
                "time_in": str(now),
                "confidence": round(float(confidence), 2),
            }
        )
    except (RuntimeError, ValueError, mysql.connector.Error) as exc:
        return jsonify({"ok": False, "message": str(exc)}), 400


@app.route("/reports")
def reports():
    selected_date = request.args.get("date", "").strip()
    params = []
    where_clause = ""

    if selected_date:
        where_clause = "WHERE a.attendance_date = %s"
        params.append(selected_date)

    records = fetch_all(
        f"""
        SELECT a.id, a.student_id, a.attendance_date, a.time_in, a.status, a.confidence,
               s.student_number, s.full_name, s.course, s.year_section
        FROM attendance a
        JOIN students s ON s.id = a.student_id
        {where_clause}
        ORDER BY a.attendance_date DESC, a.time_in DESC
        """,
        tuple(params),
    )
    return render_template(
        "reports.html",
        records=records,
        selected_date=selected_date,
    )


@app.route("/reports/attendance/<int:attendance_id>/edit", methods=["GET", "POST"])
def edit_attendance(attendance_id):
    record = fetch_one(
        """
        SELECT a.id, a.student_id, a.attendance_date, a.time_in, a.status, a.confidence,
               s.full_name, s.student_number
        FROM attendance a
        JOIN students s ON s.id = a.student_id
        WHERE a.id = %s
        """,
        (attendance_id,),
    )
    if not record:
        flash("Attendance record not found.", "error")
        return redirect(url_for("reports"))

    return_date = request.args.get("date", "").strip() if request.method == "GET" else ""
    if request.method == "POST":
        return_date = request.form.get("return_date", "").strip()

    if request.method == "POST":
        try:
            student_id = int(request.form.get("student_id", "0"))
            attendance_date = datetime.strptime(
                request.form.get("attendance_date", "").strip(), "%Y-%m-%d"
            ).date()
            time_in = parse_time_in_value(request.form.get("time_in", ""))
            status = request.form.get("status", "Present").strip()
            if status not in ("Present", "Late", "Absent"):
                raise ValueError("Invalid status.")

            conf_raw = request.form.get("confidence", "").strip()
            if conf_raw:
                confidence = float(conf_raw)
            else:
                confidence = record["confidence"]
                if confidence is not None:
                    confidence = float(confidence)

            student_exists = fetch_one("SELECT id FROM students WHERE id = %s", (student_id,))
            if not student_exists:
                flash("Selected student does not exist.", "error")
                return redirect(url_for("edit_attendance", attendance_id=attendance_id, date=return_date))

            execute_query(
                """
                UPDATE attendance
                SET student_id = %s, attendance_date = %s, time_in = %s, status = %s, confidence = %s
                WHERE id = %s
                """,
                (student_id, attendance_date, time_in, status, confidence, attendance_id),
            )
        except IntegrityError:
            flash(
                "Could not save: that student already has an attendance row for that date.",
                "error",
            )
            return redirect(url_for("edit_attendance", attendance_id=attendance_id, date=return_date))
        except (ValueError, TypeError) as exc:
            flash(str(exc), "error")
            return redirect(url_for("edit_attendance", attendance_id=attendance_id, date=return_date))
        except mysql.connector.Error as exc:
            flash(str(exc), "error")
            return redirect(url_for("edit_attendance", attendance_id=attendance_id, date=return_date))

        flash("Attendance record updated.", "success")
        return redirect(url_for("reports", date=str(attendance_date)))

    students = fetch_all(
        "SELECT id, student_number, full_name FROM students ORDER BY full_name, student_number"
    )
    time_in_value = time_value_to_input_str(record["time_in"])
    return render_template(
        "edit_attendance.html",
        record=record,
        students=students,
        time_in_value=time_in_value,
        return_date=return_date or request.args.get("date", "").strip(),
    )


@app.route("/reports/attendance/<int:attendance_id>/delete", methods=["POST"])
def delete_attendance(attendance_id):
    existing = fetch_one("SELECT id FROM attendance WHERE id = %s", (attendance_id,))
    if not existing:
        flash("Attendance record not found.", "error")
        return redirect(url_for("reports"))

    return_date = request.form.get("return_date", "").strip()
    execute_query("DELETE FROM attendance WHERE id = %s", (attendance_id,))
    flash("Attendance record deleted.", "success")
    if return_date:
        return redirect(url_for("reports", date=return_date))
    return redirect(url_for("reports"))


if __name__ == "__main__":
    app.run(debug=True)
