CREATE DATABASE IF NOT EXISTS attendify_ai
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE attendify_ai;

CREATE TABLE IF NOT EXISTS students (
    id INT AUTO_INCREMENT PRIMARY KEY,
    student_number VARCHAR(50) NOT NULL UNIQUE,
    full_name VARCHAR(150) NOT NULL,
    course VARCHAR(100),
    year_section VARCHAR(100),
    email VARCHAR(150),
    face_image_path VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

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
);
