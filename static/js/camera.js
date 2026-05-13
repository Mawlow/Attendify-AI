const cameraStreams = new Map();

async function startCamera(videoElement) {
    if (!videoElement) {
        return;
    }

    if (cameraStreams.has(videoElement.id)) {
        return;
    }

    const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false,
    });
    videoElement.srcObject = stream;
    cameraStreams.set(videoElement.id, stream);
}

function captureFrame(videoElement, canvasElement, quality = 0.9) {
    if (!videoElement || !canvasElement) {
        throw new Error("Camera is not ready.");
    }

    if (!videoElement.videoWidth || !videoElement.videoHeight) {
        throw new Error("Start the camera before capturing.");
    }

    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;
    const context = canvasElement.getContext("2d");
    context.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
    return canvasElement.toDataURL("image/jpeg", quality);
}

function showResult(element, type, html) {
    if (!element) {
        return;
    }

    element.className = `result-box ${type}`;
    element.innerHTML = html;
}

function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

const STUDENT_ENROLLMENT_POSES = [
    { key: "front", title: "Front", hint: "Face the camera straight on." },
    {
        key: "left",
        title: "Left side",
        hint: "Turn your head so your left cheek faces the camera.",
    },
    {
        key: "right",
        title: "Right side",
        hint: "Turn your head so your right cheek faces the camera.",
    },
    {
        key: "down",
        title: "Look down",
        hint: "Lower your chin slightly—keep your face in the frame.",
    },
    {
        key: "up",
        title: "Look up",
        hint: "Raise your chin slightly toward the ceiling.",
    },
];

function setupStudentEnrollmentCamera() {
    const video = document.getElementById("student-video");
    const canvas = document.getElementById("student-canvas");
    const status = document.getElementById("student-capture-status");
    const startButton = document.getElementById("start-student-camera");
    const captureButton = document.getElementById("capture-student-pose");
    const retakeButton = document.getElementById("retake-student-pose");
    const form = document.getElementById("student-form");
    const phaseEl = document.getElementById("student-pose-phase");
    const titleEl = document.getElementById("student-pose-title");
    const hintEl = document.getElementById("student-pose-hint");
    const stepsList = document.getElementById("student-pose-steps");

    if (
        !video ||
        !canvas ||
        !status ||
        !startButton ||
        !captureButton ||
        !retakeButton ||
        !form ||
        !phaseEl ||
        !titleEl ||
        !hintEl ||
        !stepsList
    ) {
        return;
    }

    const hiddenByKey = {};
    for (const pose of STUDENT_ENROLLMENT_POSES) {
        const input = document.getElementById(`student-pose-${pose.key}`);
        if (!input) {
            return;
        }
        hiddenByKey[pose.key] = input;
    }

    let stepIndex = 0;

    function hiddenForStep(index) {
        return hiddenByKey[STUDENT_ENROLLMENT_POSES[index].key];
    }

    function syncUI() {
        const total = STUDENT_ENROLLMENT_POSES.length;
        const complete = stepIndex >= total;

        STUDENT_ENROLLMENT_POSES.forEach((_, i) => {
            const li = stepsList.querySelector(`[data-pose-index="${i}"]`);
            if (!li) {
                return;
            }
            li.classList.toggle("is-done", i < stepIndex);
            li.classList.toggle("is-current", !complete && i === stepIndex);
        });

        if (complete) {
            phaseEl.textContent = "All steps complete";
            titleEl.textContent = "Ready to enroll";
            hintEl.textContent =
                "All five poses are saved. Review the form, then click Save Student.";
            captureButton.disabled = true;
            captureButton.classList.remove("primary");
        } else {
            const pose = STUDENT_ENROLLMENT_POSES[stepIndex];
            phaseEl.textContent = `Step ${stepIndex + 1} of ${total}`;
            titleEl.textContent = pose.title;
            hintEl.textContent = pose.hint;
            captureButton.disabled = false;
            captureButton.classList.add("primary");
        }
    }

    syncUI();

    startButton.addEventListener("click", async () => {
        try {
            await startCamera(video);
            status.textContent = "Camera is ready. Follow each pose, then capture.";
        } catch (error) {
            status.textContent = error.message;
        }
    });

    captureButton.addEventListener("click", () => {
        if (stepIndex >= STUDENT_ENROLLMENT_POSES.length) {
            return;
        }
        try {
            const dataUrl = captureFrame(video, canvas, 0.82);
            hiddenForStep(stepIndex).value = dataUrl;
            stepIndex += 1;
            if (stepIndex >= STUDENT_ENROLLMENT_POSES.length) {
                status.textContent = "All five poses captured. You can save the student.";
            } else {
                const next = STUDENT_ENROLLMENT_POSES[stepIndex];
                status.textContent = `Saved. Next: ${next.title}.`;
            }
            syncUI();
        } catch (error) {
            status.textContent = error.message;
        }
    });

    retakeButton.addEventListener("click", () => {
        if (stepIndex >= STUDENT_ENROLLMENT_POSES.length) {
            stepIndex = STUDENT_ENROLLMENT_POSES.length - 1;
            hiddenForStep(stepIndex).value = "";
            status.textContent = "Retake the last pose (look up), then capture.";
            syncUI();
            return;
        }

        const currentInput = hiddenForStep(stepIndex);
        if (currentInput.value) {
            currentInput.value = "";
            status.textContent = `Cleared ${STUDENT_ENROLLMENT_POSES[stepIndex].title}. Capture again.`;
        } else if (stepIndex > 0) {
            stepIndex -= 1;
            hiddenForStep(stepIndex).value = "";
            status.textContent = `Back to ${STUDENT_ENROLLMENT_POSES[stepIndex].title}. Capture again.`;
        } else {
            status.textContent = "Nothing to clear yet.";
        }
        syncUI();
    });

    form.addEventListener("submit", (event) => {
        const missing = STUDENT_ENROLLMENT_POSES.some((pose) => !hiddenByKey[pose.key].value?.trim());
        if (missing) {
            event.preventDefault();
            status.textContent =
                "Capture all five poses (front, left, right, look down, look up) before saving.";
        }
    });
}

function setupDatasetRegistrationCamera() {
    const video = document.getElementById("dataset-video");
    const canvas = document.getElementById("dataset-canvas");
    const imageInput = document.getElementById("dataset-image-data");
    const status = document.getElementById("dataset-capture-status");
    const startButton = document.getElementById("start-dataset-camera");
    const captureButton = document.getElementById("capture-dataset-face");
    const form = document.getElementById("dataset-form");

    if (!video || !canvas || !startButton || !captureButton || !form) {
        return;
    }

    startButton.addEventListener("click", async () => {
        try {
            await startCamera(video);
            status.textContent = "Camera is ready. Capture samples from different angles.";
        } catch (error) {
            status.textContent = error.message;
        }
    });

    captureButton.addEventListener("click", () => {
        try {
            imageInput.value = captureFrame(video, canvas);
            status.textContent = "Dataset sample captured. Save it to retrain the model.";
        } catch (error) {
            status.textContent = error.message;
        }
    });

    form.addEventListener("submit", (event) => {
        if (!imageInput.value) {
            event.preventDefault();
            status.textContent = "Please capture a dataset sample before saving.";
        }
    });
}

function setupAttendanceCamera() {
    const video = document.getElementById("attendance-video");
    const canvas = document.getElementById("attendance-canvas");
    const startButton = document.getElementById("start-attendance-camera");
    const scanButton = document.getElementById("scan-attendance");
    const result = document.getElementById("attendance-result");

    if (!video || !canvas || !startButton || !scanButton) {
        return;
    }

    startButton.addEventListener("click", async () => {
        try {
            await startCamera(video);
            showResult(result, "info", "<p>Camera is ready. Face the camera and scan.</p>");
        } catch (error) {
            showResult(result, "error", `<p>${escapeHtml(error.message)}</p>`);
        }
    });

    scanButton.addEventListener("click", async () => {
        try {
            const imageData = captureFrame(video, canvas);
            showResult(result, "info", "<p>Scanning face...</p>");

            const response = await fetch("/api/attendance/recognize", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image_data: imageData }),
            });
            const data = await response.json();

            if (!response.ok || !data.ok) {
                showResult(
                    result,
                    "error",
                    `<p>${escapeHtml(data.message || "Recognition failed.")}</p>`
                );
                return;
            }

            const student = data.student;
            const duplicateText = data.already_marked
                ? "Attendance was already recorded today."
                : "Attendance recorded successfully.";

            showResult(
                result,
                "success",
                `
                    <h3>${escapeHtml(duplicateText)}</h3>
                    <p><strong>${escapeHtml(student.full_name)}</strong></p>
                    <p>${escapeHtml(student.student_number)}</p>
                    <p>${escapeHtml(student.course)} ${escapeHtml(student.year_section)}</p>
                    <p>Time In: ${escapeHtml(data.time_in)}</p>
                    <p>Confidence: ${escapeHtml(data.confidence)}</p>
                `
            );
        } catch (error) {
            showResult(result, "error", `<p>${escapeHtml(error.message)}</p>`);
        }
    });
}

document.addEventListener("DOMContentLoaded", () => {
    setupStudentEnrollmentCamera();
    setupDatasetRegistrationCamera();
    setupAttendanceCamera();
});
