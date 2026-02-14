"""
API Tests for AIVerifySnap ML Service.

Tests the FastAPI endpoints including prediction, health checks,
and error handling.
"""

import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


def make_test_image(width: int = 128, height: int = 128, color: tuple = (120, 80, 200)) -> bytes:
    """Create a test image as PNG bytes."""
    image = Image.new("RGB", (width, height), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def make_test_jpeg_image(width: int = 128, height: int = 128) -> bytes:
    """Create a test image as JPEG bytes."""
    image = Image.new("RGB", (width, height), color=(100, 150, 200))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


@pytest.fixture
def client():
    """Create test client fixture."""
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_returns_ok(self, client) -> None:
        """Health endpoint should return ok status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["ok", "degraded"]
        assert "model_backend" in data
        assert "model_loaded" in data
        assert "model_status" in data

    def test_model_info(self, client) -> None:
        """Model info endpoint should return architecture details."""
        response = client.get("/model-info")
        assert response.status_code == 200
        data = response.json()
        assert data["architecture"] == "AIVerifyNet Dual-Stream Hybrid Network"
        assert "spatial_stream" in data
        assert "frequency_stream" in data
        assert "supported_formats" in data


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""

    def test_predict_multipart_png(self, client) -> None:
        """Predict should accept PNG images via multipart upload."""
        image_bytes = make_test_image()
        response = client.post(
            "/predict",
            files={"file": ("test.png", image_bytes, "image/png")},
        )
        assert response.status_code == 200
        payload = response.json()
        assert "is_deepfake" in payload
        assert "prediction" in payload
        assert payload["prediction"] in ["Real", "Fake"]
        assert 0.0 <= payload["confidence"] <= 1.0
        assert "processing_time_ms" in payload
        assert "details" in payload

    def test_predict_multipart_jpeg(self, client) -> None:
        """Predict should accept JPEG images via multipart upload."""
        image_bytes = make_test_jpeg_image()
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        payload = response.json()
        assert "is_deepfake" in payload
        assert 0.0 <= payload["confidence"] <= 1.0

    def test_predict_returns_filename(self, client) -> None:
        """Predict should return the original filename."""
        image_bytes = make_test_image()
        response = client.post(
            "/predict",
            files={"file": ("my_image.png", image_bytes, "image/png")},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["filename"] == "my_image.png"

    def test_predict_rejects_non_image(self, client) -> None:
        """Predict should reject non-image files."""
        response = client.post(
            "/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data

    def test_predict_rejects_invalid_image_bytes(self, client) -> None:
        """Predict should reject invalid image data."""
        response = client.post(
            "/predict",
            files={"file": ("fake.png", b"invalid image data", "image/png")},
        )
        assert response.status_code == 400


class TestPredictBase64Endpoint:
    """Tests for the /predict-base64 endpoint."""

    def test_predict_base64_valid(self, client) -> None:
        """Predict-base64 should accept valid base64 images."""
        image_bytes = make_test_image()
        payload = {"image_base64": base64.b64encode(image_bytes).decode("ascii")}
        response = client.post("/predict-base64", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "is_deepfake" in data
        assert "model_status" in data
        assert "confidence" in data

    def test_predict_base64_with_data_url(self, client) -> None:
        """Predict-base64 should handle data URL format."""
        image_bytes = make_test_image()
        b64_string = base64.b64encode(image_bytes).decode("ascii")
        payload = {"image_base64": f"data:image/png;base64,{b64_string}"}
        response = client.post("/predict-base64", json=payload)
        assert response.status_code == 200

    def test_predict_base64_missing_field(self, client) -> None:
        """Predict-base64 should reject requests without image_base64 field."""
        response = client.post("/predict-base64", json={})
        assert response.status_code == 400
        data = response.json()
        assert "image_base64" in data["detail"].lower()

    def test_predict_base64_invalid_encoding(self, client) -> None:
        """Predict-base64 should reject invalid base64 encoding."""
        payload = {"image_base64": "not_valid_base64!!!"}
        response = client.post("/predict-base64", json=payload)
        assert response.status_code == 400


class TestResponseFormat:
    """Tests for response format consistency."""

    def test_response_has_all_required_fields(self, client) -> None:
        """Prediction response should have all required fields."""
        image_bytes = make_test_image()
        response = client.post(
            "/predict",
            files={"file": ("test.png", image_bytes, "image/png")},
        )
        assert response.status_code == 200
        data = response.json()

        required_fields = [
            "filename",
            "prediction",
            "is_deepfake",
            "confidence",
            "raw_score",
            "processing_time_ms",
            "model_status",
            "details",
        ]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

    def test_details_has_backend_info(self, client) -> None:
        """Details should include backend information."""
        image_bytes = make_test_image()
        response = client.post(
            "/predict",
            files={"file": ("test.png", image_bytes, "image/png")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "backend" in data["details"]


class TestDifferentImageSizes:
    """Tests for handling different image sizes."""

    def test_small_image(self, client) -> None:
        """Should handle small images."""
        image_bytes = make_test_image(32, 32)
        response = client.post(
            "/predict",
            files={"file": ("small.png", image_bytes, "image/png")},
        )
        assert response.status_code == 200

    def test_large_image(self, client) -> None:
        """Should handle and resize large images."""
        image_bytes = make_test_image(2048, 2048)
        response = client.post(
            "/predict",
            files={"file": ("large.png", image_bytes, "image/png")},
        )
        assert response.status_code == 200

    def test_non_square_image(self, client) -> None:
        """Should handle non-square images."""
        image_bytes = make_test_image(640, 480)
        response = client.post(
            "/predict",
            files={"file": ("rect.png", image_bytes, "image/png")},
        )
        assert response.status_code == 200

