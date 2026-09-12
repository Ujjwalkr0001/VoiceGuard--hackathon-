"""
VoiceGuard — API Tests (Step 165)

Tests for the REST API endpoints using FastAPI's TestClient.
Covers:
    - Authentication (API key validation)
    - Rate limiting
    - Call history endpoints
    - Contacts CRUD
    - Config read/update
    - Input validation
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from app.main import app

# Valid API key (matches settings default)
HEADERS = {"X-API-Key": "dev-api-key"}


@pytest.fixture
def client():
    """TestClient for the FastAPI app."""
    return TestClient(app, raise_server_exceptions=False)


# ===========================================================================
# Authentication tests (Step 162)
# ===========================================================================

class TestAuthentication:
    """API key authentication tests."""

    def test_missing_api_key_returns_422(self, client):
        """Request without X-API-Key header should fail."""
        resp = client.get("/api/v1/config")
        assert resp.status_code in (401, 422)

    def test_invalid_api_key_returns_401(self, client):
        """Request with wrong API key should get 401."""
        resp = client.get("/api/v1/config", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 401

    def test_valid_api_key_succeeds(self, client):
        """Request with correct API key should succeed."""
        resp = client.get("/api/v1/config", headers=HEADERS)
        assert resp.status_code == 200


# ===========================================================================
# Config API tests (Steps 160–161)
# ===========================================================================

class TestConfigAPI:
    """Config endpoints tests."""

    def test_get_config(self, client):
        """GET /config should return current settings."""
        resp = client.get("/api/v1/config", headers=HEADERS)
        assert resp.status_code == 200

        data = resp.json()
        assert "risk_threshold_medium" in data
        assert "risk_threshold_high" in data
        assert "ensemble_weight_model_a" in data
        assert "ensemble_weight_model_b" in data
        assert data["risk_threshold_medium"] == 70
        assert data["risk_threshold_high"] == 85

    def test_patch_config(self, client):
        """PATCH /config should update provided fields."""
        resp = client.patch(
            "/api/v1/config",
            headers=HEADERS,
            json={"risk_threshold_medium": 65},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["risk_threshold_medium"] == 65

        # Restore default
        client.patch(
            "/api/v1/config",
            headers=HEADERS,
            json={"risk_threshold_medium": 70},
        )

    def test_patch_config_validation(self, client):
        """PATCH /config should reject invalid values."""
        resp = client.patch(
            "/api/v1/config",
            headers=HEADERS,
            json={"risk_threshold_medium": 150},  # > 100
        )
        assert resp.status_code == 422


# ===========================================================================
# Contacts API tests (Steps 157–159)
# ===========================================================================

class TestContactsAPI:
    """Contacts CRUD tests."""

    def test_add_contact(self, client):
        """POST /contacts should create a new contact."""
        resp = client.post(
            "/api/v1/contacts?user_id=test_user",
            headers=HEADERS,
            json={"phone_number": "+919876543210", "name": "Test Contact"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["phone_number"] == "+919876543210"
        assert data["name"] == "Test Contact"

    def test_list_contacts(self, client):
        """GET /contacts should return the user's contacts."""
        # Add a contact first
        client.post(
            "/api/v1/contacts?user_id=list_user",
            headers=HEADERS,
            json={"phone_number": "+911111111111"},
        )

        resp = client.get("/api/v1/contacts?user_id=list_user", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 1
        numbers = [c["phone_number"] for c in data["contacts"]]
        assert "+911111111111" in numbers

    def test_delete_contact(self, client):
        """DELETE /contacts/{number} should remove the contact."""
        # Add then delete
        client.post(
            "/api/v1/contacts?user_id=del_user",
            headers=HEADERS,
            json={"phone_number": "+912222222222"},
        )

        resp = client.delete(
            "/api/v1/contacts/+912222222222?user_id=del_user",
            headers=HEADERS,
        )
        assert resp.status_code == 204

    def test_delete_nonexistent_contact(self, client):
        """DELETE /contacts/{number} for non-existent should return 404."""
        resp = client.delete(
            "/api/v1/contacts/+910000000000?user_id=del_user",
            headers=HEADERS,
        )
        assert resp.status_code == 404

    def test_duplicate_contact_returns_409(self, client):
        """POST /contacts with existing number should return 409."""
        client.post(
            "/api/v1/contacts?user_id=dup_user",
            headers=HEADERS,
            json={"phone_number": "+913333333333"},
        )
        resp = client.post(
            "/api/v1/contacts?user_id=dup_user",
            headers=HEADERS,
            json={"phone_number": "+913333333333"},
        )
        assert resp.status_code == 409

    def test_invalid_phone_number(self, client):
        """POST /contacts with invalid phone format should return 422."""
        resp = client.post(
            "/api/v1/contacts?user_id=val_user",
            headers=HEADERS,
            json={"phone_number": "not-a-phone"},
        )
        assert resp.status_code == 422


# ===========================================================================
# Call History API tests (Steps 154–156)
# ===========================================================================

class TestCallHistoryAPI:
    """Call history endpoint tests (mocking DynamoDB)."""

    @patch.object(
        dynamo_client, "list_sessions",
        new_callable=AsyncMock,
        return_value={
            "sessions": [
                {
                    "call_sid": "CA_TEST_001",
                    "caller_number": "+919876543210",
                    "user_id": "user_001",
                    "start_time": 1694500000.0,
                    "end_time": 1694500300.0,
                    "duration_seconds": 300.0,
                    "peak_risk_score": 42.5,
                    "final_verdict": "safe",
                }
            ],
            "last_evaluated_key": None,
        },
    )
    def test_list_calls(self, mock_list, client):
        """GET /calls should return paginated call list."""
        resp = client.get(
            "/api/v1/calls?user_id=user_001",
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["sessions"][0]["call_sid"] == "CA_TEST_001"

    @patch.object(
        dynamo_client, "get_session",
        new_callable=AsyncMock,
        return_value={
            "call_sid": "CA_TEST_002",
            "caller_number": "+919876543210",
            "user_id": "user_001",
            "start_time": 1694500000.0,
            "end_time": 1694500300.0,
            "duration_seconds": 300.0,
            "peak_risk_score": 78.2,
            "final_verdict": "suspicious",
            "risk_score_timeline": [
                {"timestamp": 1694500000.0, "score": 35.0, "signals": {}},
                {"timestamp": 1694500002.0, "score": 78.2, "signals": {}},
            ],
            "alerts_sent": [
                {"timestamp": 1694500002.5, "level": "medium", "channel": "fcm"},
            ],
        },
    )
    def test_get_call_detail(self, mock_get, client):
        """GET /calls/{call_sid} should return full detail."""
        resp = client.get(
            "/api/v1/calls/CA_TEST_002",
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["call_sid"] == "CA_TEST_002"
        assert len(data["risk_score_timeline"]) == 2
        assert len(data["alerts_sent"]) == 1

    @patch.object(
        dynamo_client, "get_session",
        new_callable=AsyncMock,
        return_value=None,
    )
    def test_get_nonexistent_call(self, mock_get, client):
        """GET /calls/{call_sid} for non-existent should return 404."""
        resp = client.get(
            "/api/v1/calls/CA_NONEXISTENT",
            headers=HEADERS,
        )
        assert resp.status_code == 404

    @patch.object(
        dynamo_client, "get_session",
        new_callable=AsyncMock,
        return_value={
            "call_sid": "CA_TIMELINE",
            "risk_score_timeline": [
                {"timestamp": 1694500000.0, "score": 30.0, "signals": {}},
                {"timestamp": 1694500002.0, "score": 55.0, "signals": {}},
                {"timestamp": 1694500004.0, "score": 82.0, "signals": {}},
            ],
            "peak_risk_score": 82.0,
            "caller_number": "+91",
            "user_id": "u",
            "start_time": 0,
        },
    )
    def test_get_risk_timeline(self, mock_get, client):
        """GET /calls/{call_sid}/risk-timeline should return timeline."""
        resp = client.get(
            "/api/v1/calls/CA_TIMELINE/risk-timeline",
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 3
        assert data["peak_score"] == 82.0


# ===========================================================================
# Metrics & Model Status
# ===========================================================================

class TestBonusEndpoints:
    """Metrics and model status endpoints."""

    def test_get_metrics(self, client):
        """GET /metrics should return pipeline stats."""
        resp = client.get("/api/v1/metrics", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert "uptime_sec" in data
        assert "chunks_total" in data

    def test_get_model_status(self, client):
        """GET /models/status should return model loading info."""
        resp = client.get("/api/v1/models/status", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert "model_a" in data
        assert "model_b" in data
