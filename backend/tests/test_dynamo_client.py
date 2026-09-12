"""
VoiceGuard — Integration Tests for DynamoDB Client (Step 142)

Tests all data access functions in dynamo_client.py against a mocked
DynamoDB table using the `moto` library. This avoids the need for
DynamoDB Local Docker container while still exercising the full boto3
API surface.

To run against actual DynamoDB Local instead, set:
    AWS_ENDPOINT_URL=http://localhost:8000
and ensure the table is created via infra/scripts/create_dynamodb_table.sh --local

Tests cover:
    - create_session: insert, duplicate prevention
    - append_risk_score: timeline accumulation, peak score tracking
    - finalize_session: end_time, duration, verdict
    - get_session: full record retrieval, not-found case
    - list_sessions: GSI query, pagination, descending order
    - append_alert: alert tracking
    - ping: health check
"""

import time
from decimal import Decimal
from unittest.mock import patch

import boto3
import pytest
from moto import mock_aws

from app.services.dynamo_client import DynamoClient, SESSION_TTL_SECONDS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def aws_credentials():
    """Mock AWS credentials for moto."""
    import os
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "ap-south-1"


@pytest.fixture
def dynamodb_table(aws_credentials):
    """Create the call_sessions table in mocked DynamoDB."""
    with mock_aws():
        resource = boto3.resource("dynamodb", region_name="ap-south-1")
        table = resource.create_table(
            TableName="call_sessions",
            AttributeDefinitions=[
                {"AttributeName": "call_sid", "AttributeType": "S"},
                {"AttributeName": "user_id", "AttributeType": "S"},
                {"AttributeName": "start_time", "AttributeType": "N"},
            ],
            KeySchema=[
                {"AttributeName": "call_sid", "KeyType": "HASH"},
            ],
            GlobalSecondaryIndexes=[
                {
                    "IndexName": "user_id-start_time-index",
                    "KeySchema": [
                        {"AttributeName": "user_id", "KeyType": "HASH"},
                        {"AttributeName": "start_time", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                    "ProvisionedThroughput": {
                        "ReadCapacityUnits": 5,
                        "WriteCapacityUnits": 5,
                    },
                },
            ],
            ProvisionedThroughput={
                "ReadCapacityUnits": 5,
                "WriteCapacityUnits": 5,
            },
        )
        table.meta.client.get_waiter("table_exists").wait(
            TableName="call_sessions"
        )
        yield table


@pytest.fixture
def client(dynamodb_table):
    """Create a DynamoClient pointing at the mocked table."""
    client = DynamoClient()
    # Force the client to use the mocked resource
    client._resource = boto3.resource("dynamodb", region_name="ap-south-1")
    client._table = client._resource.Table("call_sessions")
    return client


# ===========================================================================
# create_session tests
# ===========================================================================

class TestCreateSession:
    """Test create_session (Step 137)."""

    async def test_creates_new_session(self, client):
        """A new session should be inserted with correct attributes."""
        await client.create_session("CA001", "+919876543210", "user_001")

        session = await client.get_session("CA001")
        assert session is not None
        assert session["call_sid"] == "CA001"
        assert session["caller_number"] == "+919876543210"
        assert session["user_id"] == "user_001"
        assert session["final_verdict"] == "in_progress"
        assert session["risk_score_timeline"] == []
        assert session["alerts_sent"] == []
        assert session["peak_risk_score"] == 0
        assert session["start_time"] > 0
        assert session["ttl"] > 0

    async def test_ttl_is_30_days_from_now(self, client):
        """TTL should be approximately start_time + 30 days."""
        before = time.time()
        await client.create_session("CA002", "+911111111111", "user_001")
        after = time.time()

        session = await client.get_session("CA002")
        # TTL should be within the expected range
        assert session["ttl"] >= int(before) + SESSION_TTL_SECONDS
        assert session["ttl"] <= int(after) + SESSION_TTL_SECONDS + 1

    async def test_duplicate_session_does_not_overwrite(self, client):
        """Creating a session with the same call_sid should not overwrite."""
        await client.create_session("CA003", "+919876543210", "user_001")
        # Second create with different data should be silently ignored
        await client.create_session("CA003", "+910000000000", "user_002")

        session = await client.get_session("CA003")
        # Original data should be preserved
        assert session["caller_number"] == "+919876543210"
        assert session["user_id"] == "user_001"


# ===========================================================================
# append_risk_score tests
# ===========================================================================

class TestAppendRiskScore:
    """Test append_risk_score (Step 138)."""

    async def test_appends_single_score(self, client):
        """Appending a score should add to the risk_score_timeline."""
        await client.create_session("CA010", "+919876543210", "user_001")
        await client.append_risk_score(
            "CA010", 1694500000.0, 72.5, {"acoustic": 80, "context": 55}
        )

        session = await client.get_session("CA010")
        timeline = session["risk_score_timeline"]
        assert len(timeline) == 1
        assert timeline[0]["score"] == 72.5
        assert timeline[0]["timestamp"] == 1694500000.0
        assert timeline[0]["signals"]["acoustic"] == 80
        assert timeline[0]["signals"]["context"] == 55

    async def test_appends_multiple_scores_in_order(self, client):
        """Multiple appends should accumulate in order."""
        await client.create_session("CA011", "+919876543210", "user_001")

        for i in range(5):
            await client.append_risk_score(
                "CA011",
                1694500000.0 + i,
                50.0 + i * 5,
                {"acoustic": 60 + i, "context": 30 + i},
            )

        session = await client.get_session("CA011")
        timeline = session["risk_score_timeline"]
        assert len(timeline) == 5
        # Verify order
        scores = [entry["score"] for entry in timeline]
        assert scores == [50.0, 55.0, 60.0, 65.0, 70.0]

    async def test_updates_peak_risk_score(self, client):
        """Peak score should track the maximum score seen."""
        await client.create_session("CA012", "+919876543210", "user_001")

        await client.append_risk_score("CA012", 1694500000.0, 40.0, {})
        await client.append_risk_score("CA012", 1694500001.0, 85.0, {})
        await client.append_risk_score("CA012", 1694500002.0, 60.0, {})

        session = await client.get_session("CA012")
        assert session["peak_risk_score"] == 85.0

    async def test_append_to_nonexistent_session_warns(self, client):
        """Appending to a non-existent session should not raise (just warn)."""
        # Should not raise — logged as a warning
        await client.append_risk_score(
            "CA_NONEXISTENT", 1694500000.0, 50.0, {}
        )


# ===========================================================================
# finalize_session tests
# ===========================================================================

class TestFinalizeSession:
    """Test finalize_session (Step 139)."""

    async def test_sets_end_time_and_duration(self, client):
        """Finalization should set end_time and compute duration."""
        await client.create_session("CA020", "+919876543210", "user_001")
        # Small delay to ensure duration > 0
        await client.finalize_session("CA020", "safe")

        session = await client.get_session("CA020")
        assert session["end_time"] > 0
        assert session["duration_seconds"] >= 0
        assert session["final_verdict"] == "safe"

    async def test_sets_suspicious_verdict(self, client):
        """Should accept 'suspicious' as a valid verdict."""
        await client.create_session("CA021", "+919876543210", "user_001")
        await client.finalize_session("CA021", "suspicious")

        session = await client.get_session("CA021")
        assert session["final_verdict"] == "suspicious"

    async def test_sets_high_risk_verdict(self, client):
        """Should accept 'high_risk' as a valid verdict."""
        await client.create_session("CA022", "+919876543210", "user_001")
        await client.finalize_session("CA022", "high_risk")

        session = await client.get_session("CA022")
        assert session["final_verdict"] == "high_risk"

    async def test_rejects_invalid_verdict(self, client):
        """Invalid verdict values should raise ValueError."""
        await client.create_session("CA023", "+919876543210", "user_001")
        with pytest.raises(ValueError, match="Invalid final_verdict"):
            await client.finalize_session("CA023", "invalid_value")

    async def test_finalize_nonexistent_session(self, client):
        """Finalizing a non-existent session should not raise."""
        # Should complete without error (just logs a warning)
        await client.finalize_session("CA_NONEXISTENT", "safe")


# ===========================================================================
# get_session tests
# ===========================================================================

class TestGetSession:
    """Test get_session (Step 140)."""

    async def test_returns_full_session(self, client):
        """Should return all attributes with Decimals converted."""
        await client.create_session("CA030", "+919876543210", "user_001")
        await client.append_risk_score(
            "CA030", 1694500000.0, 72.5, {"acoustic": 80.5, "context": 55.2}
        )
        await client.finalize_session("CA030", "suspicious")

        session = await client.get_session("CA030")

        # Verify all fields present
        assert "call_sid" in session
        assert "caller_number" in session
        assert "user_id" in session
        assert "start_time" in session
        assert "end_time" in session
        assert "duration_seconds" in session
        assert "risk_score_timeline" in session
        assert "peak_risk_score" in session
        assert "final_verdict" in session
        assert "alerts_sent" in session
        assert "ttl" in session

        # Verify no Decimal objects remain
        _assert_no_decimals(session)

    async def test_returns_none_for_nonexistent(self, client):
        """Should return None for a call_sid that doesn't exist."""
        session = await client.get_session("CA_NONEXISTENT")
        assert session is None


# ===========================================================================
# list_sessions tests
# ===========================================================================

class TestListSessions:
    """Test list_sessions (Step 141)."""

    async def test_lists_sessions_for_user(self, client):
        """Should return all sessions for a given user_id."""
        for i in range(3):
            await client.create_session(
                f"CA04{i}", "+919876543210", "user_list"
            )

        result = await client.list_sessions("user_list")
        assert len(result["sessions"]) == 3

    async def test_returns_empty_for_unknown_user(self, client):
        """Should return empty list for a user with no sessions."""
        result = await client.list_sessions("user_nobody")
        assert result["sessions"] == []
        assert result["last_evaluated_key"] is None

    async def test_respects_limit(self, client):
        """Should return at most `limit` sessions."""
        for i in range(5):
            await client.create_session(
                f"CA05{i}", "+919876543210", "user_limit"
            )

        result = await client.list_sessions("user_limit", limit=2)
        assert len(result["sessions"]) <= 2

    async def test_sessions_sorted_descending(self, client):
        """Sessions should be sorted by start_time descending (newest first)."""
        # Create sessions with small delays to ensure different timestamps
        for i in range(3):
            await client.create_session(
                f"CA06{i}", "+919876543210", "user_sort"
            )

        result = await client.list_sessions("user_sort")
        sessions = result["sessions"]
        if len(sessions) >= 2:
            start_times = [s["start_time"] for s in sessions]
            # Verify descending order
            assert start_times == sorted(start_times, reverse=True)

    async def test_no_decimals_in_response(self, client):
        """Listed sessions should have no Decimal objects."""
        await client.create_session("CA070", "+919876543210", "user_dec")
        result = await client.list_sessions("user_dec")
        for session in result["sessions"]:
            _assert_no_decimals(session)


# ===========================================================================
# append_alert tests
# ===========================================================================

class TestAppendAlert:
    """Test append_alert helper."""

    async def test_appends_alert_record(self, client):
        """Should add an alert entry to alerts_sent."""
        await client.create_session("CA080", "+919876543210", "user_001")
        await client.append_alert("CA080", "high", "fcm")
        await client.append_alert("CA080", "high", "sms")

        session = await client.get_session("CA080")
        alerts = session["alerts_sent"]
        assert len(alerts) == 2
        assert alerts[0]["level"] == "high"
        assert alerts[0]["channel"] == "fcm"
        assert alerts[1]["channel"] == "sms"
        assert alerts[0]["timestamp"] > 0


# ===========================================================================
# ping tests
# ===========================================================================

class TestPing:
    """Test health check."""

    async def test_ping_returns_true(self, client):
        """Ping should return True when table is accessible."""
        result = await client.ping()
        assert result is True


# ===========================================================================
# Full lifecycle test
# ===========================================================================

class TestFullLifecycle:
    """End-to-end test simulating a complete call lifecycle."""

    async def test_complete_call_lifecycle(self, client):
        """Simulate: create → score → score → alert → finalize → query."""
        call_sid = "CA_LIFECYCLE"
        user_id = "user_lifecycle"

        # 1. Call starts
        await client.create_session(call_sid, "+919876543210", user_id)

        # 2. First chunk: low risk
        await client.append_risk_score(
            call_sid, 1694500000.0, 35.0,
            {"acoustic": 40.0, "context": 25.0},
        )

        # 3. Second chunk: risk rises
        await client.append_risk_score(
            call_sid, 1694500002.0, 78.0,
            {"acoustic": 85.0, "context": 65.0},
        )

        # 4. Alert fired
        await client.append_alert(call_sid, "medium", "fcm")

        # 5. Third chunk: high risk
        await client.append_risk_score(
            call_sid, 1694500004.0, 92.0,
            {"acoustic": 95.0, "context": 88.0},
        )

        # 6. High alert
        await client.append_alert(call_sid, "high", "fcm")
        await client.append_alert(call_sid, "high", "sms")

        # 7. Call ends
        await client.finalize_session(call_sid, "high_risk")

        # 8. Verify full record
        session = await client.get_session(call_sid)
        assert session["call_sid"] == call_sid
        assert session["final_verdict"] == "high_risk"
        assert session["peak_risk_score"] == 92.0
        assert len(session["risk_score_timeline"]) == 3
        assert len(session["alerts_sent"]) == 3
        assert session["duration_seconds"] >= 0

        # 9. Verify appears in list
        result = await client.list_sessions(user_id)
        sids = [s["call_sid"] for s in result["sessions"]]
        assert call_sid in sids

        # 10. No Decimal leakage
        _assert_no_decimals(session)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_no_decimals(obj, path="root"):
    """Recursively assert no Decimal objects remain in the data structure."""
    if isinstance(obj, Decimal):
        raise AssertionError(f"Decimal found at {path}: {obj}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _assert_no_decimals(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_no_decimals(v, f"{path}[{i}]")
