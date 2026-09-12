"""
VoiceGuard Backend — DynamoDB Client (Steps 136–141)

Provides async-compatible data access functions for the `call_sessions`
DynamoDB table. Persists call session records, risk score timelines,
and alert history for post-call review and the call history API.

Table Schema:
    Partition Key: call_sid (String)
    GSI: user_id-start_time-index (user_id HASH, start_time RANGE)
    TTL: `ttl` attribute (auto-deletes after 30 days)

Note: boto3 DynamoDB operations are synchronous. We run them in a
thread pool via asyncio.to_thread() to avoid blocking the event loop.

Usage:
    from app.services.dynamo_client import dynamo_client

    # Create session when call starts
    await dynamo_client.create_session("CA123", "+919876543210", "user_001")

    # Append risk scores during the call
    await dynamo_client.append_risk_score("CA123", 1694500000.0, 72.5, {"acoustic": 80, "context": 55})

    # Finalize when call ends
    await dynamo_client.finalize_session("CA123", "suspicious")

    # Query for API
    sessions = await dynamo_client.list_sessions("user_001", limit=20)
    session = await dynamo_client.get_session("CA123")
"""

import asyncio
import logging
import time
from decimal import Decimal
from typing import Any, Optional

import boto3
from botocore.exceptions import ClientError

from app.config import settings

logger = logging.getLogger(__name__)

# DynamoDB TTL: 30 days in seconds
SESSION_TTL_SECONDS = 30 * 24 * 60 * 60  # 2,592,000 seconds


# ---------------------------------------------------------------------------
# Decimal conversion helpers
#
# DynamoDB requires Decimal for numeric types and does not accept
# Python floats. These helpers handle the conversion transparently.
# ---------------------------------------------------------------------------

def _to_decimal(value: float | int) -> Decimal:
    """Convert a numeric value to Decimal for DynamoDB storage."""
    if isinstance(value, float):
        # Round to avoid floating-point representation issues
        return Decimal(str(round(value, 6)))
    return Decimal(value)


def _from_decimal(value: Any) -> float | int | Any:
    """Convert a DynamoDB Decimal back to a Python numeric type."""
    if isinstance(value, Decimal):
        # Return int if the value has no fractional part
        if value == value.to_integral_value():
            return int(value)
        return float(value)
    return value


def _deserialize_item(item: dict) -> dict:
    """Recursively convert all Decimal values in a DynamoDB item to Python types."""
    result = {}
    for key, value in item.items():
        if isinstance(value, Decimal):
            result[key] = _from_decimal(value)
        elif isinstance(value, dict):
            result[key] = _deserialize_item(value)
        elif isinstance(value, list):
            result[key] = [
                _deserialize_item(v) if isinstance(v, dict)
                else _from_decimal(v) if isinstance(v, Decimal)
                else v
                for v in value
            ]
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# DynamoDB Client Class
# ---------------------------------------------------------------------------

class DynamoClient:
    """
    Async-compatible DynamoDB client for the `call_sessions` table.

    All public methods are async and run boto3 calls in a thread pool
    to avoid blocking the FastAPI event loop.
    """

    def __init__(self) -> None:
        self._table = None
        self._resource = None

    def _get_table(self):
        """
        Lazily initialize the DynamoDB table resource.

        Uses lazy init so the client can be instantiated at module level
        without requiring AWS credentials at import time.
        """
        if self._table is None:
            endpoint_url = settings.dynamodb_endpoint_url or None

            # DynamoDB Local ignores credentials but boto3 still refuses to
            # sign a request without them, so fall back to dummy values.
            access_key = settings.aws_access_key_id or None
            secret_key = settings.aws_secret_access_key or None
            if endpoint_url and not (access_key and secret_key):
                access_key = access_key or "local"
                secret_key = secret_key or "local"

            self._resource = boto3.resource(
                "dynamodb",
                region_name=settings.aws_region,
                endpoint_url=endpoint_url,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
            )
            self._table = self._resource.Table(settings.dynamodb_table_name)
            logger.info(
                "dynamo_client.initialized",
                extra={
                    "table_name": settings.dynamodb_table_name,
                    "region": settings.aws_region,
                    "endpoint_url": endpoint_url or "aws",
                },
            )
        return self._table

    # -------------------------------------------------------------------
    # Step 137 — create_session
    # -------------------------------------------------------------------

    async def create_session(
        self,
        call_sid: str,
        caller_number: str,
        user_id: str,
    ) -> None:
        """
        Insert a new call session record when a call starts.

        Args:
            call_sid: Twilio call SID (partition key).
            caller_number: Caller's phone number (E.164 format).
            user_id: ID of the enrolled VoiceGuard user receiving the call.
        """
        now = time.time()
        ttl_epoch = int(now) + SESSION_TTL_SECONDS

        item = {
            "call_sid": call_sid,
            "caller_number": caller_number,
            "user_id": user_id,
            "start_time": _to_decimal(now),
            "end_time": _to_decimal(0),
            "duration_seconds": _to_decimal(0),
            "risk_score_timeline": [],
            "peak_risk_score": _to_decimal(0),
            "final_verdict": "in_progress",
            "alerts_sent": [],
            "ttl": ttl_epoch,
        }

        def _put():
            table = self._get_table()
            table.put_item(
                Item=item,
                ConditionExpression="attribute_not_exists(call_sid)",
            )

        try:
            await asyncio.to_thread(_put)
            logger.info(
                "dynamo_client.session_created",
                extra={
                    "call_sid": call_sid,
                    "caller_number": caller_number,
                    "user_id": user_id,
                    "ttl_epoch": ttl_epoch,
                },
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                logger.warning(
                    "dynamo_client.session_already_exists",
                    extra={"call_sid": call_sid},
                )
            else:
                logger.error(
                    "dynamo_client.create_session_failed",
                    extra={"call_sid": call_sid, "error": str(e)},
                )
                raise

    # -------------------------------------------------------------------
    # Step 138 — append_risk_score
    # -------------------------------------------------------------------

    async def append_risk_score(
        self,
        call_sid: str,
        timestamp: float,
        score: float,
        signals: dict,
    ) -> None:
        """
        Append a risk score entry to the session's risk_score_timeline.

        Also updates peak_risk_score if the new score is higher.

        Args:
            call_sid: Twilio call SID.
            timestamp: Unix epoch timestamp of the score.
            score: Composite risk score (0–100).
            signals: Contributing signal breakdown
                     (e.g., {"acoustic": 80.5, "context": 42.0}).
        """
        entry = {
            "timestamp": _to_decimal(timestamp),
            "score": _to_decimal(score),
            "signals": {k: _to_decimal(v) if isinstance(v, (int, float)) else v
                        for k, v in signals.items()},
        }

        def _update():
            table = self._get_table()
            table.update_item(
                Key={"call_sid": call_sid},
                UpdateExpression=(
                    "SET risk_score_timeline = list_append("
                    "  if_not_exists(risk_score_timeline, :empty_list), :entry"
                    "), "
                    "peak_risk_score = if_not_exists(peak_risk_score, :zero)"
                ),
                ConditionExpression="attribute_exists(call_sid)",
                ExpressionAttributeValues={
                    ":entry": [entry],
                    ":empty_list": [],
                    ":zero": _to_decimal(0),
                },
            )

            # Update peak score in a separate conditional update
            # (DynamoDB doesn't support if-greater-than in a single expression)
            try:
                table.update_item(
                    Key={"call_sid": call_sid},
                    UpdateExpression="SET peak_risk_score = :score",
                    ConditionExpression="peak_risk_score < :score",
                    ExpressionAttributeValues={
                        ":score": _to_decimal(score),
                    },
                )
            except ClientError as e:
                if e.response["Error"]["Code"] != "ConditionalCheckFailedException":
                    raise
                # Score is not higher than current peak — that's fine

        try:
            await asyncio.to_thread(_update)
            logger.debug(
                "dynamo_client.risk_score_appended",
                extra={
                    "call_sid": call_sid,
                    "timestamp": round(timestamp, 3),
                    "score": round(score, 2),
                },
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                logger.warning(
                    "dynamo_client.session_not_found_for_append",
                    extra={"call_sid": call_sid},
                )
            else:
                logger.error(
                    "dynamo_client.append_risk_score_failed",
                    extra={"call_sid": call_sid, "error": str(e)},
                )
                raise

    # -------------------------------------------------------------------
    # Step 139 — finalize_session
    # -------------------------------------------------------------------

    async def finalize_session(
        self,
        call_sid: str,
        final_verdict: str,
    ) -> None:
        """
        Finalize a call session when the call ends.

        Sets end_time, computes duration_seconds, and writes the
        final_verdict ("safe", "suspicious", or "high_risk").

        Args:
            call_sid: Twilio call SID.
            final_verdict: Final risk assessment — one of
                           "safe", "suspicious", "high_risk".
        """
        if final_verdict not in ("safe", "suspicious", "high_risk"):
            raise ValueError(
                f"Invalid final_verdict: '{final_verdict}'. "
                f"Must be one of: safe, suspicious, high_risk"
            )

        now = time.time()

        def _finalize():
            table = self._get_table()
            # First, get the start_time to compute duration
            response = table.get_item(
                Key={"call_sid": call_sid},
                ProjectionExpression="start_time",
            )
            item = response.get("Item")
            if item is None:
                logger.warning(
                    "dynamo_client.session_not_found_for_finalize",
                    extra={"call_sid": call_sid},
                )
                return

            start_time = float(item["start_time"])
            duration = now - start_time

            table.update_item(
                Key={"call_sid": call_sid},
                UpdateExpression=(
                    "SET end_time = :end_time, "
                    "duration_seconds = :duration, "
                    "final_verdict = :verdict"
                ),
                ExpressionAttributeValues={
                    ":end_time": _to_decimal(now),
                    ":duration": _to_decimal(duration),
                    ":verdict": final_verdict,
                },
            )

        try:
            await asyncio.to_thread(_finalize)
            logger.info(
                "dynamo_client.session_finalized",
                extra={
                    "call_sid": call_sid,
                    "final_verdict": final_verdict,
                },
            )
        except ClientError as e:
            logger.error(
                "dynamo_client.finalize_session_failed",
                extra={"call_sid": call_sid, "error": str(e)},
            )
            raise

    # -------------------------------------------------------------------
    # Step 140 — get_session
    # -------------------------------------------------------------------

    async def get_session(self, call_sid: str) -> Optional[dict]:
        """
        Retrieve the full session record for a call.

        Args:
            call_sid: Twilio call SID.

        Returns:
            Full session dict with all attributes, or None if not found.
            All Decimal values are converted to Python float/int.
        """
        def _get():
            table = self._get_table()
            response = table.get_item(Key={"call_sid": call_sid})
            return response.get("Item")

        try:
            item = await asyncio.to_thread(_get)
            if item is None:
                return None
            return _deserialize_item(item)
        except ClientError as e:
            logger.error(
                "dynamo_client.get_session_failed",
                extra={"call_sid": call_sid, "error": str(e)},
            )
            raise

    # -------------------------------------------------------------------
    # Step 141 — list_sessions
    # -------------------------------------------------------------------

    async def list_sessions(
        self,
        user_id: str,
        limit: int = 20,
        last_evaluated_key: Optional[dict] = None,
    ) -> dict:
        """
        List recent call sessions for a user, sorted by start_time descending.

        Uses the `user_id-start_time-index` GSI. Returns paginated results.

        Args:
            user_id: The enrolled VoiceGuard user ID.
            limit: Maximum number of sessions to return (default: 20).
            last_evaluated_key: Pagination token from a previous call.

        Returns:
            Dict with:
                - "sessions": List of session dicts (Decimals converted).
                - "last_evaluated_key": Pagination token for next page,
                  or None if no more results.
        """
        def _query():
            table = self._get_table()
            query_kwargs = {
                "IndexName": "user_id-start_time-index",
                "KeyConditionExpression": "user_id = :uid",
                "ExpressionAttributeValues": {":uid": user_id},
                "ScanIndexForward": False,  # Descending order (newest first)
                "Limit": limit,
                # Return a subset of attributes for list view efficiency
                "ProjectionExpression": (
                    "call_sid, caller_number, user_id, start_time, "
                    "end_time, duration_seconds, peak_risk_score, final_verdict"
                ),
            }

            if last_evaluated_key is not None:
                query_kwargs["ExclusiveStartKey"] = last_evaluated_key

            response = table.query(**query_kwargs)
            return response

        try:
            response = await asyncio.to_thread(_query)
            sessions = [_deserialize_item(item) for item in response.get("Items", [])]
            next_key = response.get("LastEvaluatedKey")

            logger.debug(
                "dynamo_client.sessions_listed",
                extra={
                    "user_id": user_id,
                    "count": len(sessions),
                    "has_more": next_key is not None,
                },
            )

            return {
                "sessions": sessions,
                "last_evaluated_key": next_key,
            }
        except ClientError as e:
            logger.error(
                "dynamo_client.list_sessions_failed",
                extra={"user_id": user_id, "error": str(e)},
            )
            raise

    # -------------------------------------------------------------------
    # Alert tracking helper (used by AlertDispatcher)
    # -------------------------------------------------------------------

    async def append_alert(
        self,
        call_sid: str,
        level: str,
        channel: str,
    ) -> None:
        """
        Append an alert record to the session's alerts_sent list.

        Args:
            call_sid: Twilio call SID.
            level: Alert level ("medium" or "high").
            channel: Delivery channel ("fcm", "sms", "sns").
        """
        entry = {
            "timestamp": _to_decimal(time.time()),
            "level": level,
            "channel": channel,
        }

        def _update():
            table = self._get_table()
            table.update_item(
                Key={"call_sid": call_sid},
                UpdateExpression=(
                    "SET alerts_sent = list_append("
                    "  if_not_exists(alerts_sent, :empty_list), :entry"
                    ")"
                ),
                ExpressionAttributeValues={
                    ":entry": [entry],
                    ":empty_list": [],
                },
            )

        try:
            await asyncio.to_thread(_update)
            logger.debug(
                "dynamo_client.alert_appended",
                extra={
                    "call_sid": call_sid,
                    "level": level,
                    "channel": channel,
                },
            )
        except ClientError as e:
            logger.error(
                "dynamo_client.append_alert_failed",
                extra={"call_sid": call_sid, "error": str(e)},
            )
            raise

    # -------------------------------------------------------------------
    # Health check
    # -------------------------------------------------------------------

    async def ping(self) -> bool:
        """
        Health check: verify the DynamoDB table is accessible.

        Returns:
            True if the table exists and is ACTIVE, False otherwise.
        """
        def _describe():
            table = self._get_table()
            table.load()  # Fetches table metadata from AWS
            return table.table_status

        try:
            status = await asyncio.to_thread(_describe)
            return status == "ACTIVE"
        except Exception as e:
            logger.warning(
                "dynamo_client.ping_failed",
                extra={"error": str(e)},
            )
            return False


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

dynamo_client = DynamoClient()
