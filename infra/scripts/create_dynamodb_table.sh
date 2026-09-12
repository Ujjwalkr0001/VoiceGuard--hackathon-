#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# VoiceGuard — Create DynamoDB Table for Call Sessions (Step 135)
#
# Creates the `call_sessions` DynamoDB table with:
#   - Partition key: call_sid (String)
#   - GSI: user_id-start_time-index (for listing sessions by user)
#   - TTL enabled on `ttl` attribute (auto-deletes records after 30 days)
#
# Prerequisites:
#   - AWS CLI v2 configured with credentials (`aws configure`)
#   - IAM permissions: dynamodb:CreateTable, dynamodb:UpdateTimeToLive,
#                      dynamodb:DescribeTable, dynamodb:DescribeTimeToLive
#
# Usage:
#   chmod +x infra/scripts/create_dynamodb_table.sh
#   ./infra/scripts/create_dynamodb_table.sh
#
# For DynamoDB Local (development):
#   ./infra/scripts/create_dynamodb_table.sh --local
# ---------------------------------------------------------------------------

set -euo pipefail

TABLE_NAME="${DYNAMODB_TABLE_NAME:-call_sessions}"
REGION="${AWS_REGION:-ap-south-1}"
ENDPOINT_OPTS=""

# Support DynamoDB Local for development
if [[ "${1:-}" == "--local" ]]; then
    ENDPOINT_OPTS="--endpoint-url http://localhost:8000"
    echo "🔧 Using DynamoDB Local at http://localhost:8000"
fi

echo "Creating DynamoDB table '${TABLE_NAME}' in region '${REGION}'..."
echo ""

# ---------------------------------------------------------------------------
# Table Schema
#
# Primary Key:
#   call_sid (String) — Partition key, unique per call
#
# Global Secondary Index:
#   user_id-start_time-index — Enables querying all calls for a given user,
#   sorted by start_time descending (for the call history API)
#
# Attributes stored (not all declared in AttributeDefinitions because
# DynamoDB is schemaless — only key attributes need declaration):
#   - call_sid        (S)  PK
#   - caller_number   (S)
#   - user_id         (S)  GSI PK
#   - start_time      (N)  Unix epoch, GSI SK
#   - end_time        (N)  Unix epoch
#   - duration_seconds(N)
#   - risk_score_timeline (L)  List of Maps: [{timestamp, score, signals}]
#   - peak_risk_score (N)
#   - final_verdict   (S)  "safe" | "suspicious" | "high_risk"
#   - alerts_sent     (L)  List of Maps: [{timestamp, level, channel}]
#   - ttl             (N)  Unix epoch = start_time + 30 days (for TTL)
# ---------------------------------------------------------------------------

aws dynamodb create-table \
    --table-name "${TABLE_NAME}" \
    --attribute-definitions \
        AttributeName=call_sid,AttributeType=S \
        AttributeName=user_id,AttributeType=S \
        AttributeName=start_time,AttributeType=N \
    --key-schema \
        AttributeName=call_sid,KeyType=HASH \
    --global-secondary-indexes \
        '[
            {
                "IndexName": "user_id-start_time-index",
                "KeySchema": [
                    {"AttributeName": "user_id", "KeyType": "HASH"},
                    {"AttributeName": "start_time", "KeyType": "RANGE"}
                ],
                "Projection": {
                    "ProjectionType": "ALL"
                },
                "ProvisionedThroughput": {
                    "ReadCapacityUnits": 5,
                    "WriteCapacityUnits": 5
                }
            }
        ]' \
    --provisioned-throughput \
        ReadCapacityUnits=5,WriteCapacityUnits=5 \
    --region "${REGION}" \
    ${ENDPOINT_OPTS}

echo ""
echo "⏳ Waiting for table to become ACTIVE..."

aws dynamodb wait table-exists \
    --table-name "${TABLE_NAME}" \
    --region "${REGION}" \
    ${ENDPOINT_OPTS}

echo "✅ Table '${TABLE_NAME}' is ACTIVE."
echo ""

# ---------------------------------------------------------------------------
# Enable TTL on the `ttl` attribute
#
# DynamoDB will automatically delete items whose `ttl` attribute value
# (Unix epoch seconds) is older than the current time. Items are typically
# deleted within 48 hours of expiration.
# ---------------------------------------------------------------------------

echo "Enabling TTL on attribute 'ttl'..."

aws dynamodb update-time-to-live \
    --table-name "${TABLE_NAME}" \
    --time-to-live-specification \
        "Enabled=true, AttributeName=ttl" \
    --region "${REGION}" \
    ${ENDPOINT_OPTS}

echo "✅ TTL enabled on attribute 'ttl'."
echo ""

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

echo "📋 Table description:"
aws dynamodb describe-table \
    --table-name "${TABLE_NAME}" \
    --region "${REGION}" \
    --query 'Table.{TableName:TableName, Status:TableStatus, KeySchema:KeySchema, GSIs:GlobalSecondaryIndexes[*].{IndexName:IndexName, KeySchema:KeySchema, Status:IndexStatus}, ItemCount:ItemCount}' \
    --output table \
    ${ENDPOINT_OPTS}

echo ""
echo "📋 TTL status:"
aws dynamodb describe-time-to-live \
    --table-name "${TABLE_NAME}" \
    --region "${REGION}" \
    --output table \
    ${ENDPOINT_OPTS}

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ DynamoDB table '${TABLE_NAME}' is ready!"
echo ""
echo "   Table name:  ${TABLE_NAME}"
echo "   Region:      ${REGION}"
echo "   Primary key: call_sid (String)"
echo "   GSI:         user_id-start_time-index"
echo "   TTL:         Enabled on 'ttl' attribute (30-day auto-delete)"
echo ""
echo "   Ensure your .env has:"
echo "     DYNAMODB_TABLE_NAME=${TABLE_NAME}"
echo "     AWS_REGION=${REGION}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
