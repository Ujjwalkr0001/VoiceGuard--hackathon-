#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# VoiceGuard — Create SNS Topic for Alert Dispatch (Step 124)
#
# Creates the `voiceguard-alerts` SNS topic in AWS.
# Run this once during initial setup.
#
# Prerequisites:
#   - AWS CLI configured with credentials (`aws configure`)
#   - IAM permissions: sns:CreateTopic, sns:SetTopicAttributes
#
# Usage:
#   chmod +x infra/scripts/create_sns_topic.sh
#   ./infra/scripts/create_sns_topic.sh
# ---------------------------------------------------------------------------

set -euo pipefail

TOPIC_NAME="voiceguard-alerts"
REGION="${AWS_REGION:-ap-south-1}"

echo "Creating SNS topic '${TOPIC_NAME}' in region '${REGION}'..."

TOPIC_ARN=$(aws sns create-topic \
    --name "${TOPIC_NAME}" \
    --region "${REGION}" \
    --output text \
    --query 'TopicArn')

echo "✅ SNS topic created successfully!"
echo "   Topic ARN: ${TOPIC_ARN}"
echo ""
echo "Add this to your .env file:"
echo "   SNS_TOPIC_ARN=${TOPIC_ARN}"

# Set a display name for SMS subscribers
aws sns set-topic-attributes \
    --topic-arn "${TOPIC_ARN}" \
    --attribute-name DisplayName \
    --attribute-value "VoiceGuard" \
    --region "${REGION}"

echo "   Display name set to 'VoiceGuard'"
