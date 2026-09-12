"""
VoiceGuard Backend — Bootstrap the local DynamoDB table.

Creates the `call_sessions` table (and its `user_id-start_time-index` GSI)
against DynamoDB Local, so the call-history API works without AWS credentials.
This mirrors infra/scripts/create_dynamodb_table.sh but uses boto3 directly,
so no AWS CLI install is required.

Prerequisites:
    docker compose up -d dynamodb     # or: docker run -p 8001:8000 amazon/dynamodb-local

Usage:
    cd backend
    python scripts/init_local_dynamodb.py

Reads DYNAMODB_ENDPOINT_URL / DYNAMODB_TABLE_NAME / AWS_REGION from .env.
Refuses to run against real AWS (no endpoint override configured).
"""

import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError

# Allow running as `python scripts/init_local_dynamodb.py` from backend/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402


def main() -> int:
    endpoint = settings.dynamodb_endpoint_url
    if not endpoint:
        print(
            "[ERROR] DYNAMODB_ENDPOINT_URL is not set.\n"
            "   This script only bootstraps DynamoDB Local. For real AWS, use\n"
            "   infra/scripts/create_dynamodb_table.sh instead.",
        )
        return 1

    table_name = settings.dynamodb_table_name
    print(f"[*] Target: {endpoint}  table={table_name}  region={settings.aws_region}")

    client = boto3.client(
        "dynamodb",
        region_name=settings.aws_region,
        endpoint_url=endpoint,
        aws_access_key_id=settings.aws_access_key_id or "local",
        aws_secret_access_key=settings.aws_secret_access_key or "local",
    )

    try:
        client.create_table(
            TableName=table_name,
            AttributeDefinitions=[
                {"AttributeName": "call_sid", "AttributeType": "S"},
                {"AttributeName": "user_id", "AttributeType": "S"},
                {"AttributeName": "start_time", "AttributeType": "N"},
            ],
            KeySchema=[{"AttributeName": "call_sid", "KeyType": "HASH"}],
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
                }
            ],
            ProvisionedThroughput={"ReadCapacityUnits": 5, "WriteCapacityUnits": 5},
        )
        print("[..] Waiting for table to become ACTIVE...")
        client.get_waiter("table_exists").wait(TableName=table_name)
        print(f"[OK] Created table '{table_name}' with GSI 'user_id-start_time-index'.")
    except EndpointConnectionError:
        print(
            f"[ERROR] Cannot reach DynamoDB Local at {endpoint}.\n"
            "   Start it first:  docker compose up -d dynamodb",
        )
        return 1
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceInUseException":
            print(f"[OK] Table '{table_name}' already exists - nothing to do.")
        else:
            print(f"[ERROR] create_table failed: {e}")
            return 1

    # TTL is a no-op in DynamoDB Local (items are never expired), but enabling
    # it keeps the local schema faithful to production.
    try:
        client.update_time_to_live(
            TableName=table_name,
            TimeToLiveSpecification={"Enabled": True, "AttributeName": "ttl"},
        )
        print("[OK] TTL enabled on attribute 'ttl'.")
    except ClientError as e:
        print(f"[INFO] TTL not enabled ({e.response['Error']['Code']}) - harmless locally.")

    desc = client.describe_table(TableName=table_name)["Table"]
    print(
        f"\n[INFO] {desc['TableName']}: status={desc['TableStatus']} "
        f"items={desc.get('ItemCount', 0)}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
