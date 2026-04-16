#!/usr/bin/env bash
set -euo pipefail

# Day 2 Demo: Apply schema change (migration 004 — add is_favorited) with zero downtime.
# Usage: ./scripts/demo-schema-change.sh <namespace>
# Example: ./scripts/demo-schema-change.sh samosachaat-prod

NAMESPACE="${1:?Usage: demo-schema-change.sh <namespace>}"

echo "=== samosaChaat Day 2: Schema Change Demo ==="

echo ""
echo "Step 1: Current migration state"
kubectl exec -n "$NAMESPACE" deploy/chat-api -- alembic current 2>/dev/null || \
  echo "(Could not connect — ensure chat-api pod is running)"

echo ""
echo "Step 2: Show the migration file"
echo "File: db/migrations/versions/004_add_favorited.py"
echo "Operation: ALTER TABLE conversations ADD COLUMN is_favorited BOOLEAN DEFAULT false NOT NULL"
echo ""
echo "Key points:"
echo "  - ADD COLUMN with DEFAULT is non-blocking in PostgreSQL 11+"
echo "  - No table lock, no downtime, existing rows get default value instantly"
echo "  - Old pods (without the code change) simply ignore the new column"
echo "  - New pods (with updated SQLAlchemy model) can use it immediately"

echo "Step 3: Apply migration via Helm upgrade"
echo "The db-migrate-job.yaml Helm hook runs 'alembic upgrade head' before new pods start."
echo ""
echo "Running: helm upgrade samosachaat helm/samosachaat -n $NAMESPACE --reuse-values"
helm upgrade samosachaat helm/samosachaat -n "$NAMESPACE" --reuse-values

echo ""
echo "Step 4: Verify migration applied"
kubectl exec -n "$NAMESPACE" deploy/chat-api -- alembic current

echo ""
echo "Step 5: Verify column exists in database"
kubectl exec -n "$NAMESPACE" deploy/chat-api -- python -c "
from sqlalchemy import inspect, create_engine
import os
url = os.environ.get('DATABASE_URL', '').replace('+asyncpg', '')
if not url:
    print('DATABASE_URL not set')
    exit(1)
engine = create_engine(url)
cols = [c['name'] for c in inspect(engine).get_columns('conversations')]
print(f'Columns: {cols}')
assert 'is_favorited' in cols, 'FAIL: is_favorited not found!'
print('SUCCESS: is_favorited column present and migration is complete.')
"
