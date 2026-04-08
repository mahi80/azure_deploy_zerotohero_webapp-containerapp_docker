-- migrations/init.sql
-- Runs automatically when the PostgreSQL container starts for the first time.
-- SQLAlchemy's Base.metadata.create_all() handles table creation at app startup,
-- so this file handles database-level setup only.

-- ── Extensions ────────────────────────────────────────────────────────────────
-- pgcrypto: used for UUID generation if needed in future migrations
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ── Database comment ──────────────────────────────────────────────────────────
COMMENT ON DATABASE cc_underwriting IS
  'Credit card underwriting API database — users and prediction audit log';

-- ── Grant full privileges to the API user ────────────────────────────────────
-- The ccapi user is created by docker-compose env vars (POSTGRES_USER).
-- This ensures all future tables created by SQLAlchemy are accessible.
GRANT ALL PRIVILEGES ON DATABASE cc_underwriting TO ccapi;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ccapi;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO ccapi;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO ccapi;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO ccapi;
