package database

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

func Connect() (*pgxpool.Pool, error) {
	connString := fmt.Sprintf(
		"postgres://%s:%s@%s:%s/%s?sslmode=disable",
		os.Getenv("DB_USER"),
		os.Getenv("DB_PASSWORD"),
		os.Getenv("DB_HOST"),
		os.Getenv("DB_PORT"),
		os.Getenv("DB_NAME"),
	)

	config, err := pgxpool.ParseConfig(connString)
	if err != nil {
		return nil, err
	}

	config.MaxConns = 10
	config.MinConns = 2
	config.MaxConnLifetime = time.Hour

	pool, err := pgxpool.NewWithConfig(context.Background(), config)
	if err != nil {
		return nil, err
	}

	// Verify that PostgreSQL is reachable.
	if err := pool.Ping(context.Background()); err != nil {
		pool.Close()
		return nil, err
	}

	// Ensure meeting_minutes_versions table exists
	if err := Migrate(context.Background(), pool); err != nil {
		fmt.Printf("Warning: failed to ensure meeting_minutes_versions table: %v\n", err)
	}

	return pool, nil
}

// Migrate ensures essential tables such as meeting_minutes_versions exist.
func Migrate(ctx context.Context, pool *pgxpool.Pool) error {
	query := `
	CREATE TABLE IF NOT EXISTS meeting_minutes_versions (
		id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
		conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
		version_number INT NOT NULL,
		meeting_minutes_data JSONB NOT NULL,
		score NUMERIC(5,2) NOT NULL DEFAULT 0,
		score_breakdown JSONB NOT NULL DEFAULT '{}',
		what_is_right JSONB NOT NULL DEFAULT '[]',
		errors JSONB NOT NULL DEFAULT '[]',
		missing_information JSONB NOT NULL DEFAULT '[]',
		improvement_summary TEXT NOT NULL DEFAULT '',
		fixed_errors JSONB NOT NULL DEFAULT '[]',
		still_unfixed JSONB NOT NULL DEFAULT '[]',
		regressions JSONB NOT NULL DEFAULT '[]',
		score_delta NUMERIC(5,2) NOT NULL DEFAULT 0,
		is_improved BOOLEAN NOT NULL DEFAULT FALSE,
		created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

		CONSTRAINT uq_conversation_meeting_minutes_version UNIQUE (conversation_id, version_number)
	);

	CREATE INDEX IF NOT EXISTS idx_meeting_minutes_versions_conversation_id 
	ON meeting_minutes_versions(conversation_id);

	-- Add token and cost columns if not present
	ALTER TABLE meeting_minutes 
	ADD COLUMN IF NOT EXISTS model TEXT DEFAULT 'gemini-3.5-flash-lite',
	ADD COLUMN IF NOT EXISTS input_tokens INT DEFAULT 0,
	ADD COLUMN IF NOT EXISTS output_tokens INT DEFAULT 0,
	ADD COLUMN IF NOT EXISTS total_tokens INT DEFAULT 0,
	ADD COLUMN IF NOT EXISTS estimated_cost_inr NUMERIC(10,4) DEFAULT 0,
	ADD COLUMN IF NOT EXISTS estimated_cost_usd NUMERIC(10,6) DEFAULT 0;

	ALTER TABLE meeting_minutes_versions 
	ADD COLUMN IF NOT EXISTS model TEXT DEFAULT 'gemini-3.5-flash-lite',
	ADD COLUMN IF NOT EXISTS input_tokens INT DEFAULT 0,
	ADD COLUMN IF NOT EXISTS output_tokens INT DEFAULT 0,
	ADD COLUMN IF NOT EXISTS total_tokens INT DEFAULT 0,
	ADD COLUMN IF NOT EXISTS estimated_cost_inr NUMERIC(10,4) DEFAULT 0,
	ADD COLUMN IF NOT EXISTS estimated_cost_usd NUMERIC(10,6) DEFAULT 0;
	`
	_, err := pool.Exec(ctx, query)
	return err
}
