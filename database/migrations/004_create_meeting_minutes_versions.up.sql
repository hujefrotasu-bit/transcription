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
