CREATE TABLE IF NOT EXISTS meeting_minutes (
    id UUID PRIMARY KEY,
    conversation_id UUID NOT NULL UNIQUE,
    data JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_meeting_minutes_conversation
        FOREIGN KEY (conversation_id)
        REFERENCES conversations(id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_meeting_minutes_conversation_id
ON meeting_minutes(conversation_id);