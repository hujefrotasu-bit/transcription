CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY,
    transcript TEXT NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'processing',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS topics (
    id UUID PRIMARY KEY,
    conversation_id UUID NOT NULL,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_topics_conversation
        FOREIGN KEY (conversation_id)
        REFERENCES conversations(id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_topics_conversation_id
ON topics(conversation_id);