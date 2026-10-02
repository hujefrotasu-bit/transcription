ALTER TABLE meeting_minutes 
DROP COLUMN IF EXISTS model,
DROP COLUMN IF EXISTS input_tokens,
DROP COLUMN IF EXISTS output_tokens,
DROP COLUMN IF EXISTS total_tokens,
DROP COLUMN IF EXISTS estimated_cost_inr,
DROP COLUMN IF EXISTS estimated_cost_usd;

ALTER TABLE meeting_minutes_versions 
DROP COLUMN IF EXISTS model,
DROP COLUMN IF EXISTS input_tokens,
DROP COLUMN IF EXISTS output_tokens,
DROP COLUMN IF EXISTS total_tokens,
DROP COLUMN IF EXISTS estimated_cost_inr,
DROP COLUMN IF EXISTS estimated_cost_usd;
