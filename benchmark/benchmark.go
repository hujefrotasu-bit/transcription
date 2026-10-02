package benchmark

import (
	"context"
	"fmt"
	"log"

	"transcript/verification"
)

// MeetingBenchmark holds the benchmark performance metrics for a single meeting comparison.
type MeetingBenchmark struct {
	MeetingID             string                             `json:"meeting_id"`
	PreviousVersion       int                                `json:"previous_version"`
	CurrentVersion        int                                `json:"current_version"`
	PreviousScore         float64                            `json:"previous_score"`
	CurrentScore          float64                            `json:"current_score"`
	ScoreDelta            float64                            `json:"score_delta"`
	IsImproved            bool                               `json:"is_improved"`
	FixedErrorsCount      int                                `json:"fixed_errors_count"`
	RegressionsCount      int                                `json:"regressions_count"`
	RemainingErrorsCount  int                                `json:"remaining_errors_count"`
	Comparison            *verification.VersionComparisonResult `json:"comparison,omitempty"`
}

// BenchmarkRunner evaluates meeting minutes version improvements using Claude Fable.
type BenchmarkRunner struct {
	Verifier *verification.Verifier
}

// NewBenchmarkRunner initializes a new BenchmarkRunner.
func NewBenchmarkRunner(verifier *verification.Verifier) *BenchmarkRunner {
	return &BenchmarkRunner{Verifier: verifier}
}

// BenchmarkVersions compares an initial/previous meeting minutes against an improved candidate.
func (r *BenchmarkRunner) BenchmarkVersions(ctx context.Context, meetingID, transcript, prevMinutesJSON, newMinutesJSON string) (*MeetingBenchmark, error) {
	log.Printf("[Benchmark] Running version comparison evaluation for meeting: %s", meetingID)

	// 1. Audit baseline previous version
	prevAudit, err := r.Verifier.AuditMeetingMinutes(ctx, transcript, prevMinutesJSON)
	if err != nil {
		return nil, fmt.Errorf("baseline audit failed for %s: %w", meetingID, err)
	}

	// 2. Run comparative analysis
	compResult, err := r.Verifier.CompareVersions(ctx, transcript, prevMinutesJSON, newMinutesJSON, prevAudit.WhatIsWrong, prevAudit.Score)
	if err != nil {
		return nil, fmt.Errorf("comparison failed for %s: %w", meetingID, err)
	}

	// 3. Audit current version for remaining error count
	currAudit, err := r.Verifier.AuditMeetingMinutes(ctx, transcript, newMinutesJSON)
	remainingCount := 0
	if err == nil && currAudit != nil {
		remainingCount = len(currAudit.WhatIsWrong)
		compResult.PopulateFromAudit(currAudit)
	}

	bm := &MeetingBenchmark{
		MeetingID:            meetingID,
		PreviousVersion:      1,
		CurrentVersion:       2,
		PreviousScore:        prevAudit.Score,
		CurrentScore:         compResult.CurrentScore,
		ScoreDelta:           compResult.ScoreDelta,
		IsImproved:           compResult.IsImproved,
		FixedErrorsCount:     len(compResult.FixedErrors),
		RegressionsCount:     len(compResult.Regressions),
		RemainingErrorsCount: remainingCount,
		Comparison:           compResult,
	}

	return bm, nil
}
