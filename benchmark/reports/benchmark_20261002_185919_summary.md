# Benchmark Evaluation & Tier Analysis Report

**Date & Time:** October 02, 2026, 18:59:19  
**Evaluation Engine:** Claude Fable 5.1 Real Audit (LLM-as-a-Judge)  
**Transcription Engine:** gemini-3.5-transcribe  
**Meeting Minutes Engine:** gemini-3.5-flash-lite  
**Total Files Evaluated:** 4 Audio Files  
**Overall Pass Rate:** 75.0% (3 / 4)  
**Average Score:** 91.8 / 100  
**Average Latency:** 70.6s per file  

---

## 1. Multi-Tier Performance Breakdown

| Tier | Files | Pass Rate | Avg Score | Avg Latency | Total Attendees | Actions | Decisions | Discussions | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **🟢 Easy** | 2 | **100.0%** | **100.0 / 100** | 54.5s | 2 | 1 | 1 | 2 | **PASS** |
| **🟡 Medium** | 1 | **100.0%** | **100.0 / 100** | 55.7s | 4 | 1 | 1 | 2 | **PASS** |
| **🔴 Hard** | 1 | **0.0%** | **67.0 / 100** | 117.5s | 5 | 3 | 4 | 4 | **FAIL** |

---

## 2. AI Model Consistency & Bias Diagnostics

> [!WARNING]
> **Complexity Sensitivity Detected (-33.0% drop):**  
> The model achieves a perfect **100.0% score** on Easy and Medium single/few-speaker meetings, but drops significantly to **67.0%** on Hard multi-speaker meetings.  
> **Key Driver:** Rapid multi-speaker cross-talk and ambiguous mentions of absent participants lead to missed attribution and disputed decisions.

> [!NOTE]
> **Latency Scaling (2.2x multiplier):**  
> Hard meetings average **117.5s** total processing time vs **54.5s** on Easy meetings.

---

## 3. Itemized Results

| File | Type | Tier | Status | Audit Score | Latency | Attendees | Actions | Decisions | Errors |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `audio1.ogg` | audio | 🟢 Easy | **PASS** | 100.0% | 53.9s | 1 (Sarah) | 1 | 1 | 0 |
| `meeting-clip1.mp3` | audio | 🟢 Easy | **PASS** | 100.0% | 55.1s | 1 (Paul) | 0 | 0 | 0 |
| `business.mp3` | audio | 🟡 Medium | **PASS** | 100.0% | 55.7s | 4 (Marcus, Maya, David, Anna) | 1 | 1 | 0 |
| `Effective Meetings Simulated Exercise for Chairing & Minute Taking.mp3` | audio | 🔴 Hard | **FAIL** | 67.0% | 117.5s | 5 (Rita, Lucy Stokes, David, Marcus, Anna) | 3 | 4 | 2 |

---

## 4. Associated Raw Report Files

- **CSV Report:** [`backend/benchmark/reports/benchmark_20261002_185919.csv`](file:///c:/Users/Hujefkhan/OneDrive/Pictures/Desktop/rotasu/ROTASU/Transcript/backend/benchmark/reports/benchmark_20261002_185919.csv)
- **JSON Report:** [`backend/benchmark/reports/benchmark_20261002_185919.json`](file:///c:/Users/Hujefkhan/OneDrive/Pictures/Desktop/rotasu/ROTASU/Transcript/backend/benchmark/reports/benchmark_20261002_185919.json)
