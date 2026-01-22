## Test Plan: Inference Speed Optimization

### Scope
- **Feature**: In-memory frame processing for Anomaly Detection (detect) and Summarization.
- **Component**: `backend/cvad_detector.py` and helper functions (`key_func`, `attn_func`, `grid_func`).

### Strategy
| Type | Count | Focus |
|------|-------|-------|
| Unit | 0 | (Covered by existing test_edge_opt.py, though needs update) |
| Integration | 1 | Verify full pipeline flow without disk I/O using mocked models but real data flow. |
| E2E | 0 | (Manual verification done previously) |

### Test Cases
| ID | Case | Input | Expected | Edge? |
|----|------|-------|----------|-------|
| T1 | Detect Flow | Dummy Video (30 frames) | Pipeline completes, returns scores (24 frames), KFS receives PIL images. | No |
| T2 | Summarize Flow | Dummy Video (30 frames) | Pipeline completes, returns summary, KFS receives PIL images. | No |
| T3 | Resolution Check | Dummy Video (640x480) | `process_video` resizes to target (448px) | No |

---

## Test Report: Inference Optimization

### Summary
| Metric | Value |
|--------|-------|
| Total | 3 |
| Passed | 3 |
| Failed | 0 |
| Skipped | 0 |

### ✅ Successes
- **T1 (Detect Flow)**: Verified that `detect()` runs end-to-end with in-memory frames and correctly handles CLIP/KFS interaction using mocked models.
- **T2 (Summarize Flow)**: Verified that `summarize()` runs end-to-end with in-memory frames.
- **T3 (Resolution Check)**: Confirmed that image resizing logic in `process_video` produces correct dimensions (approx 448px short edge).
