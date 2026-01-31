## 2024-05-23 - Video Processing Optimization
**Learning:** Video processing in OpenCV can be significantly optimized by skipping full frame decoding. Using `cap.grab()` instead of `cap.read()` for skipped frames avoids the expensive decoding step, providing ~30-40% speedup when `frame_interval > 1`.
**Action:** Always use `cap.grab()` when iterating through a video if the pixel data for that frame is not needed.

## 2024-05-23 - OpenCV Resizing Performance
**Learning:** `cv2.resize` with `cv2.INTER_AREA` is significantly slower (~23x) than `cv2.INTER_LINEAR`. While `INTER_AREA` is better for downsampling quality (anti-aliasing), the performance cost is massive for real-time or high-throughput applications.
**Action:** Evaluate the trade-off between quality and speed for resizing. For performance-critical paths, consider `INTER_LINEAR` if slight aliasing is acceptable.

## 2024-05-24 - Redundant Tokenization in Loop
**Learning:** `clip.tokenize()` is CPU-bound and expensive. In loops (like video chunk processing), pre-computing text features once and reusing them saves significant time compared to re-tokenizing and encoding every iteration.
**Action:** Always check if constant text inputs in loops can be pre-encoded outside the loop.

## 2024-05-24 - GPU Kernel Launch Overhead in Loops
**Learning:** Iterating over a list of small tensors and calling `model.encode_image` (or similar GPU ops) individually incurs massive kernel launch overhead and CPU-GPU synchronization costs.
**Action:** Batch small tensors using `torch.cat` or `torch.stack` into a single large tensor and perform the operation once. This can yield dramatic speedups even if the total FLOPs count is the same.
