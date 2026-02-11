# =============================
# Утилиты для облака точек (GPU pipeline)
# =============================
import numpy as np

# Нормализованный порог кластера (0..1) — доступен извне для передачи в шейдер
cluster_threshold_t = 0.5


# Pre-allocated histogram buffers (reused across calls at ~30 Hz)
_hist_counts = None
_hist_bins = None
_hist_n_bins = 0


def _find_cluster_threshold(depths, d_min, d_max):
    """
    Находит наибольший разрыв в распределении глубин.
    Возвращает абсолютный порог и нормализованный (0..1) в диапазоне [d_min, d_max].
    """
    global _hist_counts, _hist_bins, _hist_n_bins

    n = len(depths)
    if n < 2:
        return (d_min + d_max) * 0.5, 0.5

    n_bins = min(128, max(16, n // 50))

    # Reuse histogram output buffers when bin count is stable
    if n_bins != _hist_n_bins:
        _hist_counts = np.empty(n_bins, dtype=np.intp)
        _hist_bins = np.empty(n_bins + 1, dtype=np.float64)
        _hist_n_bins = n_bins

    counts, bin_edges = np.histogram(depths, bins=n_bins,
                                     range=(d_min, d_max))

    search_start = max(1, n_bins // 8)
    search_end = min(n_bins - 1, n_bins - n_bins // 8)
    if search_start >= search_end:
        search_start, search_end = 1, n_bins - 1

    # Use a view — argmin on a contiguous slice is fine without copy
    gap_idx = search_start + int(
        np.argmin(counts[search_start:search_end]))
    threshold = float(bin_edges[gap_idx + 1])

    rng = max(d_max - d_min, 0.001)
    threshold_norm = (threshold - d_min) / rng
    threshold_norm = max(0.0, min(1.0, threshold_norm))
    return threshold, threshold_norm

