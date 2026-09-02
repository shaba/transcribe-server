//! Energy-based VAD chunker for long-form audio.
//!
//! Splits PCM (16 kHz mono f32 in [-1, 1]) into ranges each at most
//! `max_sec` long, preferring to cut inside silence so that words are
//! not split across chunk boundaries.

use std::ops::Range;

use crate::audio::TARGET_SR;

/// Seconds to look back from the end of a full window for a silent frame.
const SEARCH_BACK_SEC: f32 = 10.0;

/// Cap on the search-back window as a fraction of the chunk window, so a short
/// window still cuts near its end rather than anywhere inside it.
const SEARCH_BACK_FRACTION: f32 = 0.4;

/// Frame length used for energy estimation, in milliseconds.
const FRAME_MS: usize = 30;

/// Frame length in samples; at least one sample so the scan always advances.
fn frame_len() -> usize {
    (TARGET_SR * FRAME_MS / 1000).max(1)
}

/// Window length in samples, clamped the way the chunker clamps it: a
/// `max_sec` shorter than one frame is raised to one frame, so every range is
/// non-empty and the loop always makes progress.
///
/// Exposed so a caller buffering audio for the chunker sizes its buffer from
/// the same number, rather than re-deriving it and drifting.
pub fn window_samples(max_sec: f32) -> usize {
    ((max_sec * TARGET_SR as f32) as usize).max(frame_len())
}

/// End of the first range [`chunk_ranges`] would produce for `pcm`, i.e. where
/// to cut a buffer that has filled one window. `pcm.len()` when the whole
/// buffer fits in one window.
///
/// Frame energy = mean(|x|) over 30 ms frames; "silence" = energy below
/// `vad_threshold`; the minimum-energy silent frame in the search-back window
/// is the cut point (cut at frame center). No silence found means a hard cut
/// at the window edge.
pub fn first_cut(pcm: &[f32], max_sec: f32, vad_threshold: f32) -> usize {
    let max_samples = window_samples(max_sec);
    if pcm.len() <= max_samples {
        return pcm.len();
    }
    let frame_len = frame_len();
    // Scaled to the window, not fixed: a model whose own limit is shorter than
    // SEARCH_BACK_SEC would otherwise have its whole window searched, making
    // the quietest frame anywhere in it the cut -- which turns a window that
    // opens with a pause into a chunk of a few frames.
    let search_back =
        ((SEARCH_BACK_SEC.min(max_sec * SEARCH_BACK_FRACTION)) * TARGET_SR as f32) as usize;
    let search_start = max_samples.saturating_sub(search_back);

    let mut best: Option<(f32, usize)> = None;
    let mut pos = search_start;
    while pos + frame_len <= max_samples {
        let energy = pcm[pos..pos + frame_len]
            .iter()
            .map(|x| x.abs())
            .sum::<f32>()
            / frame_len as f32;
        if energy < vad_threshold && best.is_none_or(|(e, _)| energy < e) {
            best = Some((energy, pos + frame_len / 2));
        }
        pos += frame_len;
    }
    match best {
        // A cut at zero would make no progress, so the window edge wins.
        Some((_, center)) if center > 0 => center,
        _ => max_samples,
    }
}

/// Split pcm into ranges each <= `max_sec` long, cutting at silence where
/// there is any (see [`first_cut`]). Ranges cover pcm without gaps; empty pcm
/// yields no ranges.
pub fn chunk_ranges(pcm: &[f32], max_sec: f32, vad_threshold: f32) -> Vec<Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = 0usize;
    while start < pcm.len() {
        let cut = start + first_cut(&pcm[start..], max_sec, vad_threshold);
        ranges.push(start..cut);
        start = cut;
    }
    ranges
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: usize = 16000;
    const THRESHOLD: f32 = 0.01;

    /// Sine at 440 Hz with amplitude 0.5: mean(|x|) ~= 0.318, well above THRESHOLD.
    fn sine(seconds: f32) -> Vec<f32> {
        let n = (seconds * SR as f32) as usize;
        (0..n)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SR as f32).sin())
            .collect()
    }

    fn silence(seconds: f32) -> Vec<f32> {
        vec![0.0; (seconds * SR as f32) as usize]
    }

    fn frame_energy(pcm: &[f32], center: usize) -> f32 {
        let frame_len = SR * FRAME_MS / 1000;
        let start = center.saturating_sub(frame_len / 2);
        let end = (start + frame_len).min(pcm.len());
        pcm[start..end].iter().map(|x| x.abs()).sum::<f32>() / (end - start) as f32
    }

    fn assert_gapless(ranges: &[Range<usize>], total: usize) {
        assert!(!ranges.is_empty(), "expected at least one range");
        assert_eq!(ranges[0].start, 0, "first range must start at 0");
        assert_eq!(
            ranges.last().unwrap().end,
            total,
            "last range must end at pcm.len()"
        );
        for pair in ranges.windows(2) {
            assert_eq!(
                pair[0].end, pair[1].start,
                "ranges must be gapless and ordered"
            );
        }
        for r in ranges {
            assert!(r.start < r.end, "each range must be non-empty: {r:?}");
        }
    }

    #[test]
    fn short_input_single_full_range() {
        let pcm = sine(1.0);
        let ranges = chunk_ranges(&pcm, 25.0, THRESHOLD);
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0], 0..16000);
    }

    #[test]
    fn cuts_inside_silence() {
        // 20 s sine + 1 s silence + 20 s sine + 1 s silence + 18 s sine = 60 s.
        let mut pcm = sine(20.0);
        pcm.extend(silence(1.0));
        pcm.extend(sine(20.0));
        pcm.extend(silence(1.0));
        pcm.extend(sine(18.0));
        let max_sec = 25.0;
        let max_samples = (max_sec * SR as f32) as usize;

        let ranges = chunk_ranges(&pcm, max_sec, THRESHOLD);
        assert_gapless(&ranges, pcm.len());
        assert_eq!(ranges.len(), 3, "expected 3 ranges, got {ranges:?}");
        for r in &ranges {
            assert!(r.len() <= max_samples, "range longer than max: {r:?}");
        }
        // Both cuts must land inside silence.
        for cut in [ranges[0].end, ranges[1].end] {
            assert!(
                frame_energy(&pcm, cut) < THRESHOLD,
                "cut at {cut} is not inside silence"
            );
        }
    }

    #[test]
    fn hard_cut_without_silence() {
        let pcm = sine(60.0);
        let max_sec = 25.0;
        let max_samples = (max_sec * SR as f32) as usize;

        let ranges = chunk_ranges(&pcm, max_sec, THRESHOLD);
        assert_gapless(&ranges, pcm.len());
        assert_eq!(ranges.len(), 3);
        assert_eq!(ranges[0], 0..max_samples);
        assert_eq!(ranges[1], max_samples..2 * max_samples);
        assert_eq!(ranges[2], 2 * max_samples..pcm.len());
    }

    /// A window shorter than the fixed search-back would otherwise have the
    /// whole window searched, so a pause near its start became the cut and the
    /// chunk came out a handful of frames long. The cut has to stay near the
    /// window end whatever the window size.
    #[test]
    fn a_short_window_still_cuts_near_its_end() {
        let mut pcm = silence(0.3); // a pause right after the window opens
        pcm.extend(sine(9.7));
        let max_sec = 4.0;
        let ranges = chunk_ranges(&pcm, max_sec, THRESHOLD);
        let first = ranges[0].len() as f32 / SR as f32;
        assert!(
            first > max_sec * (1.0 - SEARCH_BACK_FRACTION) - 0.1,
            "first chunk of {first} s is far short of the {max_sec} s window"
        );
        assert!(first <= max_sec, "chunk longer than the window: {first} s");
    }

    #[test]
    fn empty_pcm_returns_empty_vec() {
        assert!(chunk_ranges(&[], 25.0, THRESHOLD).is_empty());
    }

    #[test]
    fn shorter_than_one_frame_single_range() {
        let pcm = vec![0.1f32; 100]; // < 480 samples (one 30 ms frame at 16k)
        let ranges = chunk_ranges(&pcm, 25.0, THRESHOLD);
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0], 0..100);
    }

    #[test]
    fn non_positive_max_sec_clamped_to_one_frame() {
        let pcm = sine(1.0);
        let ranges = chunk_ranges(&pcm, 0.0, THRESHOLD);
        assert_gapless(&ranges, pcm.len());
        // Clamped window is one frame; pure sine has no silence, so hard cuts.
        assert!(ranges.iter().all(|r| r.len() <= SR * FRAME_MS / 1000));
    }
}
