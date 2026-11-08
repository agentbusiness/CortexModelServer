/// Longest window the detector reads, in characters.
const WINDOW_CHARS: usize = 400;

/// Shortest window kept when a text is cut; shorter tails join no window.
const MIN_WINDOW_CHARS: usize = 40;

/// Cuts a text into the windows the detector scores.
///
/// Breaks at line ends, and inside an over-long line at spaces, so each window
/// holds at most [`WINDOW_CHARS`] characters of whole lines or words. A text
/// too short to yield a full-sized window is scored whole; an empty text
/// yields none.
pub fn text_windows(text: &str) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut current = String::new();
    for line in text.split_inclusive('\n') {
        if current.chars().count() + line.chars().count() > WINDOW_CHARS
            && !current.trim().is_empty()
        {
            pieces.push(std::mem::take(&mut current));
        }
        if line.chars().count() > WINDOW_CHARS {
            for word in line.split_inclusive(' ') {
                if current.chars().count() + word.chars().count() > WINDOW_CHARS
                    && !current.trim().is_empty()
                {
                    pieces.push(std::mem::take(&mut current));
                }
                current.push_str(word);
            }
        } else {
            current.push_str(line);
        }
    }
    if !current.trim().is_empty() {
        pieces.push(current);
    }
    let windows: Vec<String> = pieces
        .into_iter()
        .map(|piece| piece.trim().to_string())
        .filter(|piece| piece.chars().count() >= MIN_WINDOW_CHARS)
        .collect();
    if windows.is_empty() && !text.trim().is_empty() {
        return vec![text.trim().to_string()];
    }
    windows
}
