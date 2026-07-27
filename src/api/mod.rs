pub mod error;
pub(crate) mod exec;
pub mod health;
pub mod models;
pub mod stream;
pub mod transcriptions;

/// Join chunk transcripts with single spaces, skipping chunks for which the
/// engine returned an empty string (so they do not produce double spaces).
pub(crate) fn join_parts(parts: &[String]) -> String {
    parts
        .iter()
        .filter(|s| !s.is_empty())
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::join_parts;

    fn owned(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn joins_with_single_spaces() {
        assert_eq!(join_parts(&owned(&["a", "b", "c"])), "a b c");
    }

    #[test]
    fn skips_empty_parts() {
        assert_eq!(join_parts(&owned(&["a", "", "b", ""])), "a b");
    }

    #[test]
    fn all_empty_yields_empty() {
        assert_eq!(join_parts(&owned(&["", ""])), "");
        assert_eq!(join_parts(&[]), "");
    }

    #[test]
    fn trims_surrounding_whitespace() {
        assert_eq!(join_parts(&owned(&[" a ", "b"])), "a  b");
        assert_eq!(join_parts(&owned(&[" a "])), "a");
    }
}
