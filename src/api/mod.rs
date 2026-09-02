pub mod error;
pub(crate) mod exec;
pub mod health;
pub mod models;
pub mod stream;
pub mod transcriptions;
pub(crate) mod verbose;

/// Join chunk transcripts with single spaces, skipping chunks for which the
/// engine returned an empty string (so they do not produce double spaces).
pub(crate) fn join_parts<'a>(parts: impl IntoIterator<Item = &'a str>) -> String {
    parts
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::join_parts;

    #[test]
    fn joins_with_single_spaces() {
        assert_eq!(join_parts(["a", "b", "c"]), "a b c");
    }

    #[test]
    fn skips_empty_parts() {
        assert_eq!(join_parts(["a", "", "b", ""]), "a b");
    }

    #[test]
    fn all_empty_yields_empty() {
        assert_eq!(join_parts(["", ""]), "");
        assert_eq!(join_parts([]), "");
    }

    #[test]
    fn trims_surrounding_whitespace() {
        assert_eq!(join_parts([" a ", "b"]), "a  b");
        assert_eq!(join_parts([" a "]), "a");
    }
}
