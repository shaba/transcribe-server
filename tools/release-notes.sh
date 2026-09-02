#!/bin/sh
# Release notes for a tag, from NEWS.md.
#
# CI calls `body` on a tag push and hands the result to the forge, so the
# release page carries what NEWS.md says rather than a pointer to the commit
# log. Both forge workflows call this one script: they are kept in sync by
# hand, and two copies of the extraction would drift.
#
# `draft` is for writing the NEWS entry in the first place: it groups the
# commits since the previous tag by their conventional prefix, as a starting
# point to edit, not as the answer.
#
# POSIX sh, and nothing beyond git/sed/awk: this runs in whatever container
# the forge gives the release job.
set -eu

usage() {
	cat >&2 <<EOF
usage: $0 section <version>   NEWS.md section for <version>, or exit 1
       $0 draft [<since>]     grouped commit subjects since <since> (default: last tag)
       $0 body <tag>          what CI publishes: the section, else the draft
EOF
	exit 2
}

news_file() {
	# Run from anywhere in the checkout.
	root=$(git rev-parse --show-toplevel 2>/dev/null) || root=.
	echo "$root/NEWS.md"
}

# The section for one version: everything between its "## <version>" heading
# and the next "## " heading, with trailing blank lines trimmed.
section() {
	version=${1#v}
	file=$(news_file)
	[ -f "$file" ] || return 1
	# Captured rather than piped straight out: the pipeline's status is sed's,
	# which succeeds on the empty input a missing section produces, and an
	# empty release body would then look like a successful publish.
	text=$(awk -v want="## $version" '
		$0 == want { collecting = 1; next }
		collecting && /^## / { exit }
		collecting { print }
	' "$file" | sed -e '/./,$!d' -e :a -e '/^\n*$/{$d;N;ba' -e '}')
	[ -n "$text" ] || return 1
	printf '%s\n' "$text"
}

# One bullet per commit, grouped by conventional prefix. Anything without a
# known prefix lands under "Other" rather than being dropped silently.
draft() {
	since=${1:-}
	if [ -z "$since" ]; then
		since=$(git describe --tags --abbrev=0 2>/dev/null) || since=
	fi
	range=${since:+$since..}HEAD

	subjects=$(git log --no-merges --format='%s' "$range")
	[ -n "$subjects" ] || return 1

	printf '%s\n' "$subjects" | awk '
		function flush(title, key,   n) {
			n = counts[key]
			if (n == 0) return
			printf "### %s\n\n", title
			for (i = 1; i <= n; i++) printf "- %s\n", lines[key, i]
			printf "\n"
		}
		{
			key = "other"
			if (match($0, /^[a-z]+(\([^)]*\))?: /)) {
				key = substr($0, 1, index($0, ":") - 1)
				sub(/\(.*/, "", key)
				$0 = substr($0, RLENGTH + 1)
			}
			# Releases describe themselves; they are not news.
			if ($0 ~ /^Release [0-9]/) next
			if (key != "feat" && key != "fix" && key != "docs" && \
			    key != "refactor" && key != "test" && key != "chore" && \
			    key != "ci" && key != "perf") key = "other"
			lines[key, ++counts[key]] = $0
		}
		END {
			flush("New", "feat")
			flush("Fixed", "fix")
			flush("Changed", "refactor")
			flush("Performance", "perf")
			flush("Documentation", "docs")
			flush("Tests", "test")
			flush("Packaging and CI", "chore")
			flush("Packaging and CI", "ci")
			flush("Other", "other")
		}
	'
}

# What CI publishes. A missing section must not block a release: the tag is
# already pushed by then, and a commit list beats an empty page.
body() {
	tag=$1
	if section "$tag"; then
		return 0
	fi
	echo "No NEWS.md section for $tag; falling back to the commit list." >&2
	printf 'No NEWS.md entry for this release yet. Changes since the previous tag:\n\n'
	# The tag itself is the end of the range, and the previous tag its start.
	previous=$(git describe --tags --abbrev=0 "$tag^" 2>/dev/null) || previous=
	draft_range=${previous:+$previous..}$tag
	git log --no-merges --format='%s' "$draft_range" |
		sed -e '/^Release [0-9]/d' -e 's/^/- /'
}

[ $# -ge 1 ] || usage
command=$1
shift

case $command in
section)
	[ $# -eq 1 ] || usage
	section "$1" || {
		echo "no NEWS.md section for $1" >&2
		exit 1
	}
	;;
draft)
	[ $# -le 1 ] || usage
	draft "${1:-}" || {
		echo "no commits to draft from" >&2
		exit 1
	}
	;;
body)
	[ $# -eq 1 ] || usage
	body "$1"
	;;
*)
	usage
	;;
esac
