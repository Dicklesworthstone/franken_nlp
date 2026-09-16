# Model-free token and text commands

The commands below execute implemented library paths without model weights.
This change has not been compiled, tested or qualified under the current
code-first policy. No token accuracy, language coverage, throughput or
platform certification is inferred from source inspection.

## Exact tokenizer inspection

```sh
fnlp tokens document.txt
fnlp tokens --ids --json document.txt
fnlp tokens --no-bos --eos --json document.txt
fnlp tokens --byte-fallback --ids --json document.txt
```

The default is the pinned SentencePiece L0 BPE engine with its actual added-token
registry, BOS enabled and EOS disabled. Plain mode prints the count followed by
a newline. `--ids` prints a JSON ID array instead; with `--json`, the envelope
always includes the count and optionally the IDs. It also records the encoding
profile, BOS/EOS policy, source byte/scalar counts and all four embedded-asset
digests. No model is loaded, fetched or substituted.

This is **tokenizer inspection**, not untrusted-prompt construction. Marker-like
text may be recognized as an added token in L0 mode. SourceDocument's separate
trust boundary must still be used for model prompts.

The simple reference BPE algorithm is currently admitted only for input budgets
up to **4096 bytes**. The command refuses larger declared budgets before reading
input. It does not silently switch algorithms or return approximate counts.

`--byte-fallback` explicitly selects a different byte-table encoding with no
BOS/EOS insertion or added-token recognition. Its count equals the original
UTF-8 byte length; this is not the ordinary BPE count. The exact byte decode is
checked before output. This mode can accept a larger explicit input budget,
subject to the token-count budget. It conflicts with `--no-bos` and `--eos`.
No decode-to-text or lossy decode CLI mode is implied.

## Lossless splitting

```sh
fnlp split --max-chunk-bytes 4096 document.txt
fnlp split --max-chunk-bytes 4096 --json document.txt
```

Without `--json`, emit one versioned NDJSON row per chunk; each row contains
`index`, `profile`, and `chunk` with original byte/scalar spans and exact text.
With `--json`, emit one complete envelope with a `chunks` array. Empty input
has zero chunks. Concatenating chunk text reconstructs the input byte-for-byte.

This is a conservative size/whitespace partition, **not sentence detection**.
Cuts prefer ASCII whitespace in the latter half of a bounded chunk; otherwise
they use a UTF-8 boundary. CRLF pairs remain together. No whitespace is trimmed,
no overlap duplicated, and no character or separator is synthesized. The byte
ceiling must be at least 4 to admit every Unicode scalar. Chunk-count or output
exhaustion returns no result rather than truncating the document.

## Narrow normalization and coordinate maps

```sh
fnlp normalize document.txt
fnlp normalize --trim-ascii-horizontal --collapse-ascii-horizontal --json document.txt
```

CRLF and bare CR always become LF. All other changes require explicit flags:
`--trim-ascii-horizontal` removes ASCII space/tab at each logical line's edges;
`--collapse-ascii-horizontal` turns retained runs of ASCII space/tab into one
ASCII space. Non-ASCII spaces, combining marks, zero-width text and all other
Unicode scalars are untouched. There is no NFC/NFKC, case folding, grapheme
rewriting, fuzzy relocation or generic Unicode-whitespace cleanup.

Plain output adds no trailing newline. The JSON envelope includes changed-run
maps with original and normalized byte/scalar intervals and the operation.
Unchanged gaps are exact translations. The library's `original_to_normalized`
and `normalized_to_original` boundary methods return `None` for interior UTF-8
bytes, changed-run interiors, or ambiguous inverse boundaries of deleted runs.
They do not select an arbitrary original endpoint. Maps describe coordinates,
not proof that a transformed span still has the original surface text.

## Common admission and output behavior

Each command accepts a positional UTF-8 file, or `-`/omission for stdin; malformed
UTF-8 is refused rather than repaired. `--max-input-bytes` bounds reads to at
most the cap plus one overflow-detection byte. Its default is 1 MiB for split
and normalize, and 4096 bytes for token inspection. `--max-items` bounds tokens,
chunks or normalization edits, respectively (default 16384). `--max-output-bytes`
bounds the complete result including metadata and emitted record newlines
(default 4 MiB). CLI byte budgets cannot exceed 64 MiB, and item limits cannot
exceed 1000000. Input and output budgets are distinct.

All output is prepared before the first external write. Errors before publication
leave stdout empty and emit fixed-reason JSON diagnostics on stderr. OS write
failure can still occur after a partial stdout write. `-o -` selects stdout.

`-o FILE` reuses the protected local-file implementation: Linux x86_64/aarch64,
procfs, an existing process-owned 0700 parent, no symlink path components or
traversal, exclusive 0600 staging, complete write/sync, and non-replacing
same-directory hard-link publication. Existing files are never overwritten.
An uncertain publication/sync is an error and may leave a complete final file.
Other platforms refuse protected file output rather than weakening its policy.
The model-root transaction remains independently gated; this does not ratify it.
