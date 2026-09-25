//! Complete control containment from the binary's immutable tokenizer metadata.
//!
//! This is a runtime constructor, not a promotion of the tokenizer parity gate.
//! It applies the same policy as `scripts/gen_token_registries.py`: every
//! tokenizer-special ID plus the four template-significant ordinary IDs. No
//! model file, generated truth-pack file, network or caller-supplied census is
//! needed. Source hashes are checked before any metadata is interpreted.

use serde_json::{Value, json};

use super::{
    embedded::{
        PINNED_ADDED_TOKENS_BYTES, PINNED_SPECIAL_TOKENS_MAP_BYTES,
        PINNED_TOKENIZER_CONFIG_BYTES,
    },
    specials::{ArchivedControlRegistries, ControlRegistryError},
};
use crate::execution_identity::Sha256Digest;

const CONFIG_SHA256: &str = "3edfa64a0826a77e9412b9008f1febf3fe906a68fd616b6de4cd15897a8c8518";
const SPECIALS_SHA256: &str = "b718fce2b7a8940ffeddc1e67f3b092cc0d13ac885c63a021528786f8c4cf6c0";
const ADDED_SHA256: &str = "9e3b127a27647df2c353cc1e5500826f7cdbe8bd15a458e368bba8422e9719cf";
const TEMPLATE_ONLY: [&str; 4] = ["<think>", "</think>", "<tool_call>", "</tool_call>"];

/// Construct the complete pinned tokenizer-special and template-control sets.
///
/// In particular, the forbidden set includes `<unk>`, legacy BOS/EOS,
/// end-of-text, role boundaries, thinking boundaries and tool-call boundaries.
/// Ordinary IDs are not relabeled as tokenizer-special to achieve containment.
pub fn pinned() -> Result<ArchivedControlRegistries, ControlRegistryError> {
    from_sources(
        PINNED_TOKENIZER_CONFIG_BYTES,
        PINNED_SPECIAL_TOKENS_MAP_BYTES,
        PINNED_ADDED_TOKENS_BYTES,
    )
}

fn refused(detail: &str) -> ControlRegistryError {
    ControlRegistryError::InvalidArchive {
        registry: "PinnedControlIds",
        detail: detail.to_owned(),
    }
}

fn verify(bytes: &[u8], length: usize, digest: &str) -> Result<(), ControlRegistryError> {
    if bytes.len() != length || Sha256Digest::of_bytes(bytes).to_hex() != digest {
        return Err(refused("embedded tokenizer metadata differs from pinned source"));
    }
    Ok(())
}

fn from_sources(config: &[u8], specials: &[u8], added: &[u8])
    -> Result<ArchivedControlRegistries, ControlRegistryError>
{
    // Validate the complete three-file authority even though the decoder table
    // contains the IDs. A different special-map/added-token revision is not an
    // interchangeable source for this constructor.
    verify(config, 10_990, CONFIG_SHA256)?;
    verify(specials, 623, SPECIALS_SHA256)?;
    verify(added, 174, ADDED_SHA256)?;
    // These are exact digest-verified bytes, not untrusted JSON. The returned
    // archives still go through the existing duplicate/subset validator.
    let config: Value = serde_json::from_slice(config)
        .map_err(|_| refused("pinned tokenizer configuration is not JSON"))?;
    let decoder = config.get("added_tokens_decoder").and_then(Value::as_object)
        .ok_or_else(|| refused("missing pinned token decoder"))?;
    let mut special_entries = Vec::new();
    let mut control_entries = Vec::new();
    let mut found_template_only = [false; 4];
    for (key, metadata) in decoder {
        let id: u32 = key.parse().map_err(|_| refused("invalid pinned token ID"))?;
        if id >= 166_144 || id.to_string() != *key {
            return Err(refused("noncanonical or out-of-vocabulary pinned token ID"));
        }
        let surface = metadata.get("content").and_then(Value::as_str)
            .filter(|surface| !surface.is_empty())
            .ok_or_else(|| refused("missing pinned token surface"))?;
        let special = metadata.get("special").and_then(Value::as_bool)
            .ok_or_else(|| refused("missing pinned special flag"))?;
        let template_only = TEMPLATE_ONLY.iter().position(|&candidate| candidate == surface);
        if let Some(index) = template_only {
            if special { return Err(refused("template-only token changed tokenizer-special class")); }
            found_template_only[index] = true;
        }
        if special || template_only.is_some() {
            let entry = json!({"id": id, "surface": surface, "special": special});
            if special { special_entries.push(entry.clone()); }
            control_entries.push(entry);
        }
    }
    if found_template_only.iter().any(|found| !found) {
        return Err(refused("incomplete pinned template-control census"));
    }
    let archive = |registry: &str, entries: Vec<Value>| {
        serde_json::to_string(&json!({"schema_version": 1, "registry": registry, "entries": entries}))
            .map_err(|_| refused("control archive serialization failed"))
    };
    let registries = ArchivedControlRegistries::from_archived_json(
        &archive("TokenizerSpecialIds", special_entries)?,
        &archive("TemplateControlIds", control_entries)?,
    )?;
    for (id, surface) in [(0, "<unk>"), (166_100, "<|im_start|>"), (166_101, "<|im_end|>")] {
        if !registries.template_controls().entry(id)
            .is_some_and(|entry| entry.special && entry.surface == surface)
        {
            return Err(refused("pinned padding/BOS/EOS control mismatch"));
        }
    }
    Ok(registries)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_census_contains_all_specials_and_all_template_only_controls() {
        let registries = pinned().unwrap();
        let specials = registries.tokenizer_special_ids();
        let controls = registries.template_controls();
        assert_eq!(specials.ids().iter().copied().collect::<Vec<_>>(),
            [0, 1, 2, 166_100, 166_101, 166_102]);
        assert_eq!(controls.ids().iter().copied().collect::<Vec<_>>(),
            [0, 1, 2, 166_100, 166_101, 166_102, 166_103, 166_104, 166_105, 166_106]);
        for (index, surface) in TEMPLATE_ONLY.iter().enumerate() {
            let id = 166_103 + index as u32;
            let entry = controls.entry(id).unwrap();
            assert_eq!(entry.surface, *surface);
            assert!(!entry.special);
            assert!(!specials.ids().contains(&id));
        }
    }

    #[test]
    fn each_source_is_hash_checked_not_just_the_decoder_table() {
        let sources = [PINNED_TOKENIZER_CONFIG_BYTES, PINNED_SPECIAL_TOKENS_MAP_BYTES, PINNED_ADDED_TOKENS_BYTES];
        for index in 0..sources.len() {
            let mut changed = sources[index].to_vec();
            changed[0] ^= 1; // Keep the length: this must exercise digest refusal.
            let mut inputs = sources;
            inputs[index] = &changed;
            assert!(from_sources(inputs[0], inputs[1], inputs[2]).is_err());
            assert!(verify(&sources[index][..sources[index].len() - 1], sources[index].len(), "unused").is_err());
        }
    }

    #[test]
    fn every_control_is_excluded_from_untrusted_source_tokens() {
        let registries = pinned().unwrap();
        let controls = registries.template_controls();
        let encoder = crate::tasks::extract::SourceDocumentEncoder::pinned(controls).unwrap();
        let text = controls.entries().iter().map(|entry| entry.surface.as_str())
            .collect::<Vec<_>>().join(" é 上海 ");
        let source = encoder.encode(&text, 4096, 4096).unwrap();
        assert!(!source.token_ids().is_empty());
        assert!(source.token_ids().iter().all(|&id| !controls.contains(id)));
    }
}
