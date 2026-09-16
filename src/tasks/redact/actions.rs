//! Transactional in-memory edits: build every replacement and check the entire
//! result envelope before returning. No filesystem write or partial stream.

use std::collections::{BTreeMap, BTreeSet};
use serde::{Deserialize, Serialize};
use crate::{canonjson, execution_identity::Sha256Digest, tasks::ner::EntityType};
use super::{PiiKind, RedactError, detectors::{RuleSet, RULE_PROFILE},
    pseudonym::{PseudonymIdentity, Pseudonyms}, union::{DetectedDocument, RedactionRegion, OVERLAP_POLICY}};

pub const ACTION_POLICY_VERSION: &str = "redact-mask-placeholder-pseudonym-v1";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RedactionAction { Mask, Placeholder, Pseudonymize }

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ActionPolicy {
    pub default_action: RedactionAction,
    pub per_type: BTreeMap<PiiKind, RedactionAction>,
    /// Maps are sensitive output, never telemetry, and are opt-in.
    pub include_map: bool,
    /// Resume must name the saved HMAC commitment, not a caller key label.
    pub expected_key_commitment: Option<String>,
}
impl Default for ActionPolicy {
    fn default() -> Self {
        Self { default_action: RedactionAction::Placeholder, per_type: BTreeMap::new(),
            include_map: false, expected_key_commitment: None }
    }
}
impl ActionPolicy {
    pub(crate) fn check_key(&self, context: Option<&Pseudonyms<'_>>) -> Result<(), RedactError> {
        let needs_key = self.default_action == RedactionAction::Pseudonymize
            || self.per_type.values().any(|a| *a == RedactionAction::Pseudonymize);
        if needs_key && context.is_none() { return Err(RedactError::MissingKey); }
        if let Some(expected) = &self.expected_key_commitment {
            if context.is_none_or(|c| c.identity().key_commitment != *expected) { return Err(RedactError::KeyMismatch); }
        }
        Ok(())
    }
    fn action(&self, region: &RedactionRegion) -> Result<RedactionAction, RedactError> {
        if region.kinds.is_empty() { return Err(RedactError::InvalidSpan); }
        let mut actions = region.kinds.iter().map(|k| self.per_type.get(k).copied().unwrap_or(self.default_action));
        let first = actions.next().ok_or(RedactError::InvalidSpan)?;
        // The whole connected component is masked if type policies disagree
        // or a pseudonym would require inventing one type for a mixed region.
        // This conservative fallback is versioned and visible in each edit.
        if actions.any(|a| a != first) || (region.kinds.len() > 1 && first == RedactionAction::Pseudonymize) {
            Ok(RedactionAction::Mask)
        } else { Ok(first) }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EditBudget { pub max_regions: usize, pub max_output_bytes: usize }
impl Default for EditBudget {
    fn default() -> Self { Self { max_regions: 16_384, max_output_bytes: 4 * 1024 * 1024 } }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RedactionEdit {
    pub original: RedactionRegion,
    pub output_byte_start: usize,
    pub output_byte_end: usize,
    pub output_scalar_start: usize,
    pub output_scalar_end: usize,
    pub applied_action: RedactionAction,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationStatus { NotRequested, CleanDeclaredUnion }

/// No Debug or Deserialize: a successful result is emitted explicitly, never
/// constructed by deserializing an unverified receipt or dumped into a log.
#[derive(Serialize)]
pub struct RedactionResult {
    pub(crate) schema_version: u32,
    pub(crate) task_spec_version: String,
    pub(crate) text: String,
    pub(crate) rule_profile: String,
    pub(crate) rules: RuleSet,
    pub(crate) model_types: BTreeSet<EntityType>,
    pub(crate) overlap_policy: String,
    pub(crate) action_policy: String,
    pub(crate) policy_digest: Sha256Digest,
    pub(crate) detection_count: usize,
    pub(crate) region_count: usize,
    pub(crate) pseudonym_identity: Option<PseudonymIdentity>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) edits: Vec<RedactionEdit>,
    pub(crate) verification: VerificationStatus,
}
impl RedactionResult {
    pub fn text(&self) -> &str { &self.text }
    pub fn edits(&self) -> &[RedactionEdit] { &self.edits }
    pub fn model_types(&self) -> &BTreeSet<EntityType> { &self.model_types }
    pub fn region_count(&self) -> usize { self.region_count }
    pub fn policy_digest(&self) -> Sha256Digest { self.policy_digest }
    pub fn verification(&self) -> VerificationStatus { self.verification }
    pub(crate) fn check_size(&self, cap: usize) -> Result<(), RedactError> {
        if canonjson::canonical_bytes(self).map_err(|_| RedactError::Serialization)?.len() > cap {
            return Err(RedactError::OutputBudget);
        }
        Ok(())
    }
}

#[derive(Serialize)]
struct PolicyBinding<'a> {
    rule_profile: &'static str, overlap_policy: &'static str, action_version: &'static str,
    policy: &'a ActionPolicy, rules: &'a RuleSet, model_types: &'a BTreeSet<EntityType>,
    pseudonyms: Option<&'a PseudonymIdentity>, budget: EditBudget,
}

/// Replace against immutable original coordinates. Forward construction is
/// equivalent to back-to-front replacement without repeated suffix shifting;
/// every untouched slice stays byte-exact. No raw originals enter the map.
pub fn apply(document: &DetectedDocument<'_>, policy: &ActionPolicy, context: Option<&Pseudonyms<'_>>, budget: EditBudget) -> Result<RedactionResult, RedactError> {
    policy.check_key(context)?;
    if document.regions.len() > budget.max_regions || budget.max_regions > 16_384 { return Err(RedactError::DetectionBudget); }
    let mut replacements = Vec::new();
    replacements.try_reserve_exact(document.regions.len()).map_err(|_| RedactError::AllocationRefused)?;
    let mut old_byte = 0; let mut old_scalar = 0; let mut size = document.source.len();
    let mut replacement_bytes = 0_usize;
    for region in &document.regions {
        let span = region.span;
        let prefix = document.source.get(old_byte..span.byte_start).ok_or(RedactError::InvalidSpan)?;
        old_scalar += prefix.chars().count();
        if old_scalar != span.scalar_start || span.byte_start >= span.byte_end { return Err(RedactError::InvalidSpan); }
        let value = document.source.get(span.byte_start..span.byte_end).ok_or(RedactError::InvalidSpan)?;
        old_scalar += value.chars().count();
        if old_scalar != span.scalar_end { return Err(RedactError::InvalidSpan); }
        old_byte = span.byte_end;
        let action = policy.action(region)?;
        let kind = *region.kinds.iter().next().ok_or(RedactError::InvalidSpan)?;
        let predicted = match action {
            RedactionAction::Mask => span.scalar_end - span.scalar_start,
            RedactionAction::Placeholder => "[redacted:]".len() + if region.kinds.len() == 1 { kind.label().len() } else { "overlap".len() },
            RedactionAction::Pseudonymize => kind.label().len() + 3 + match context.ok_or(RedactError::MissingKey)?.identity().encoding {
                super::pseudonym::PseudonymEncoding::Full256 => 64,
                super::pseudonym::PseudonymEncoding::Preflighted128 => 32,
            },
        };
        replacement_bytes = replacement_bytes.checked_add(predicted).filter(|&n| n <= budget.max_output_bytes)
            .ok_or(RedactError::OutputBudget)?;
        let replacement = match action {
            RedactionAction::Mask => {
                let len = span.scalar_end - span.scalar_start;
                if len > budget.max_output_bytes { return Err(RedactError::OutputBudget); }
                let mut s = String::new(); s.try_reserve_exact(len).map_err(|_| RedactError::AllocationRefused)?;
                for _ in 0..len { s.push('*'); } s
            }
            RedactionAction::Placeholder => format!("[redacted:{}]", if region.kinds.len() == 1 {
                region.kinds.iter().next().ok_or(RedactError::InvalidSpan)?.label()
            } else { "overlap" }),
            RedactionAction::Pseudonymize => context.ok_or(RedactError::MissingKey)?
                .pseudonym(*region.kinds.iter().next().ok_or(RedactError::InvalidSpan)?, value)?,
        };
        size = size.checked_sub(value.len()).and_then(|n| n.checked_add(replacement.len())).ok_or(RedactError::OutputBudget)?;
        if replacement.len() != predicted { return Err(RedactError::InvalidOptions); }
        replacements.push((replacement, action));
    }
    if size > budget.max_output_bytes || replacement_bytes > budget.max_output_bytes { return Err(RedactError::OutputBudget); }
    let mut text = String::new(); text.try_reserve_exact(size).map_err(|_| RedactError::AllocationRefused)?;
    let mut edits = Vec::new();
    if policy.include_map { edits.try_reserve_exact(document.regions.len()).map_err(|_| RedactError::AllocationRefused)?; }
    let mut old = 0; let mut scalars = 0;
    for (region, (replacement, action)) in document.regions.iter().zip(replacements) {
        let prefix = &document.source[old..region.span.byte_start];
        text.push_str(prefix); scalars += prefix.chars().count();
        let start_byte = text.len(); let start_scalar = scalars;
        text.push_str(&replacement); scalars += replacement.chars().count();
        if policy.include_map { edits.push(RedactionEdit { original: region.clone(), output_byte_start: start_byte,
            output_byte_end: text.len(), output_scalar_start: start_scalar, output_scalar_end: scalars, applied_action: action }); }
        old = region.span.byte_end;
    }
    text.push_str(&document.source[old..]);
    let binding = PolicyBinding { rule_profile: RULE_PROFILE, overlap_policy: OVERLAP_POLICY,
        action_version: ACTION_POLICY_VERSION, policy, rules: &document.rule_set, model_types: &document.model_types,
        pseudonyms: context.map(Pseudonyms::identity), budget };
    let policy_digest = Sha256Digest::of_bytes(&canonjson::canonical_bytes(&binding).map_err(|_| RedactError::Serialization)?);
    let result = RedactionResult { schema_version: 1, task_spec_version: "redact-v1".to_owned(), text,
        rule_profile: RULE_PROFILE.to_owned(), rules: document.rule_set.clone(), model_types: document.model_types.clone(),
        overlap_policy: OVERLAP_POLICY.to_owned(), action_policy: ACTION_POLICY_VERSION.to_owned(), policy_digest,
        detection_count: document.detection_count, region_count: document.regions.len(),
        pseudonym_identity: context.map(|c| c.identity().clone()), edits, verification: VerificationStatus::NotRequested };
    result.check_size(budget.max_output_bytes)?; Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::{detectors::RuleBudget, pseudonym::{PseudonymKey, PseudonymBudget}};
    fn document(s: &str) -> DetectedDocument<'_> { DetectedDocument::rules_only(s, &RuleSet::default(), RuleBudget::default()).unwrap() }
    #[test]
    fn mask_and_map_keep_original_and_transformed_coordinates_separate() {
        let s = "é a@example.org 上海"; let d = document(s);
        let p = ActionPolicy { default_action: RedactionAction::Mask, include_map: true, ..ActionPolicy::default() };
        let out = apply(&d, &p, None, EditBudget::default()).unwrap();
        assert_eq!(out.text, "é ************* 上海"); assert_eq!(out.edits.len(), 1);
        assert_eq!(out.edits[0].original.span.byte_start, 3); assert_eq!(out.edits[0].output_scalar_start, 2);
    }
    #[test]
    fn placeholders_remove_all_original_detected_bytes() {
        let s = "a@example.org; https://example.org/private";
        let out = apply(&document(s), &ActionPolicy::default(), None, EditBudget::default()).unwrap();
        assert_eq!(out.text, "[redacted:email]; [redacted:url]"); assert!(out.edits.is_empty());
        assert_eq!(out.verification(), VerificationStatus::NotRequested);
    }
    #[test]
    fn missing_key_or_wrong_resume_key_never_falls_back_to_raw_output() {
        let p = ActionPolicy { default_action: RedactionAction::Pseudonymize, ..ActionPolicy::default() };
        assert!(matches!(apply(&document("a@b.org"), &p, None, EditBudget::default()), Err(RedactError::MissingKey)));
        let k = PseudonymKey::from_bytes(&[1;32], "key").unwrap(); let c = Pseudonyms::full256(&k, "job", None).unwrap();
        let mut p = p; p.expected_key_commitment = Some("wrong".to_owned());
        assert!(matches!(apply(&document("a@b.org"), &p, Some(&c), EditBudget::default()), Err(RedactError::KeyMismatch)));
    }
    #[test]
    fn pseudonyms_are_consistent_and_omitted_values_cannot_extend_short_mode() {
        let k = PseudonymKey::from_bytes(&[1;32], "key").unwrap();
        let c = Pseudonyms::preflight128(&k, "job", &[(PiiKind::Email, "a@b.org")], PseudonymBudget::default(), None).unwrap();
        let p = ActionPolicy { default_action: RedactionAction::Pseudonymize, ..ActionPolicy::default() };
        let out = apply(&document("a@b.org a@b.org"), &p, Some(&c), EditBudget::default()).unwrap();
        let parts: Vec<_> = out.text.split(' ').collect(); assert_eq!(parts[0], parts[1]);
        assert!(apply(&document("x@y.org"), &p, Some(&c), EditBudget::default()).is_err());
    }
    #[test]
    fn complete_envelope_including_map_is_budgeted() {
        let p = ActionPolicy { include_map: true, ..ActionPolicy::default() };
        assert!(matches!(apply(&document("a@b.org"), &p, None, EditBudget { max_regions: 10, max_output_bytes: 40 }), Err(RedactError::OutputBudget)));
    }
    #[test]
    fn source_is_unchanged_outside_edits_and_maps_never_contain_original_values() {
        let s = "e\u{301}:a@b.org\n𐀀";
        let out = apply(&document(s), &ActionPolicy { include_map: true, ..ActionPolicy::default() }, None, EditBudget::default()).unwrap();
        assert_eq!(out.text, "e\u{301}:[redacted:email]\n𐀀");
        let map = canonjson::canonical_string(&out.edits).unwrap(); assert!(!map.contains("a@b.org"));
    }
}
