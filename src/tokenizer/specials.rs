//! Archived token-control registries for the trusted/untrusted prompt seam.
//!
//! `TemplateControlIds` is the containment authority.  It includes the
//! tokenizer's `special=true` ids *and* template-significant `special=false`
//! markers such as `<think>` and `<tool_call>`.  Do not replace this registry
//! with a `skip_special_tokens` policy: that flag cannot represent the latter
//! class of control token.

use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

use serde_json::Value;

/// One id/surface record retained from an archived truth-pack registry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ControlId {
    pub id: u32,
    pub special: bool,
    pub surface: String,
}

/// The ids whose pinned tokenizer metadata declares `special=true`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenizerSpecialIds {
    entries: Vec<ControlId>,
    ids: BTreeSet<u32>,
}

impl TokenizerSpecialIds {
    #[must_use]
    pub fn entries(&self) -> &[ControlId] {
        &self.entries
    }

    #[must_use]
    pub fn ids(&self) -> &BTreeSet<u32> {
        &self.ids
    }
}

/// The complete untrusted-document forbidden set, loaded only from the
/// archived `TemplateControlIds` census.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TemplateControlIds {
    by_id: BTreeMap<u32, ControlId>,
    entries: Vec<ControlId>,
    ids: BTreeSet<u32>,
}

impl TemplateControlIds {
    #[must_use]
    pub fn contains(&self, id: u32) -> bool {
        self.ids.contains(&id)
    }

    #[must_use]
    pub fn entries(&self) -> &[ControlId] {
        &self.entries
    }

    #[must_use]
    pub fn ids(&self) -> &BTreeSet<u32> {
        &self.ids
    }

    #[must_use]
    pub fn entry(&self, id: u32) -> Option<&ControlId> {
        self.by_id.get(&id)
    }
}

/// Both normative registries after their archive-level subset relation has
/// been validated.  This is the only public archive-loading entry point.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArchivedControlRegistries {
    template_controls: TemplateControlIds,
    tokenizer_special_ids: TokenizerSpecialIds,
}

impl ArchivedControlRegistries {
    /// Parse canonical, duplicate-key-rejecting archive bytes and prove that
    /// all tokenizer-special ids are also forbidden template controls.
    pub fn from_archived_json(
        tokenizer_special_ids: &str,
        template_control_ids: &str,
    ) -> Result<Self, ControlRegistryError> {
        let special_entries = parse_registry(tokenizer_special_ids, "TokenizerSpecialIds")?;
        let control_entries = parse_registry(template_control_ids, "TemplateControlIds")?;

        let tokenizer_special_ids = TokenizerSpecialIds {
            ids: special_entries.iter().map(|entry| entry.id).collect(),
            entries: special_entries,
        };
        if tokenizer_special_ids
            .entries
            .iter()
            .any(|entry| !entry.special)
        {
            return Err(ControlRegistryError::InvalidArchive {
                registry: "TokenizerSpecialIds",
                detail: "every TokenizerSpecialIds entry must record special=true".to_owned(),
            });
        }

        let ids = control_entries.iter().map(|entry| entry.id).collect();
        let by_id = control_entries
            .iter()
            .cloned()
            .map(|entry| (entry.id, entry))
            .collect();
        let template_controls = TemplateControlIds {
            by_id,
            entries: control_entries,
            ids,
        };

        for special in tokenizer_special_ids.entries() {
            let Some(control) = template_controls.entry(special.id) else {
                return Err(ControlRegistryError::MissingTemplateControl {
                    id: special.id,
                    surface: special.surface.clone(),
                });
            };
            if control.surface != special.surface {
                return Err(ControlRegistryError::MismatchedTemplateControl {
                    id: special.id,
                    special_surface: special.surface.clone(),
                    control_surface: control.surface.clone(),
                });
            }
        }

        Ok(Self {
            template_controls,
            tokenizer_special_ids,
        })
    }

    #[must_use]
    pub fn template_controls(&self) -> &TemplateControlIds {
        &self.template_controls
    }

    #[must_use]
    pub fn tokenizer_special_ids(&self) -> &TokenizerSpecialIds {
        &self.tokenizer_special_ids
    }
}

/// A malformed or non-normative archived registry is rejected before it can
/// become an untrusted-document exclusion policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ControlRegistryError {
    InvalidArchive {
        registry: &'static str,
        detail: String,
    },
    MissingTemplateControl {
        id: u32,
        surface: String,
    },
    MismatchedTemplateControl {
        id: u32,
        special_surface: String,
        control_surface: String,
    },
}

impl fmt::Display for ControlRegistryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidArchive { registry, detail } => {
                write!(formatter, "invalid {registry} archive: {detail}")
            }
            Self::MissingTemplateControl { id, surface } => write!(
                formatter,
                "TemplateControlIds is missing tokenizer-special id={id} surface={surface:?}"
            ),
            Self::MismatchedTemplateControl {
                id,
                special_surface,
                control_surface,
            } => write!(
                formatter,
                "TemplateControlIds surface mismatch id={id} special={special_surface:?} control={control_surface:?}"
            ),
        }
    }
}

impl Error for ControlRegistryError {}

fn parse_registry(
    source: &str,
    expected_name: &'static str,
) -> Result<Vec<ControlId>, ControlRegistryError> {
    let value = crate::canonjson::parse_str(source).map_err(|error| {
        ControlRegistryError::InvalidArchive {
            registry: expected_name,
            detail: error.to_string(),
        }
    })?;
    let root = value
        .as_object()
        .ok_or_else(|| ControlRegistryError::InvalidArchive {
            registry: expected_name,
            detail: "root must be an object".to_owned(),
        })?;
    let name = root
        .get("registry")
        .and_then(Value::as_str)
        .ok_or_else(|| ControlRegistryError::InvalidArchive {
            registry: expected_name,
            detail: "missing string /registry".to_owned(),
        })?;
    if name != expected_name {
        return Err(ControlRegistryError::InvalidArchive {
            registry: expected_name,
            detail: format!("expected registry={expected_name:?}, observed={name:?}"),
        });
    }
    if root.get("schema_version").and_then(Value::as_u64) != Some(1) {
        return Err(ControlRegistryError::InvalidArchive {
            registry: expected_name,
            detail: "expected schema_version=1".to_owned(),
        });
    }
    let raw_entries = root
        .get("entries")
        .and_then(Value::as_array)
        .ok_or_else(|| ControlRegistryError::InvalidArchive {
            registry: expected_name,
            detail: "missing array /entries".to_owned(),
        })?;
    if raw_entries.is_empty() {
        return Err(ControlRegistryError::InvalidArchive {
            registry: expected_name,
            detail: "/entries must not be empty".to_owned(),
        });
    }

    let mut ids = BTreeSet::new();
    let mut surfaces = BTreeSet::new();
    let mut entries = Vec::with_capacity(raw_entries.len());
    for (index, raw_entry) in raw_entries.iter().enumerate() {
        let entry = raw_entry
            .as_object()
            .ok_or_else(|| ControlRegistryError::InvalidArchive {
                registry: expected_name,
                detail: format!("/entries/{index} must be an object"),
            })?;
        let id = entry
            .get("id")
            .and_then(Value::as_u64)
            .and_then(|id| u32::try_from(id).ok())
            .ok_or_else(|| ControlRegistryError::InvalidArchive {
                registry: expected_name,
                detail: format!("/entries/{index}/id must be a u32"),
            })?;
        let surface = entry
            .get("surface")
            .and_then(Value::as_str)
            .filter(|surface| !surface.is_empty())
            .ok_or_else(|| ControlRegistryError::InvalidArchive {
                registry: expected_name,
                detail: format!("/entries/{index}/surface must be a non-empty string"),
            })?;
        let special = entry
            .get("special")
            .and_then(Value::as_bool)
            .ok_or_else(|| ControlRegistryError::InvalidArchive {
                registry: expected_name,
                detail: format!("/entries/{index}/special must be a boolean"),
            })?;
        if !ids.insert(id) || !surfaces.insert(surface.to_owned()) {
            return Err(ControlRegistryError::InvalidArchive {
                registry: expected_name,
                detail: format!("duplicate id or surface at /entries/{index}"),
            });
        }
        entries.push(ControlId {
            id,
            special,
            surface: surface.to_owned(),
        });
    }
    entries.sort_by_key(|entry| entry.id);
    Ok(entries)
}
