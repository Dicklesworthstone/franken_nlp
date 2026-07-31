use franken_nlp::storage::{
    ColumnClass, ColumnSpec, MetadataStore, StorageError, StoreConfig, TableSpec, schema_tables,
    validate_schema_policy, validate_table_schema,
};

fn report(columns_checked: usize) {
    for table in schema_tables() {
        for column in table.columns {
            println!(
                "STORAGE_POLICY table={}.{} class={:?} allowlist=PASS",
                table.name, column.name, column.class
            );
        }
    }
    println!("STORAGE_POLICY RESULT=PASS columns_checked={columns_checked}");
}

#[test]
fn schema_allowlist_contains_only_metadata_columns() {
    let columns_checked = validate_schema_policy(schema_tables())
        .expect("the compiled storage schema must remain metadata-only");
    assert!(columns_checked > 0);
    report(columns_checked);
}

#[test]
fn content_bearing_column_is_rejected() {
    const BAD_COLUMNS: &[ColumnSpec] = &[
        ColumnSpec {
            name: "job_id",
            class: ColumnClass::Identifier,
        },
        ColumnSpec {
            name: "manifest_digest",
            class: ColumnClass::Digest,
        },
        ColumnSpec {
            name: "state",
            class: ColumnClass::State,
        },
        ColumnSpec {
            name: "created_at_ms",
            class: ColumnClass::TimestampMillis,
        },
        ColumnSpec {
            name: "document_text",
            class: ColumnClass::Digest,
        },
    ];
    let bad_table = TableSpec {
        name: "jobs",
        columns: BAD_COLUMNS,
    };
    assert_eq!(
        validate_table_schema(&bad_table),
        Err(StorageError::SchemaPolicyViolation {
            table: "jobs",
            column: "document_text",
        })
    );
    println!(
        "STORAGE_POLICY RESULT=EXPECTED-FAIL table=jobs.document_text class={:?}",
        ColumnClass::Digest
    );
    report(
        schema_tables()
            .iter()
            .map(|table| table.columns.len())
            .sum(),
    );
}

#[test]
fn typed_error_summary_cannot_carry_error_message_text() {
    assert_eq!(
        std::mem::size_of::<franken_nlp::storage::TypedErrorSummary>(),
        4,
        "the persisted error surface is exactly two u16 codes"
    );
    report(
        schema_tables()
            .iter()
            .map(|table| table.columns.len())
            .sum(),
    );
}

#[test]
fn disabled_store_does_not_touch_its_path() {
    let path = std::env::temp_dir().join(format!(
        "franken-nlp-storage-disabled-{}",
        std::process::id()
    ));
    let store = MetadataStore::open(StoreConfig::disabled_at_path(path.clone()))
        .expect("disabled store must not attempt to open a database");
    assert!(!store.is_open());
    assert!(
        !path.exists(),
        "disabled configuration must make no filesystem touch"
    );
    report(
        schema_tables()
            .iter()
            .map(|table| table.columns.len())
            .sum(),
    );
}

#[cfg(unix)]
#[test]
fn unix_owner_only_database_contract_is_fixed_at_0600() {
    assert_eq!(franken_nlp::storage::OWNER_ONLY_DATABASE_MODE, 0o600);
    report(
        schema_tables()
            .iter()
            .map(|table| table.columns.len())
            .sum(),
    );
}

#[cfg(not(feature = "metadata-store"))]
#[test]
fn enabled_runtime_configuration_refuses_an_uncompiled_store() {
    let result = MetadataStore::open(StoreConfig::metadata_only("not-opened.db"));
    assert_eq!(
        result.err(),
        Some(StorageError::MetadataStoreFeatureDisabled)
    );
    report(
        schema_tables()
            .iter()
            .map(|table| table.columns.len())
            .sum(),
    );
}

#[cfg(all(feature = "metadata-store", unix))]
#[test]
fn enabled_store_round_trips_metadata_only_state_with_owner_only_permissions() {
    use std::os::unix::fs::PermissionsExt;

    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock must be after the Unix epoch")
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "franken-nlp-storage-policy-{}-{nanos}.db",
        std::process::id()
    ));

    let job_id = franken_nlp::storage::MetadataId::new(1);
    let item_id = franken_nlp::storage::MetadataId::new(2);
    let store = MetadataStore::open(StoreConfig::metadata_only(path.clone()))
        .expect("enabled store must create the metadata schema");
    assert!(store.is_open());
    assert_eq!(
        std::fs::metadata(&path)
            .expect("enabled store must create its database")
            .permissions()
            .mode()
            & 0o077,
        0,
        "created database must remain owner-only"
    );

    store
        .record_job(franken_nlp::storage::JobMetadata {
            job_id,
            manifest_digest: franken_nlp::storage::Sha256Digest::new([1; 32]),
            state: franken_nlp::storage::JobState::Pending,
            created_at_ms: 1,
            updated_at_ms: 1,
        })
        .expect("job metadata must persist");
    store
        .record_item(franken_nlp::storage::ItemMetadata {
            job_id,
            item_id,
            input_digest: franken_nlp::storage::Sha256Digest::new([2; 32]),
            state: franken_nlp::storage::JobState::Pending,
            attempt_count: 0,
            updated_at_ms: 1,
        })
        .expect("item metadata must persist");
    store
        .record_state_transition(franken_nlp::storage::StateTransitionMetadata {
            transition_id: franken_nlp::storage::MetadataId::new(3),
            job_id,
            item_id,
            state: franken_nlp::storage::JobState::Running,
            recorded_at_ms: 2,
        })
        .expect("state transition must persist");
    store
        .record_error(franken_nlp::storage::ErrorMetadata {
            error_id: franken_nlp::storage::MetadataId::new(4),
            job_id,
            item_id,
            summary: franken_nlp::storage::TypedErrorSummary {
                code: 7,
                context_code: 9,
            },
            recorded_at_ms: 3,
        })
        .expect("typed error summary must persist without message text");
    assert_eq!(
        store.job_state(job_id),
        Ok(Some(franken_nlp::storage::JobState::Pending))
    );
    assert_eq!(
        store.item_state(item_id),
        Ok(Some(franken_nlp::storage::JobState::Running))
    );
    store.close();

    let reopened = MetadataStore::open(StoreConfig::metadata_only(path))
        .expect("owner-only database must reopen");
    assert_eq!(
        reopened.job_state(job_id),
        Ok(Some(franken_nlp::storage::JobState::Pending))
    );
    assert_eq!(
        reopened.item_state(item_id),
        Ok(Some(franken_nlp::storage::JobState::Running))
    );
    report(
        schema_tables()
            .iter()
            .map(|table| table.columns.len())
            .sum(),
    );
}
