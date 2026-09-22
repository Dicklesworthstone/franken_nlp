//! Concrete fsqlite journal, separate from the metadata-only history schema.
//! Rows contain only keyed bindings, counters, states and spool pointers.
use super::{Commitment, JobError, JobSecret, manifest::bounded_json};
use crate::canonjson::{self, ParseLimits};
use fsqlite::{Connection, SqliteValue};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::path::Path;

const ROW_BYTES: usize = 16 * 1024;
pub(super) struct Journal { connection: Connection }
#[derive(Clone, Copy)]
pub(super) enum Table { Header, Item }
impl Table {
    fn read(self) -> &'static str {
        match self {
            Self::Header => "SELECT length(body), substr(body, 1, 16385) FROM job_header WHERE ordinal = ?1",
            Self::Item => "SELECT length(body), substr(body, 1, 16385) FROM job_items WHERE ordinal = ?1",
        }
    }
    fn write(self, insert: bool) -> &'static str {
        match (self, insert) {
            (Self::Header, true) => "INSERT INTO job_header (ordinal, body) VALUES (?1, ?2)",
            (Self::Header, false) => "UPDATE job_header SET body = ?2 WHERE ordinal = ?1",
            (Self::Item, true) => "INSERT INTO job_items (ordinal, body) VALUES (?1, ?2)",
            (Self::Item, false) => "UPDATE job_items SET body = ?2 WHERE ordinal = ?1",
        }
    }
    fn domain(self) -> &'static [u8] { match self { Self::Header => b"journal-header", Self::Item => b"journal-item" } }
}
#[derive(Serialize)]
struct SealedRef<'a, T> { body: &'a T, mac: Commitment }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Sealed<T> { body: T, mac: Commitment }

impl Journal {
    pub fn open(path: &Path, max_bytes: u64) -> Result<Self, JobError> {
        let connection = Connection::open(path.to_str().ok_or(JobError::UnsafeStorage)?).map_err(|_| JobError::Journal)?;
        let journal = Self { connection };
        // No optimistic claim based on accepting a PRAGMA: verify the actual
        // mode and limits. Unsupported foundation behavior fails closed.
        let rows = journal.connection.query_with_params("PRAGMA journal_mode = DELETE", &[]).map_err(|_| JobError::Journal)?;
        match rows.as_slice() {
            [row] if matches!(row.get(0), Some(SqliteValue::Text(mode)) if mode.as_str().eq_ignore_ascii_case("delete")) => {},
            _ => return Err(JobError::Journal),
        }
        journal.connection.execute_batch("PRAGMA synchronous = FULL; PRAGMA page_size = 4096;").map_err(|_| JobError::Journal)?;
        if journal.integer("PRAGMA synchronous")? != 2 || journal.integer("PRAGMA page_size")? != 4096 {
            return Err(JobError::Journal);
        }
        let pages = max_bytes / 4096;
        if journal.integer(&format!("PRAGMA max_page_count = {pages}"))? != pages { return Err(JobError::Limit); }
        Ok(journal)
    }
    fn integer(&self, sql: &str) -> Result<u64, JobError> {
        let rows = self.connection.query_with_params(sql, &[]).map_err(|_| JobError::Journal)?;
        match rows.as_slice() {
            [row] => match row.get(0) {
                Some(SqliteValue::Integer(n)) => u64::try_from(*n).map_err(|_| JobError::Corrupt),
                _ => Err(JobError::Corrupt),
            },
            _ => Err(JobError::Corrupt),
        }
    }
    pub fn check_count(&self, expected: u64) -> Result<(), JobError> {
        if self.integer("SELECT COUNT(*) FROM job_header")? != 1
            || self.integer("SELECT COUNT(*) FROM job_items")? != expected { return Err(JobError::Corrupt); }
        Ok(())
    }
    pub fn create_schema(&self) -> Result<(), JobError> {
        self.connection.execute_batch(
            "CREATE TABLE job_header (ordinal INTEGER PRIMARY KEY, body TEXT NOT NULL);\
             CREATE TABLE job_items (ordinal INTEGER PRIMARY KEY, body TEXT NOT NULL);"
        ).map_err(|_| JobError::Journal)
    }
    pub fn read<T: DeserializeOwned + Serialize>(&self, table: Table, ordinal: u64, key: &JobSecret) -> Result<T, JobError> {
        let parameter = SqliteValue::Integer(i64::try_from(ordinal).map_err(|_| JobError::Limit)?);
        let rows = self.connection.query_with_params(table.read(), &[parameter]).map_err(|_| JobError::Journal)?;
        let row = match rows.as_slice() { [row] => row, _ => return Err(JobError::Corrupt) };
        let text = match (row.get(0), row.get(1)) {
            (Some(SqliteValue::Integer(n)), Some(SqliteValue::Text(text)))
                if *n >= 0 && *n <= ROW_BYTES as i64 && text.len() <= ROW_BYTES => text.as_str(),
            _ => return Err(JobError::Corrupt),
        };
        // Authority-bearing disk JSON uses the repository's duplicate-key,
        // nesting and string-limit chokepoint, never from_str/from_reader.
        let value = canonjson::parse_str_with_limits(text, ParseLimits { max_depth: 32, max_string_bytes: ROW_BYTES })
            .map_err(|_| JobError::Corrupt)?;
        let sealed: Sealed<T> = serde_json::from_value(value).map_err(|_| JobError::Corrupt)?;
        let bytes = bounded_json(&sealed.body, ROW_BYTES)?;
        if !key.commit(table.domain(), &[&ordinal.to_le_bytes(), &bytes]).matches(sealed.mac) {
            return Err(JobError::Authentication);
        }
        Ok(sealed.body)
    }
    pub fn write<T: Serialize>(&self, table: Table, ordinal: u64, key: &JobSecret, value: &T, insert: bool) -> Result<(), JobError> {
        let bytes = bounded_json(value, ROW_BYTES)?;
        let mac = key.commit(table.domain(), &[&ordinal.to_le_bytes(), &bytes]);
        let sealed = bounded_json(&SealedRef { body: value, mac }, ROW_BYTES)?;
        let text = String::from_utf8(sealed).map_err(|_| JobError::Serialization)?;
        let parameters = [SqliteValue::Integer(i64::try_from(ordinal).map_err(|_| JobError::Limit)?), SqliteValue::Text(text.into())];
        if self.connection.execute_with_params(table.write(insert), &parameters).map_err(|_| JobError::Journal)? != 1 {
            return Err(JobError::Corrupt);
        }
        Ok(())
    }
    pub fn transaction<T>(&self, operation: impl FnOnce(&Self) -> Result<T, JobError>) -> Result<T, JobError> {
        self.connection.execute_batch("BEGIN IMMEDIATE").map_err(|_| JobError::Journal)?;
        let mut transaction = Transaction { journal: self, committed: false };
        let result = operation(self)?;
        // Failure can be ambiguous. The owner poisons the current session and
        // requires authenticated recovery; it never retries COMMIT in place.
        self.connection.execute_batch("COMMIT").map_err(|_| JobError::Journal)?;
        transaction.committed = true;
        Ok(result)
    }
}
struct Transaction<'a> { journal: &'a Journal, committed: bool }
impl Drop for Transaction<'_> {
    fn drop(&mut self) {
        if !self.committed { let _ = self.journal.connection.execute_batch("ROLLBACK"); }
    }
}
