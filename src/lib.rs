#![deny(unsafe_code)]

pub mod artifact;
pub mod batch;
pub mod calibration;
pub mod canonjson;
pub mod cli;
pub mod error;
pub mod execution_identity;
pub mod grammar;
pub mod jobs;
pub mod native_engine;
pub mod orchestrator;
pub mod receipt;
pub mod robot;
pub mod storage;
pub mod tasks;
pub mod template;
pub mod textutil;
pub mod tokenizer;
pub mod validation;

pub use orchestrator::{
    AdmissionBuildError, AdmissionCertificate, AdmissionDecision, AdmissionError, AdmissionRejection,
    AdmissionRequest, AdmissionReservation, AdmissionTerm, AdmissionTerms, BF16_KV_BYTES_PER_TOKEN,
    BlockingClosureGuard, CommittedAdmission, CommittedMemory, DEFAULT_CONTEXT_TOKEN_CAP,
    EngineBuildError, EngineCallGuard, EngineLease, EngineResources, FULL_F32_LOGIT_ROW_BYTES,
    INT8_KV_F16_SCALE_BYTES_PER_TOKEN, INT8_KV_F32_SCALE_BYTES_PER_TOKEN,
    INT8_KV_PAYLOAD_BYTES_PER_TOKEN, KvCacheQuantization, LeakResponsePolicy, MemoryClass,
    MemoryClassCharge, MemoryReservation, MemorySnapshot, NlpEngine, NlpEngineBuilder,
    OutstandingClosureSnapshot, ReentrantCall, ReservationError, ResidencyAccounting,
    ResourceBrokerError, ResourceConfigConflict, ResourceConfigError, ResourceConfigField,
    ResourceConfigValue, ResourceHostConfig, RuntimeHostError, RuntimePreset, ThreadInventory,
    install_process_resources, installed_process_resources,
};

pub fn cli_main() -> std::process::ExitCode {
    cli::cli_main()
}
