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
pub mod robot;
pub mod storage;
pub mod tasks;
pub mod template;
pub mod textutil;
pub mod tokenizer;
pub mod validation;

pub use orchestrator::{
    CommittedMemory, EngineBuildError, EngineLease, EngineResources, LeakResponsePolicy,
    MemoryClass, MemoryClassCharge, MemoryReservation, MemorySnapshot, NlpEngine,
    NlpEngineBuilder, ReservationError, ResourceBrokerError, ResourceConfigConflict,
    ResourceConfigError, ResourceConfigField, ResourceConfigValue, ResourceHostConfig,
    RuntimeHostError, RuntimePreset, ThreadInventory, install_process_resources,
    installed_process_resources,
};

pub fn cli_main() -> std::process::ExitCode {
    cli::cli_main()
}
