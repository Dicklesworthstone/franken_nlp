//! Native calibration, authenticated reuse and mixed raw/calibrated comparison.
//! Every operation owns preparation, actual replay, fitting/application and
//! final reports in one physical invocation and one nonrenewable work budget.

use super::*;
use crate::assay::{calibrate::FitOptions,
    native_classify::calibrate::{self as calibration, ApplicationOutcome, FitOutcome,
        PreparedApplication, QualificationOutcome}};

/// Select the complete transient envelope before reading input or artifacts.
/// Raw evaluation/qualification retains the original quote. A fit has one
/// model; application retains one or two independently authenticated artifacts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Int4AssayOperation {
    Raw,
    Fit,
    Apply { artifacts: usize },
}

impl Int4AssayHostLimits {
    /// Complete native Assay quote for the selected operation. Resident models,
    /// outer caller IO/parse staging and RunLimits cleanup remain separate.
    /// Added input ownership covers fit clone/sort/PAV storage, parameters and
    /// artifact authentication/restoration/report staging, as applicable.
    pub fn requirements_for_operation(self, group_widths: &[usize], limits: metrics::Limits,
        operation: Int4AssayOperation) -> Result<Int4AssayHostRequirements, HostedError> {
        let mut required = self.requirements(group_widths, limits)?;
        let additional = match operation {
            Int4AssayOperation::Raw => 0,
            Int4AssayOperation::Fit if group_widths.len() == 1 =>
                calibration::required_calibration_bytes(limits, true, 0)?,
            Int4AssayOperation::Apply { artifacts } if (1..=group_widths.len()).contains(&artifacts) =>
                calibration::required_calibration_bytes(limits, false, artifacts)?,
            _ => return Err(HostedError::Limits("INT4 Assay calibration operation shape")),
        };
        required.input_bytes = sum(&[required.input_bytes, additional])?;
        Ok(required)
    }
}

struct CalibrationInput {
    base: Input,
    applications: [Option<PreparedApplication>; 2],
}

impl NlpEngine {
    /// Fit only the sealed calibration split of real native observations and
    /// measure only its disjoint held-out population. The persisted artifact
    /// is scoped to the complete frozen suite and actual native observations.
    #[allow(clippy::too_many_arguments)]
    pub fn calibrate_int4_classification(&self, model: &ResidentInt4, dataset: Dataset,
        key: CommitmentKey, namespace: String, fit_options: FitOptions, options: EvalOptions,
        limits: metrics::Limits, host: Int4AssayHostLimits, kernel: PackedInt4Kernel,
        cancellation: CancellationToken) -> Result<HostedOutput<FitOutcome>, HostedError> {
        calibration::validate_fit_options(fit_options, limits)?;
        let input = CalibrationInput { base: Input { dataset, key, namespace, host_scope: None },
            applications: [None, None] };
        dispatch_owned(self, model, None, input, true, options, limits, host, kernel, cancellation,
            move |replay, _, input, control| calibration::fit_with_checkpoint::<HostedError>(
                replay, &input.base.key, &input.base.namespace, fit_options, options, limits,
                &mut || checkpoint(control)))
    }

    /// Apply one previously authenticated calibration artifact to a new actual
    /// replay of its exact frozen suite. Dataset/model/score mismatches remain
    /// hard errors; validity and declared shift produce explicit typed outcomes.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_int4_classification_calibrated(&self, model: &ResidentInt4, dataset: Dataset,
        application: PreparedApplication, key: CommitmentKey, namespace: String,
        options: EvalOptions, limits: metrics::Limits, host: Int4AssayHostLimits,
        kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<ApplicationOutcome>, HostedError> {
        let input = CalibrationInput { base: Input { dataset, key, namespace, host_scope: None },
            applications: [Some(application), None] };
        dispatch_owned(self, model, None, input, false, options, limits, host, kernel, cancellation,
            move |replay, _, input, control| calibration::apply_with_checkpoint::<HostedError>(
                replay, input.applications[0].as_ref().ok_or(HostedError::CompletionMissing)?,
                &input.base.key, &input.base.namespace, options, limits, &mut || checkpoint(control)))
    }

    /// Compare raw and/or calibrated arms on the same actual native population.
    /// Each Some artifact belongs to exactly its arm; None explicitly selects
    /// raw reliability. Conservative calibration abstention cannot pass gates.
    /// Both arms retain one deadline, one five-axis allowance and native peak.
    #[allow(clippy::too_many_arguments)]
    pub fn qualify_int4_classification_calibrated(&self, candidate: &ResidentInt4, baseline: &ResidentInt4,
        dataset: Dataset, candidate_application: Option<PreparedApplication>,
        baseline_application: Option<PreparedApplication>, key: CommitmentKey, namespace: String,
        host_scope: String, gates: Gates, options: EvalOptions, limits: metrics::Limits,
        host: Int4AssayHostLimits, kernel: PackedInt4Kernel, cancellation: CancellationToken)
        -> Result<HostedOutput<QualificationOutcome>, HostedError> {
        gates.validate(limits)?;
        native_classify::validate_scope(&namespace, Some(&host_scope))?;
        let input = CalibrationInput { base: Input { dataset, key, namespace, host_scope: Some(host_scope) },
            applications: [candidate_application, baseline_application] };
        dispatch_owned(self, candidate, Some(baseline), input, false, options, limits, host, kernel,
            cancellation, move |candidate, baseline, input, control| {
                calibration::compare_with_checkpoint::<HostedError>(candidate,
                    baseline.ok_or(HostedError::CompletionMissing)?, input.applications[0].as_ref(),
                    input.applications[1].as_ref(), &input.base.key, &input.base.namespace,
                    input.base.host_scope.as_deref().ok_or(HostedError::CompletionMissing)?,
                    gates, options, limits, &mut || checkpoint(control))
            })
    }
}

/// The operation is derived from owned artifacts, never from an underpriced
/// caller declaration. A missing baseline cannot hide its artifact owner.
fn owned_operation(fit: bool, arms: usize, present: [bool; 2]) -> Result<Int4AssayOperation, HostedError> {
    if !(1..=2).contains(&arms) || (arms == 1 && present[1]) {
        return Err(HostedError::Limits("INT4 Assay calibration owner shape"));
    }
    let artifacts = present.into_iter().filter(|&value| value).count();
    if fit {
        if arms != 1 || artifacts != 0 {
            return Err(HostedError::Limits("INT4 Assay fitting owner shape"));
        }
        Ok(Int4AssayOperation::Fit)
    } else if artifacts == 0 { Ok(Int4AssayOperation::Raw) }
    else { Ok(Int4AssayOperation::Apply { artifacts }) }
}

#[allow(clippy::too_many_arguments)]
fn dispatch_owned<T, F>(engine: &NlpEngine, candidate: &ResidentInt4, baseline: Option<&ResidentInt4>,
    input: CalibrationInput, fit: bool, options: EvalOptions, limits: metrics::Limits,
    host: Int4AssayHostLimits, kernel: PackedInt4Kernel, cancellation: CancellationToken, finalize: F)
    -> Result<HostedOutput<T>, HostedError>
where T: Send + 'static,
    F: FnOnce(&NativeReplay, Option<&NativeReplay>, &CalibrationInput, &mut RunControl)
        -> Result<T, HostedError> + Send + 'static {
    dispatch::preflight(engine, host.native.run)?;
    check_domain(engine, candidate)?;
    if let Some(baseline) = baseline { check_domain(engine, baseline)?; }
    let arms = 1 + usize::from(baseline.is_some());
    validate_inputs(&input.base.dataset, &input.base.key, &input.base.namespace, options, limits, arms)?;
    let operation = owned_operation(fit, arms, input.applications.each_ref().map(Option::is_some))?;
    for application in input.applications.iter().flatten() { application.validate_limits(limits)?; }
    let widths = [candidate.model_identity().group_width(),
        baseline.map_or(candidate.model_identity().group_width(), |model| model.model_identity().group_width())];
    let required = host.requirements_for_operation(&widths[..arms], limits, operation)?;
    let lease = engine.resources().acquire_lease();
    let input = allocate(Pending::reserve(&lease, MemoryClass::JobBuffers, required.input_bytes)?, || Ok(input))?;
    let kv = Pending::reserve(&lease, MemoryClass::KvPages, required.native.kv_bytes)?;
    let workspace = Pending::reserve(&lease, MemoryClass::ActivationScratch, required.workspace_bytes)?;
    let output = output_claim(&lease, limits.max_output_bytes as u64, 0)?;
    let candidate = candidate.clone(); let baseline = baseline.cloned();
    dispatch::run(engine, host.native.run, cancellation, move |control| {
        let input = input;
        let native_claims = (kv, workspace);
        let result = allocate(output, || {
            checkpoint(control)?;
            // Authentication and exact source binding allocate only inside the
            // admitted envelope, before either native engine executes.
            for application in input.value.applications.iter().flatten() {
                application.validate_dataset(&input.value.base.dataset, &input.value.base.key,
                    &input.value.base.namespace, limits)?;
                checkpoint(control)?;
            }
            let mut work = Int4Work::default();
            let candidate = replay_arm(&candidate, &input.value.base.dataset, host, limits,
                required.native, kernel, &mut work, control)?;
            let baseline = baseline.as_ref().map(|model| replay_arm(model, &input.value.base.dataset,
                host, limits, required.native, kernel, &mut work, control)).transpose()?;
            checkpoint(control)?;
            let result = finalize(&candidate, baseline.as_ref(), &input.value, control)?;
            checkpoint(control)?;
            Ok(result)
        })?;
        drop(input); drop(native_claims); drop(lease);
        Ok(GuardedOutput::new(result.value, result._memory))
    })
}

#[cfg(test)] mod tests;
