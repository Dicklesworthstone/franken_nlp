//! Rebuild the exact source/question plans before execution. Opaque task
//! preflight types and all native identities remain the only plan authority.
use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Proof {
    Independent(Int4QuestionPreflight), Synthesis(Int4QuestionSynthesisPreflight), Hierarchy(Int4QuestionHierarchyPreflight),
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Expected { proof: Proof, pub(super) work: Int4Work, pub(super) masks: u64 }
impl Expected {
    pub(super) fn discovery(self) -> Int4QuestionPreflight {
        match self.proof { Proof::Independent(p) => p, Proof::Synthesis(p) => p.discovery(),
            Proof::Hierarchy(p) => p.discovery() }
    }
    pub(super) fn verify(self, output: &Int4QuestionDocumentRun) -> Result<(), BatchItemFailure> {
        match (self.proof, output) {
            (Proof::Independent(p), Int4QuestionDocumentRun::Independent(run)) => p.verify_completed(run),
            (Proof::Synthesis(p), Int4QuestionDocumentRun::Synthesis(run)) => p.verify_completed(run),
            (Proof::Hierarchy(p), Int4QuestionDocumentRun::Hierarchy(run)) => p.verify_completed(run),
            _ => return Err(BatchItemFailure::fatal(BatchCode::InvalidExecution)),
        }.map_err(|e| map_failure(e, true))
    }
}
pub(super) enum NativePlan<'s> {
    Independent(PreparedInt4Question<'s>), Synthesis(PreparedInt4QuestionSynthesis<'s>),
    Hierarchy(PreparedInt4QuestionHierarchy<'s>),
}
impl NativePlan<'_> {
    pub(super) fn expected(&self) -> Result<Expected, BatchItemFailure> {
        let (proof, work, masks) = match self {
            Self::Independent(plan) => {
                let p = plan.preflight_metadata(); (Proof::Independent(p), p.planned_work(), p.reserved_mask_visits())
            }
            Self::Synthesis(plan) => {
                let p = plan.preflight_metadata();
                (Proof::Synthesis(p), p.reserved_model_work().map_err(|e| map_failure(e, true))?,
                    p.reserved_mask_visits().map_err(|e| map_failure(e, true))?)
            }
            Self::Hierarchy(plan) => {
                let p = plan.preflight_metadata();
                (Proof::Hierarchy(p), p.reserved_model_work(), p.reserved_mask_visits())
            }
        };
        Ok(Expected { proof, work, masks })
    }
    pub(super) fn max_kv_bytes(&self) -> u64 {
        match self { Self::Independent(p) => p.max_kv_bytes(), Self::Synthesis(p) => p.max_kv_bytes(),
            Self::Hierarchy(p) => p.max_kv_bytes() }
    }
    fn max_result_bytes(&self) -> u64 {
        match self { Self::Independent(p) => p.max_result_bytes(), Self::Synthesis(p) => p.max_result_bytes(),
            Self::Hierarchy(p) => p.max_result_bytes() }
    }
    fn identities<C: DecodeStepControl>(&self, control: &mut C) -> Result<Vec<ExecutionIdentity>, BatchItemFailure> {
        match self { Self::Independent(p) => copy_identities(p.execution_identities(), control),
            Self::Synthesis(p) => copy_identities(p.execution_identities(), control),
            Self::Hierarchy(p) => copy_identities(p.execution_identities(), control) }
    }
    pub(super) fn preflight<C: DecodeStepControl>(&self, identities: &[ExecutionIdentity],
        model: &Int4ModelIdentity, control: &mut C) -> Result<(), BatchItemFailure> {
        if identities.is_empty() || identities.len() != self.expected()?.discovery().native_chunks()
            || self.identities(control)? != identities || model.group_width() != self.expected()?.discovery().group_width() {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        for identity in identities {
            poll(control)?;
            constrained_int4::check_model(identity, model).map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        }
        poll(control)
    }
    pub(super) fn cohort_requirements(&self, context: usize) -> Result<Int4SourceMapCohortRequirement, BatchItemFailure> {
        match self { Self::Independent(p) => p.cohort_requirements(context), Self::Synthesis(p) => p.cohort_requirements(context),
            Self::Hierarchy(p) => p.cohort_requirements(context) }
            .map_err(|e| map_failure(e, true))
    }
    pub(super) fn phase_requirements(&self, context: usize, schedule: Option<Int4HierarchyCohortSchedule>)
        -> Result<Int4SourceMapCohortRequirement, BatchItemFailure> {
        match schedule {
            None => self.cohort_requirements(context),
            Some(schedule) => match self {
                Self::Hierarchy(p) => p.hierarchy_cohort_requirements(context, schedule).map_err(|e| map_failure(e, true)),
                _ => Err(BatchItemFailure::fatal(BatchCode::Admission)),
            },
        }
    }
}
impl Int4QuestionDocumentBatchPlanner {
    pub(super) fn plan<'s, C: DecodeStepControl>(&self, source: &'s str, question: &str,
        control: &mut C, fatal: bool) -> Result<NativePlan<'s>, BatchItemFailure> {
        poll(control)?;
        let reject = |code| if fatal { BatchItemFailure::fatal(code) } else { BatchItemFailure::reject(code) };
        let c = &self.config;
        if source.is_empty() || source.len() > c.mapping.chunks.max_input_bytes
            || question.is_empty() || question.len() > c.max_question_bytes {
            return Err(reject(BatchCode::DocumentLimit));
        }
        if !question.chars().any(|ch| !ch.is_whitespace()) { return Err(reject(BatchCode::Planning)); }
        let context = PlanContext::new(&self.identity, c.budget)
            .map_err(|_| BatchItemFailure::fatal(BatchCode::Admission))?;
        let mut owned_question = String::new();
        owned_question.try_reserve_exact(question.len()).map_err(|_| BatchItemFailure::fatal(BatchCode::Allocation))?;
        owned_question.push_str(question);
        let question = SourceQuestion { question: owned_question, options: c.options, verification: c.verification };
        let plan = match (c.synthesis, c.hierarchy) {
            (None, None) => NativePlan::Independent(self.planner.plan_int4_question_with_control(source, &question,
                c.budget, &context, c.planning, c.mapping, self.group_width, control).map_err(|e| map_failure(e, fatal))?),
            (Some(limits), None) => NativePlan::Synthesis(self.planner.plan_int4_question_synthesis_with_control(source,
                &SourceQuestionSynthesis { question, limits }, c.budget, &context, c.planning, c.mapping,
                self.group_width, control).map_err(|e| map_failure(e, fatal))?),
            (Some(limits), Some(hierarchy)) => NativePlan::Hierarchy(self.planner.plan_int4_question_hierarchy_with_control(
                source, &SourceQuestionSynthesis { question, limits }, hierarchy, c.budget, &context,
                c.planning, c.mapping, self.group_width, control).map_err(|e| map_failure(e, fatal))?),
            (None, Some(_)) => return Err(BatchItemFailure::fatal(BatchCode::InvalidLimits)),
        };
        let expected = plan.expected()?; let discovery = expected.discovery();
        if discovery.group_width() != self.group_width || discovery.chunk_count() == 0
            || discovery.chunk_count() > c.mapping.chunks.max_chunks || discovery.native_chunks() == 0
            || discovery.native_chunks() > discovery.chunk_count()
            || discovery.source_span().byte_start != 0 || discovery.source_span().byte_end != source.len()
            || plan.max_kv_bytes() != c.budget.max_kv_bytes || plan.max_result_bytes() != self.max_result_bytes()
            || !positive(expected.work) || !fits(expected.work, c.mapping.max_model_work)
            || expected.masks == 0 || expected.masks > c.mapping.max_mask_visits {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        poll(control)?;
        Ok(plan)
    }
    pub(super) fn snapshot<C: DecodeStepControl>(&self, plan: &NativePlan<'_>, control: &mut C)
        -> Result<(Expected, Vec<ExecutionIdentity>), BatchItemFailure> {
        let expected = plan.expected()?; let identities = plan.identities(control)?;
        if identities.len() != expected.discovery().native_chunks() || identities.is_empty() {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        Ok((expected, identities))
    }
    pub(super) fn compare<C: DecodeStepControl>(&self, plan: &NativePlan<'_>,
        prepared: &PreparedInt4QuestionDocument, control: &mut C) -> Result<(), BatchItemFailure> {
        self.owns(prepared)?;
        if plan.expected()? != prepared.expected || plan.identities(control)? != prepared.identities
            || plan.max_kv_bytes() != self.config.budget.max_kv_bytes || plan.max_result_bytes() != self.max_result_bytes() {
            return Err(BatchItemFailure::fatal(BatchCode::Admission));
        }
        poll(control)
    }
}
fn copy_identities<'a, C: DecodeStepControl>(identities: impl ExactSizeIterator<Item = &'a ExecutionIdentity>,
    control: &mut C) -> Result<Vec<ExecutionIdentity>, BatchItemFailure> {
    let mut copied = Vec::new();
    copied.try_reserve_exact(identities.len()).map_err(|_| BatchItemFailure::fatal(BatchCode::Allocation))?;
    for identity in identities {
        poll(control)?; check_identity_size(identity).map_err(BatchItemFailure::fatal)?;
        copied.push(identity.clone());
    }
    Ok(copied)
}
