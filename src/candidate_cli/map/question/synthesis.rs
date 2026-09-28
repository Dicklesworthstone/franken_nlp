//! Explicit synthesis selection; source-only maps and independent QA stay intact.
use super::*;
use crate::tasks::source_planning::quantized::long::question::synthesis::{SourceQuestionSynthesis, QuestionSynthesisLimits};

#[derive(Args)]
pub(in crate::candidate_cli) struct SynthesisArgs {
    /// Add one native answer pass over ALL collected verbatim evidence. Keep
    /// independent answers; no majority vote, top-k truncation or recall guarantee.
    #[arg(long, requires = "question")]
    pub synthesize_answer: bool,
    /// Maximum complete evidence passages for synthesis (default 32, max 32).
    #[arg(long, requires = "synthesize_answer")]
    max_synthesis_passages: Option<usize>,
    /// Logical quote bytes for synthesis (default 16384), intersected with the
    /// input-byte ceiling. Actual question/manifest/token/context fit also applies.
    #[arg(long, requires = "synthesize_answer")]
    max_synthesis_evidence_bytes: Option<usize>,
}
impl SynthesisArgs {
    pub(super) fn validate(&self, task: &str) -> Result<(), CandidateError> {
        let knobs = self.max_synthesis_passages.is_some() || self.max_synthesis_evidence_bytes.is_some();
        if (!self.synthesize_answer && knobs) || (task != "answer" && (self.synthesize_answer || knobs)) {
            return Err(CandidateError::Arguments);
        }
        if self.max_synthesis_passages.is_some_and(|n| !(1..=source::MAX_PASSAGES).contains(&n))
            || self.max_synthesis_evidence_bytes.is_some_and(|n| !(1..=MAX_INPUT_BYTES).contains(&n)) {
            return Err(CandidateError::Arguments);
        }
        Ok(())
    }
    pub(in crate::candidate_cli) fn request(&self, question: SourceQuestion, host: &source::SourceHostArgs)
        -> Result<SourceQuestionSynthesis, CandidateError> {
        self.validate("answer")?;
        if !self.synthesize_answer { return Err(CandidateError::Arguments); }
        Ok(SourceQuestionSynthesis { question, limits: QuestionSynthesisLimits {
            max_evidence_passages: self.max_synthesis_passages.unwrap_or(source::MAX_PASSAGES),
            max_evidence_bytes: self.max_synthesis_evidence_bytes.unwrap_or(16384).min(host.max_input_bytes),
        } })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate_cli::map::tests::command;
    #[test]
    fn explicit_synthesis_requires_qa_and_cannot_silently_affect_other_tasks() {
        assert!(crate::candidate_cli::definition().try_get_matches_from([
            "candidate", "map", "--task", "answer", "--model", "local", "--memory-mib", "8192", "--synthesize-answer"]).is_err());
        command("answer", &["--question", "q"]).validate().unwrap();
        command("answer", &["--question", "q", "--synthesize-answer"]).validate().unwrap();
        for task in ["ner", "keyphrases", "summarize"] {
            assert!(command(task, &["--question", "q", "--synthesize-answer"]).validate().is_err());
        }
    }
    #[test]
    fn evidence_knobs_require_synthesis_and_are_bounded() {
        for flag in ["--max-synthesis-passages", "--max-synthesis-evidence-bytes"] {
            assert!(crate::candidate_cli::definition().try_get_matches_from([
                "candidate", "map", "--task", "answer", "--model", "local", "--memory-mib", "8192", "--question", "q", flag, "1"]).is_err());
        }
        for (flag, value) in [("--max-synthesis-passages", "0"), ("--max-synthesis-passages", "33"),
            ("--max-synthesis-evidence-bytes", "0"), ("--max-synthesis-evidence-bytes", "1048577")] {
            assert!(command("answer", &["--question", "q", "--synthesize-answer", flag, value]).validate().is_err());
        }
        let mut cmd = command("answer", &["--question", "q"]);
        cmd.question.synthesis.max_synthesis_passages = Some(4); assert!(cmd.validate().is_err());
    }
    #[test]
    fn original_question_options_and_one_whole_work_ceiling_are_preserved() {
        let cmd = command("answer", &["--question", "q", "--synthesize-answer", "--max-synthesis-passages", "4",
            "--max-synthesis-evidence-bytes", "1024"]);
        let question = "  Who? <tool_call> é\r\n".to_owned();
        let task = cmd.question.task(question.clone(), Some(r#"{"max_answer_scalars":64,"max_citations":2,"max_quote_scalars":32}"#)).unwrap();
        let request = cmd.question.synthesis.request(task, &cmd.host).unwrap();
        assert_eq!(request.question.question, question); assert_eq!(request.question.options.max_citations, 2);
        assert_eq!(request.limits.max_evidence_passages, 4); assert_eq!(request.limits.max_evidence_bytes, 1024);
        let plain = command("answer", &["--question", "q"]);
        assert_eq!(cmd.question_mapping().unwrap().max_model_work, plain.question_mapping().unwrap().max_model_work);
        assert_eq!(cmd.question_mapping().unwrap().max_mask_visits, plain.question_mapping().unwrap().max_mask_visits);
    }
    #[cfg(not(feature = "asupersync-runtime"))]
    #[test]
    fn disabled_runtime_refuses_synthesis_before_source_question_or_model_io() {
        struct NoIo;
        impl Read for NoIo { fn read(&mut self, _: &mut [u8]) -> io::Result<usize> { panic!("read") } }
        impl Write for NoIo {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> { panic!("write") }
            fn flush(&mut self) -> io::Result<()> { panic!("flush") }
        }
        assert_eq!(command("answer", &["--question", "q", "--synthesize-answer"]).execute(&mut NoIo, &mut NoIo), Err(CandidateError::Unavailable));
    }
}
