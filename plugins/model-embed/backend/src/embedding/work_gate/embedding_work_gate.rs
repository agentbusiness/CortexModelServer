use super::*;

pub(super) fn embedding_work_gate() -> &'static EmbeddingWorkGate {
    EMBEDDING_WORK_GATE.get_or_init(EmbeddingWorkGate::default)
}
