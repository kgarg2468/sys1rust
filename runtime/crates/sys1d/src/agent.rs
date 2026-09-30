//! The real predictor: a `laya_core::Agent` on the MLX backend, plus the warm-up requests
//! run before the server binds (one question, then four) so the first client request does
//! not pay for the first forward's kernel compilation and buffer allocation.

use crate::worker::{Factory, Predictor};
use laya_core::{Agent, BackendOptions};
use serde_json::{json, Value};
use std::path::PathBuf;

pub struct AgentPredictor {
    agent: Agent,
}

impl AgentPredictor {
    /// A factory that loads the checkpoint on the inference thread.
    pub fn factory(dir: PathBuf, opts: BackendOptions) -> Factory {
        Box::new(move || {
            let agent = Agent::load(&dir, &opts, Box::new(laya_mlx::make_backend))?;
            Ok(Box::new(AgentPredictor { agent }) as Box<dyn Predictor>)
        })
    }
}

impl Predictor for AgentPredictor {
    fn predict(&self, state: &Value, questions: &Value) -> laya_core::Result<Value> {
        self.agent.predict(state, questions)
    }

    fn warmup(&self) -> laya_core::Result<()> {
        let state = warmup_state();
        for questions in [warmup_questions_1(), warmup_questions_4()] {
            self.agent.predict(&state, &questions)?;
        }
        Ok(())
    }

    fn engine(&self) -> String {
        self.agent.backend_name()
    }
}

/// A short customer-service state in the shape of the bench workloads.
pub fn warmup_state() -> Value {
    json!("{\"account\":{\"tier\":\"standard\",\"tenure_months\":3},\"thread\":[{\"role\":\"customer\",\"text\":\"My payment did not go through and now I am locked out of my account.\"}]}")
}

pub fn warmup_questions_1() -> Value {
    json!({
        "action": {
            "type": "choice",
            "instructions": "What should the assistant do next with this conversation?",
            "criteria": {
                "answer_directly": "The assistant can resolve this itself.",
                "escalate_to_human": "Hand off to a human agent.",
                "request_information": "More detail is needed from the customer."
            }
        }
    })
}

pub fn warmup_questions_4() -> Value {
    json!({
        "category": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": "billing and refunds", "account": "login and access", "technical": "bugs and outages"}
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["calm", "firm", "angry", "furious"]
        },
        "refund": {
            "type": "noul",
            "instructions": "Is the customer asking for money back?",
            "criteria": {"true": "a refund or credit is requested", "false": "no money is asked for"}
        },
        "action": warmup_questions_1()["action"]
    })
}
