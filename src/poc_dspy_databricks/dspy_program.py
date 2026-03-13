from __future__ import annotations

import dspy


class DomainAnswer(dspy.Signature):
    """Provide a specialized support answer for the invoicing SaaS domain."""

    context = dspy.InputField(desc="Domain policy and constraints")
    question = dspy.InputField(desc="User question")
    answer = dspy.OutputField(desc="Accurate, concise, and actionable answer in English")


class DomainAssistant(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.respond = dspy.Predict(DomainAnswer)

    def forward(self, question: str, context: str):
        return self.respond(question=question, context=context)
