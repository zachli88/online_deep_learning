# file written with the help of chatgpt 5.1
from .base_llm import BaseLLM

class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Convert a question into a chat-style prompt that encourages concise, correct
        chain-of-thought reasoning.

        The model is instructed to:
        - give a brief reasoning,
        - end with the final number inside <answer></answer>,
        - and avoid putting units inside the answer tag.
        """

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a helpful reasoning assistant for unit conversion.\n"
                    "Solve the problem with a brief step-by-step explanation.\n"
                    "Finish with the final numeric result inside <answer>...</answer>.\n"
                    "Do not include units inside the <answer> tag.\n"
                    "Be concise."
                ),
            },

            # High-quality in-context example
            {
                "role": "user",
                "content": "How many grams are in 6 kg?",
            },
            {
                "role": "assistant",
                "content": (
                    "We know 1 kg = 1000 g.\n"
                    "So 6 kg = 6 × 1000 = <answer>6000</answer>"
                ),
            },

            # Actual question to solve
            {
                "role": "user",
                "content": question,
            },
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire
    Fire({"test": test_model, "load": load})
