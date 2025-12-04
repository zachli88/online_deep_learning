from typing import overload

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class BaseLLM:
    def __init__(self, checkpoint=checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.device = device

    def format_prompt(self, question: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": question},
        ]

        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )



    def parse_answer(self, answer: str) -> float:
        """
        Parse the <answer></answer> tag and return a float.
        This function is somewhat robust to output errors (e.g. missing </answer> tags).
        """
        try:
            return float(answer.split("<answer>")[1].split("</answer>")[0])
        except (IndexError, ValueError):
            return float("nan")

    def generate(self, prompt: str) -> str:
        """
        (Optional) Implement this method first and then implement batched_generate below.
        It is much easier to implement generation without batching.

        The overall flow is the same:
        - apply format_prompt to the input prompt
        - tokenize the prompt with self.tokenizer
        - call self.model.generate
        - decode the outputs with self.tokenizer.decode

        """
        # formatted = self.format_prompt(prompt)

        # # 2. Tokenize
        # inputs = self.tokenizer(
        #     formatted,
        #     return_tensors="pt"
        # ).to(self.device)

        # # 3. Generate
        # with torch.no_grad():
        #     output_ids = self.model.generate(
        #         **inputs,
        #         max_new_tokens=50,
        #         do_sample=False,               # greedy decode
        #         eos_token_id=self.tokenizer.eos_token_id,
        #     )

        # # 4. Decode ONLY the generated portion
        # #    (slice off the input prompt tokens)
        # generated_ids = output_ids[0][len(inputs["input_ids"][0]):]

        # return self.tokenizer.decode(
        #     generated_ids,
        #     skip_special_tokens=True
        # )
        return self.batched_generate([prompt])[0]   # If you feel confident, just use this line of code and move straight to batched_generate.

    @overload
    def batched_generate(
        self,
        prompts: list[str],
        num_return_sequences: None = None,
        temperature: float = 0,
    ) -> list[str]:
        """
        Batched version of `generate` method.
        This version returns a single generation for each prompt.
        """

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: int, temperature: float = 0
    ) -> list[list[str]]:
        """
        Batched version of `generate` method.
        This version returns a list of generation for each prompt.
        """

    def batched_generate(
        self,
        prompts: list[str],
        num_return_sequences: int | None = None,
        temperature: float = 0,
    ) -> list[str] | list[list[str]]:

        from tqdm import tqdm

        # ------------------------------------------------------
        # Micro-batching for large prompt arrays
        # ------------------------------------------------------
        micro_batch_size = 32
        if len(prompts) > micro_batch_size:
            return [
                r
                for idx in tqdm(
                    range(0, len(prompts), micro_batch_size),
                    desc=f"LLM Running on Micro Batches {micro_batch_size}",
                )
                for r in self.batched_generate(
                    prompts[idx : idx + micro_batch_size],
                    num_return_sequences,
                    temperature,
                )
            ]

        # ------------------------------------------------------
        # Format prompts
        # ------------------------------------------------------
        formatted_prompts = [self.format_prompt(p) for p in prompts]

        # ------------------------------------------------------
        # Tokenize (with left padding!)
        # ------------------------------------------------------
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # ------------------------------------------------------
        # Generation parameters
        # ------------------------------------------------------
        do_sample = temperature > 0
        nseq = 1 if num_return_sequences is None else num_return_sequences

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                num_return_sequences=nseq,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # ------------------------------------------------------
        # Decode generated outputs
        # ------------------------------------------------------
        prompt_len = inputs["input_ids"].shape[1]
        generated_only = output_ids[:, prompt_len:]

        decoded = self.tokenizer.batch_decode(
            generated_only,
            skip_special_tokens=True
        )

        # ------------------------------------------------------
        # Reshape
        # ------------------------------------------------------
        if num_return_sequences is None:
            return decoded

        out = []
        i = 0
        for _ in prompts:
            out.append(decoded[i : i + num_return_sequences])
            i += num_return_sequences

        return out


    def answer(self, *questions) -> list[float]:
        """
        Answer questions given as individual string arguments.
        """
        generations = self.batched_generate(questions)
        return [self.parse_answer(g) for g in generations]


def test_model():
    # The following code simply tests of the BaseLLM is able to complete text.
    # It should produce garbage answers, but it should not crash.
    # In my case it talks about cats eating cats, and dogs being happy.
    testset = ["The cat went up", "The dog went down"]
    model = BaseLLM()
    for t in testset:
        print("testing generate function")
        print("input", t)
        answer = model.generate(t)
        print("output", answer)
    answers = model.batched_generate(testset)
    print(answers)


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model})
