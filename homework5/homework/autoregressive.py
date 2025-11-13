import abc

import torch
import torch.nn as nn


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    # Written with the help of ChatGPT
    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        self.d_latent = d_latent
        self.n_tokens = n_tokens

        self.max_seq_len = 1024

        self.token_emb = nn.Embedding(n_tokens, d_latent)
        self.pos_emb = nn.Embedding(self.max_seq_len, d_latent)

        self.start_token_emb = nn.Parameter(torch.zeros(1, 1, d_latent))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_latent,
            nhead=8,
            dim_feedforward=4 * d_latent,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4,
        )

        self.output_proj = nn.Linear(d_latent, n_tokens)

        self._causal_mask = None
        self._causal_mask_len = None

    # Written with the help of ChatGPT
    def _get_causal_mask(self, L: int, device: torch.device) -> torch.Tensor:
        """
        Return a [L, L] causal mask so that position t can only attend to <= t.
        """
        if self._causal_mask is None or self._causal_mask_len != L or self._causal_mask.device != device:
            mask = torch.nn.Transformer.generate_square_subsequent_mask(L).to(device)
            self._causal_mask = mask
            self._causal_mask_len = L
        return self._causal_mask

    # Written with the help of ChatGPT
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        x: (B, h, w) integer tokens

        Returns:
            logits: (B, h, w, n_tokens) – logits over next token at each position
            extra:  dict of extra stats (empty here)
        """
        B, h, w = x.shape
        L = h * w
        device = x.device

        x_seq = x.view(B, L)

        tok_emb = self.token_emb(x_seq)

        if L > self.max_seq_len:
            raise ValueError(f"Sequence length {L} exceeds max_seq_len {self.max_seq_len}")
        positions = torch.arange(L, device=device).unsqueeze(0)
        pos_emb = self.pos_emb(positions)

        emb = tok_emb + pos_emb

        start = self.start_token_emb.expand(B, 1, self.d_latent)
        emb_shifted = torch.cat([start, emb[:, :-1, :]], dim=1)

        causal_mask = self._get_causal_mask(L, device=device)

        hidden = self.transformer(emb_shifted, mask=causal_mask)

        logits = self.output_proj(hidden)

        logits = logits.view(B, h, w, self.n_tokens)

        extra: dict[str, torch.Tensor] = {}
        return logits, extra

    # Written with the help of ChatGPT
    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        """
        Autoregressively generate a batch of token grids of shape (B, h, w).

        We fill tokens left-to-right, top-to-bottom:
            pos 0 = (0,0), pos 1 = (0,1), ..., pos w-1 = (0,w-1),
            pos w = (1,0), etc.
        """
        if device is None:
            device = next(self.parameters()).device

        L = h * w
        if L > self.max_seq_len:
            raise ValueError(f"Sequence length {L} exceeds max_seq_len {self.max_seq_len}")

        seq = torch.zeros(B, h, w, dtype=torch.long, device=device)

        for t in range(L):
            i = t // w
            j = t % w

            logits, _ = self.forward(seq)

            logits_t = logits[:, i, j, :]

            probs_t = torch.softmax(logits_t, dim=-1)
            next_token = torch.multinomial(probs_t, num_samples=1).squeeze(-1)

            seq[:, i, j] = next_token

        return seq
