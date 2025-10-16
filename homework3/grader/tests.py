import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .grader import Case, Grader

# A different split will be used for grading
DATA_SPLIT = "data/val"
INPUT_SHAPE = (1, 3, 64, 64)
LABEL_NAMES = ["background", "kart", "pickup", "nitro", "bomb", "projectile"]


@dataclass
class TracerStats:
    num_linear_in_shortest_path: int = 0
    num_linear_relu_in_longest_path: int = 0
    has_relu: bool = False

    @classmethod
    def trace(cls, model, input_sample):
        visited = {}
        unknown_layers = set()

        def graph_tracer(node):
            if node in visited:
                return visited[node]
            visited[node] = TracerStats(100000, 0, False)

            if node.kind() in ["prim::Param"]:
                return TracerStats(0, 0, True)
            elif node.kind() in ["aten::linear", "aten::_convolution"]:
                stats = graph_tracer(list(node.inputs())[0].node())
                visited[node] = TracerStats(
                    stats.num_linear_in_shortest_path + 1,
                    stats.num_linear_relu_in_longest_path + int(stats.has_relu),
                    False,
                )
                return visited[node]
            elif node.kind() in ["aten::relu_", "aten::relu", "aten::gelu_", "aten::gelu", "aten::elu_", "aten::elu", "aten::leaky_relu_", "aten::leaky_relu"]:
                stats = graph_tracer(list(node.inputs())[0].node())
                visited[node] = TracerStats(
                    stats.num_linear_in_shortest_path, stats.num_linear_relu_in_longest_path, True
                )
                return visited[node]
            elif node.kind() in ["aten::add_", "aten::add", "aten::cat"]:
                all_stats = [graph_tracer(i.node()) for i in node.inputs() if i.node().kind() != "prim::Constant"]
                visited[node] = TracerStats(
                    min(s.num_linear_in_shortest_path for s in all_stats),
                    max(s.num_linear_relu_in_longest_path for s in all_stats),
                    True,
                )
                return visited[node]
            elif list(node.inputs()):
                unknown_layers.add(node.kind())
                visited[node] = graph_tracer(list(node.inputs())[0].node())
                return visited[node]
            return visited[node]

        traced_model = torch.jit.trace(model, input_sample)
        graph = traced_model.graph
        torch._C._jit_pass_inline(graph)
        stats = graph_tracer(graph.return_node())
        return stats


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path: str):
        """
        Pairs of images and labels (int) for classification
        """
        to_tensor = transforms.ToTensor()

        self.data = []

        with open(Path(dataset_path, "labels.csv"), newline="") as f:
            for fname, label, _ in csv.reader(f):
                if label in LABEL_NAMES:
                    image = Image.open(Path(dataset_path, fname))
                    label_id = LABEL_NAMES.index(label)

                    self.data.append((to_tensor(image), label_id))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_data(dataset_path: str, num_workers: int = 0, batch_size: int = 128, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        SuperTuxDataset(dataset_path), num_workers=num_workers, batch_size=batch_size, shuffle=shuffle, drop_last=True
    )


# ClassificationLoss is provided from HW2 - no testing needed


class MLPResidualGrader(Grader):
    """MLP with Residual Connections"""

    KIND = "mlp_deep_residual"
    ACC_RANGE = 0.5, 0.8

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data = list(load_data(DATA_SPLIT, num_workers=2))
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    @torch.inference_mode()
    def accuracy(
        self, model: torch.nn.Module, min_accuracy: float = 0.5, max_accuracy: float = 1.0
    ) -> tuple[float, str]:
        """
        Returns the accuracy of the model normalized to [0, 1]
        """
        model.to(self.device)

        accuracy_list = []

        for img, label in self.data:
            pred = model(img.to(self.device)).detach().cpu()
            acc = (pred.argmax(dim=1).type_as(label) == label).float()
            accuracy_list.extend(acc.numpy())

        accuracy = torch.as_tensor(accuracy_list).mean()
        accuracy_normalized = (accuracy - min_accuracy) / (max_accuracy - min_accuracy)

        return accuracy_normalized.clip(0, 1).item(), f"accuracy = {accuracy.item():0.2f}"

    def check_model(self, model):
        stats = TracerStats.trace(model, torch.rand(*INPUT_SHAPE))
        assert stats.num_linear_relu_in_longest_path >= 4, "Model not deep enough (need at least 4 layers)"
        assert (
            stats.num_linear_relu_in_longest_path > stats.num_linear_in_shortest_path
        ), "Model does not contain residual connections"

    @Case(score=10, timeout=500)
    def test_model(self):
        """Model"""
        model = self.module.load_model(self.KIND, with_weights=False)
        self.check_model(model)

    @Case(score=30, timeout=10000)
    def test_accuracy(self):
        """Accuracy"""
        model = self.module.load_model(self.KIND, with_weights=True)
        self.check_model(model)
        return self.accuracy(model, *self.ACC_RANGE)

    @Case(score=2, timeout=10000, extra_credit=True)
    def test_accuracy_extra(self):
        """Accuracy: Extra Credit"""
        model = self.module.load_model(self.KIND, with_weights=True)
        self.check_model(model)
        return self.accuracy(model, self.ACC_RANGE[1], self.ACC_RANGE[1] + 0.05)


class ViTGrader(MLPResidualGrader):
    """Vision Transformer"""

    KIND = "vit"
    ACC_RANGE = 0.6, 0.8

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._core_test_results = {"transformer_impl": None, "attention_mechanism": None}

    def check_model(self, model):
        input_sample = torch.rand(*INPUT_SHAPE)
        try:
            output = model(input_sample)
            assert output.shape == (1, 6), f"Model output shape should be (1, 6), got {output.shape}"
        except Exception as e:
            assert False, f"Model forward pass failed: {e}"

    @Case(score=10, timeout=500)
    def test_no_builtin_transformer(self):
        """No Built-in Transformer Usage"""
        try:
            model = self.module.load_model(self.KIND, with_weights=False)

            # Check that students don't use PyTorch's built-in transformer blocks
            forbidden_modules = [
                torch.nn.TransformerEncoder,
                torch.nn.TransformerEncoderLayer,
                torch.nn.TransformerDecoder,
                torch.nn.TransformerDecoderLayer,
                torch.nn.Transformer
            ]

            for name, module in model.named_modules():
                for forbidden_class in forbidden_modules:
                    if isinstance(module, forbidden_class):
                        assert False, f"Found forbidden built-in transformer module: {forbidden_class.__name__}. You must implement the transformer block from scratch."

            # Also check if TransformerBlock is actually implemented (not just NotImplementedError)
            has_transformer_block = False
            for name, module in model.named_modules():
                if 'TransformerBlock' in str(type(module)):
                    has_transformer_block = True
                    # Try to run it to make sure it's implemented
                    try:
                        test_input = torch.randn(1, 10, 256)  # (batch, seq_len, embed_dim)
                        _ = module(test_input)
                    except NotImplementedError:
                        assert False, "TransformerBlock is not implemented - you need to implement it from scratch"
                    except Exception:
                        # Other exceptions are okay - might be due to wrong input size, etc.
                        pass
                    break

            assert has_transformer_block, "No TransformerBlock found in the model - you need to use and implement the TransformerBlock class"

            # Mark core test as passed
            self._core_test_results["transformer_impl"] = True
        except Exception as e:
            # Mark core test as failed
            self._core_test_results["transformer_impl"] = False
            raise e

    @Case(score=10, timeout=500)
    def test_information_isolation(self):
        """Information Isolation (Attention Mechanism)"""
        try:
            model = self.module.load_model(self.KIND, with_weights=False)

            # Find the TransformerBlock in the model
            transformer_block = None
            for name, module in model.named_modules():
                if 'TransformerBlock' in str(type(module)):
                    transformer_block = module
                    break

            assert transformer_block is not None, "No TransformerBlock found in model"

            # Get the actual embedding dimension from the transformer block parameters
            embed_dim = None
            for name, param in transformer_block.named_parameters():
                if 'weight' in name and len(param.shape) >= 2:
                    # Use the last dimension of the first weight matrix found
                    embed_dim = param.shape[-1]
                    break

            if embed_dim is None:
                embed_dim = 256  # Fallback if we can't determine it

            seq_len = 4     # 4 tokens for clear isolation test

            input_clean = torch.zeros(1, seq_len, embed_dim)

            input_noisy = input_clean.clone()
            torch.manual_seed(42)  # Reproducible noise
            input_noisy[0, 1:, :] = torch.randn(seq_len - 1, embed_dim) * 0.1  

            # Get outputs for both inputs
            with torch.no_grad():
                try:
                    output_clean = transformer_block(input_clean)
                    output_noisy = transformer_block(input_noisy)
                except Exception as e:
                    assert False, f"TransformerBlock failed: {e}"

            assert output_clean.shape == output_noisy.shape, f"Output shapes don't match: {output_clean.shape} vs {output_noisy.shape}"

            tolerance = 1e-3  # Threshold for detecting changes

            output_diffs = []
            for pos in range(seq_len):
                diff = torch.max(torch.abs(output_clean[0, pos, :] - output_noisy[0, pos, :])).item()
                output_diffs.append(diff)
            pos_0_changed = output_diffs[0] > tolerance

            assert pos_0_changed, f"Position 0 output unchanged ({output_diffs[0]:.6f}) despite noise in other positions. This suggests no attention mechanism - with attention, all positions should be affected by changes in any position."

            positions_changed = sum(1 for diff in output_diffs if diff > tolerance)
            assert positions_changed >= 2, f"Only {positions_changed} positions changed significantly. With attention, multiple positions should be affected by input changes."

            # Mark core test as passed
            self._core_test_results["attention_mechanism"] = True
        except Exception as e:
            # Mark core test as failed
            self._core_test_results["attention_mechanism"] = False
            raise e

    @Case(score=10, timeout=500)
    def test_model(self):
        """Model"""
        # Check if core tests passed
        if False in self._core_test_results.values():
            assert False, "Core transformer tests failed - you must implement TransformerBlock correctly with attention mechanism to receive any points for ViT"

        model = self.module.load_model(self.KIND, with_weights=False)
        self.check_model(model)

    @Case(score=30, timeout=10000)
    def test_accuracy(self):
        """Accuracy"""
        # Check if core tests passed
        if False in self._core_test_results.values():
            assert False, "Core transformer tests failed - you must implement TransformerBlock correctly with attention mechanism to receive any points for ViT"

        model = self.module.load_model(self.KIND, with_weights=True)
        self.check_model(model)
        return self.accuracy(model, *self.ACC_RANGE)

    @Case(score=2, timeout=10000, extra_credit=True)
    def test_accuracy_extra(self):
        """Accuracy: Extra Credit"""
        # Check if core tests passed
        if False in self._core_test_results.values():
            assert False, "Core transformer tests failed - you must implement TransformerBlock correctly with attention mechanism to receive any points for ViT"

        model = self.module.load_model(self.KIND, with_weights=True)
        self.check_model(model)
        return self.accuracy(model, self.ACC_RANGE[1], self.ACC_RANGE[1] + 0.05)

