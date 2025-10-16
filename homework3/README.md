# Homework 3

In this homework, we will extend our work from HW2 by implementing **advanced neural network architectures** to classify images from *SuperTuxKart*. You will implement two state-of-the-art models: a deep residual network (ResNet) and a Vision Transformer (ViT).

This homework will require a GPU - if you don't have access to one, you can use Google Colab.

## Setup + Starter Code

The starter code is similar to HW2, but focuses on advanced architectures.

The starter code contains a `data` directory where you'll copy (or symlink) the [SuperTuxKart Classification Dataset](https://www.cs.utexas.edu/~bzhou/dl_class/classification_data.zip).
Unzip the data directly into the homework folder, replacing the existing data directory completely.

Make sure you see the following directories and files inside your main directory
```
homework/
grader/
bundle.py
data/train
data/val
```
You will run all scripts from inside this main directory.

In the `homework` directory, you'll find the following starter code files
- `train.py` - code to train, evaluate and save your models
- `models.py` - where you will implement the two new advanced models
- `logger.py` - complete implementation for logging your model's performance using Tensorboard
- `utils.py` - data loader for the SuperTuxKart dataset

### Data Loader

The data loader is the same as HW2. We use the SuperTuxKart dataset with 6 classes: `background` (0), `kart` (1), `pickup` (2), `nitro` (3), `bomb` (4), and `projectile` (5).

### Local Grader Instructions

You can grade your implementation after any part of the homework by running the following command from the main directory:
- `python3 -m grader homework -v` for medium verbosity
- `python3 -m grader homework -vv` to include print statements

## Deep Residual Network (40 pts)

Implement the `MLPClassifierDeepResidual` class in `models.py`. This model should be a deep MLP (at least 4 layers) with **residual connections** to help with the vanishing gradient problem.

### Key Concepts

**Residual Connections**: Instead of learning a mapping H(x), residual blocks learn the residual F(x) = H(x) - x, so that H(x) = F(x) + x. This allows gradients to flow directly through skip connections, enabling training of much deeper networks.

The key insight is that it's easier to learn small changes (residuals) than to learn the complete transformation from scratch.

### Implementation Requirements

- At least 4 hidden layers
- Residual connections (skip connections) between layers
- Input and output dimensions must match for residual connections to work
- You can use `torch.nn.ModuleList` to store your layers

### Training

Train your network using:
```bash
python3 -m homework.train --model_name mlp_deep_residual
```

### Hints/Tips

- Design your architecture so that residual connections are possible (matching dimensions)
- Residual connections typically add the input to the output: `output = layer(input) + input`
- You may need to tune your learning rate `lr` and `batch_size`
- Consider the depth vs. performance trade-off

### Relevant Operations

- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [torch.nn.ModuleList](https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html)

## Vision Transformer (ViT) (60 pts)

Implement the `ViTClassifier` class in `models.py`. This model applies the Transformer architecture to computer vision by treating image patches as tokens.

### Key Concepts

**Vision Transformer**: ViT splits an image into fixed-size patches, linearly embeds each patch, adds positional embeddings, and feeds the sequence to a standard Transformer encoder. A classification token is used to aggregate information for the final prediction.

**Architecture Components**:
1. **Patch Embedding**: Split 64×64 image into patches (e.g., 8×8 patches = 64 tokens)
2. **Linear Projection**: Flatten each patch and project to embedding dimension
3. **Positional Encoding**: Add learnable positional embeddings to patch embeddings
4. **Classification Token**: Special `[CLS]` token for aggregating information
5. **Transformer Encoder**: Multi-head self-attention and MLP blocks
6. **Classification Head**: Final linear layer for prediction

### Implementation Requirements

- Patch embedding that converts 64×64 images to sequence of patches
- Learnable positional embeddings
- At least one Transformer encoder block with multi-head self-attention
- Classification token (`[CLS]`) for final prediction
- You can use the provided `PatchEmbedding` class, but **must implement `TransformerBlock` from scratch**
- Do NOT use PyTorch's built-in transformer modules (e.g., `nn.TransformerEncoder`, `nn.TransformerEncoderLayer`)

### Training

Train your network using:
```bash
python3 -m homework.train --model_name vit
```

### Hints/Tips

- Start with a simple implementation: 8×8 patches, single attention layer
- Use the provided `PatchEmbedding` class for converting images to patch sequences
- Implement the `TransformerBlock` class following the hints in the docstring
- The classification token should be prepended to the patch sequence
- Positional embeddings are typically added, not concatenated
- You can create multiple transformer layers with `nn.ModuleList`
- You may need different hyperparameters (learning rate, batch size) than previous models

### Relevant Operations

- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)
- [torch.nn.MultiheadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
- [torch.nn.LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)
- [torch.nn.GELU](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html)
- [torch.Tensor.unfold](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.unfold)


## Submission

Once you finished the assignment, create a submission bundle using
```bash
python3 bundle.py homework [YOUR UT ID]
```
and submit the zip file on canvas. Please note that the maximum file size our grader accepts is **40MB**. Please keep your model compact.

Please double-check that your zip file was properly created, by grading it again
```bash
python3 -m grader [YOUR UT ID].zip
```


