# Diffusion Transformer (DiT) from Scratch — CIFAR-10

A from-scratch **Micro Diffusion Transformer (DiT)** trained in pixel space on CIFAR-10 (32×32×3). No pre-trained VAE — pure pixel-level diffusion with a transformer backbone.

| Spec | Value |
|---|---|
| Dataset | CIFAR-10 (50k images, 32×32×3) |
| Patch size | 4×4 → **64 tokens** |
| Hidden dim | 384 |
| Attention heads | 6 |
| Transformer blocks | 6 (adaLN-Zero) |
| Parameters | ~30M |
| Diffusion steps | T = 1000 (linear β schedule) |
| Loss | MSE (ε-prediction) |

---

## Results

### Generated Samples

<p align="center">
  <img src="assets/image.png" alt="Generated CIFAR-10 Samples" width="500"/>
</p>

### Reverse Diffusion Trajectory

<p align="center">
  <img src="assets/image2.png" alt="Reverse Diffusion Trajectory" width="700"/>
</p>

---

## Architecture Overview

```mermaid
flowchart TD
    subgraph Input
        A[" Noisy Image\n(B, 3, 32, 32)"]
        T[" Timestep t"]
    end

    subgraph Patchify
        A --> P["Conv2d(3→384, k=4, s=4)\n→ (B, 64, 384)"]
        P --> PE["+ 2D Sinusoidal\nPositional Embedding"]
    end

    subgraph Conditioning
        T --> SE["Sinusoidal Encoding"]
        SE --> MLP["MLP: Linear→SiLU→Linear\n→ c ∈ ℝ³⁸⁴"]
    end

    subgraph Transformer["6× adaLN-Zero Transformer Blocks"]
        PE --> BLOCK
        MLP --> MOD["c → MLP → γ₁,β₁,α₁,γ₂,β₂,α₂"]

        subgraph BLOCK["Each Block"]
            direction TB
            IN["Tokens x"] --> LN1["LayerNorm (no affine)"]
            LN1 --> SC1["Scale & Shift (γ₁, β₁)"]
            SC1 --> ATTN["Multi-Head Self-Attention"]
            ATTN --> G1["× α₁ (gate, init=0)"]
            G1 --> R1["+ x (residual)"]
            IN --> R1
            R1 --> LN2["LayerNorm (no affine)"]
            LN2 --> SC2["Scale & Shift (γ₂, β₂)"]
            SC2 --> FFN["Feed-Forward Network\nLinear→GELU→Linear"]
            FFN --> G2["× α₂ (gate, init=0)"]
            G2 --> R2["+ residual"]
            R1 --> R2
        end

        MOD --> SC1
        MOD --> SC2
    end

    subgraph Decode["Final Layer & Unpatchify"]
        R2 --> FL["adaLN + Linear(384→48)"]
        FL --> UP["Reshape (B,64,48)\n→ (B,3,32,32)"]
    end

    UP --> OUT[" Predicted Noise ε_θ\n(B, 3, 32, 32)"]
```

### Training Loop

```mermaid
flowchart LR
    X0["x₀ (clean)"] --> QS["q_sample:\nx_t = √ᾱ_t·x₀ + √(1-ᾱ_t)·ε"]
    E["ε ~ N(0,I)"] --> QS
    T["t ~ U[1,T]"] --> QS
    QS --> DiT["DiT(x_t, t) → ε_θ"]
    DiT --> L["L = MSE(ε, ε_θ)"]
    L --> BP["Backprop\nAdamW"]
    BP -.-> X0
```

### Sampling (Reverse Diffusion)

```mermaid
flowchart LR
    N["x_T ~ N(0,I)"] --> D1["Denoise\nt=999"]
    D1 --> D2["Denoise\nt=998"]
    D2 --> DOTS["⋯"]
    DOTS --> D3["Denoise\nt=1"]
    D3 --> D4["Denoise\nt=0"]
    D4 --> IMG[" Generated\nImage x₀"]
```


---

## How to Run

1. Open `DiT.ipynb` in **Google Colab**
2. Set runtime to **T4 GPU** (`Runtime → Change runtime type → T4`)
3. Click **Run All**
4. Training runs for ~80 epochs with checkpoints every 10
5. After training, sampling cells generate a 4×4 grid of images

---

## Project Structure

```
DiT-from-Scratch/
├── DiT.ipynb              # Complete notebook (all 4 phases)
├── README.MD              # This file
├── assets/                # Result images
├── dit_epoch_80.pt        # Trained model weights
└── cifar-10-batches-py/   # CIFAR-10 dataset (auto-downloaded)
```

---

## References

- [Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) — Peebles & Xie, 2023
- [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239) — Ho et al., 2020
- [An Image is Worth 16x16 Words (ViT)](https://arxiv.org/abs/2010.11929) — Dosovitskiy et al., 2021
