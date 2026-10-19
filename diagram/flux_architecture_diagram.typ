#import "lib.typ": *
#import "global.typ": global-structure
#import "double_block.typ": double-block
#import "single_block.typ": single-block
#import "small.typ": *
#import "attention.typ": *
#import "pipeline.typ": *

#set document(title: "Flux.1 architecture diagram", author: "Valérien")
#set page(width: auto, height: auto, margin: 1.5cm, fill: white)
#set text(font: font)
#set align(center)

#let fig(body, name, size: 15pt, caption: none) = box(align(center, stack(
  dir: ttb,
  spacing: 0.6cm,
  body,
  text(size: size, name),
  ..if caption != none { (text(size: 7pt, caption),) },
)))

#let row(..figs) = align(center + bottom, stack(dir: ltr, spacing: 2.2cm, ..figs.pos().map(f => align(bottom, f))))

#stack(
  dir: ttb,
  spacing: 2.4cm,
  fig(scale(135%, reflow: true, global-structure()), [Flux.1 global architecture], size: 30pt),
  row(
    fig(double-block(), [DoubleStream block], caption: [separate weights for the txt and img streams, they only meet in the attention]),
    fig(single-block(), [SingleStream block], caption: [attention and MLP in parallel out of one Linear]),
  ),
  row(
    fig(mlp-emb(), [MLP Emb.]),
    fig(modulation(), [Modulation], caption: [two per DoubleStream block (one per stream), one per SingleStream block with only shift, scale and gate (3 × 3072)]),
    fig(last-layer(), [LastLayer], caption: [image tokens only]),
    fig(rope-attention(), [RoPE Attention]),
    fig(rope-apply(), [RoPE], caption: [interleaved pairs (2i, 2i + 1), one frequency per pair]),
  ),
  row(
    fig(stack(dir: ttb, spacing: 0.5cm, embed-nd(), position-table()), [EmbedND], caption: [axis 0 is always 0 here, Kontext sets it to 1 for the reference image]),
    fig(latent-space(), [Latent space]),
    fig(sampling(), [Sampling step], caption: [default: 50 Euler steps, guidance 2.5, schedule shifted with the resolution]),
  ),
  align(left, text(size: 8pt)[
    References
    - source code: github.com/black-forest-labs/flux (`model.py`, `modules/layers.py`, `math.py`, `sampling.py`)
    - first architecture diagram: \@nrehiew on X
    - parallel attention and MLP in the SingleStream blocks: Dehghani et al., _Scaling Vision Transformers to 22 Billion Parameters_, 2023
    - H, W: latent token grid (image size / 16), the AE latent is 2H × 2W, the T5 prompt is always padded to 512 tokens
  ]),
)
