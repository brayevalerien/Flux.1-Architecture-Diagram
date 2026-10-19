#import "lib.typ": *

#let rope-attention() = cetz.canvas({
  import cetz.draw: *
  setup()
  io((0, 1.2), "q", raw("Q"), side: "east")
  io((0, 0), "k", raw("K"), side: "east")
  io((0, -1.4), "v", raw("V"), side: "east")
  block((1.4, 1.2), "rq", "RoPE", h: 1)
  block((1.4, -0.1), "rk", "RoPE", h: 1)
  arrow("q.east", "rq.west")
  arrow("k.east", (1.4 - block-w / 2, 0))
  io((0, 2.6), "pe", [P.E.], side: "east")
  wire("pe.east", (0.9, 2.6))
  wire((0.9, 2.6), (0.9, -0.85), (1.4, -0.85))
  arrow((0.9, 2.6), (1.4, 2.6), "rq.north")
  arrow((1.4, -0.85), "rk.south")
  op((2.8, 0.6), "qk", "·")
  arrow((1.4 + block-w / 2, 1.2), ((), "-|", "qk"), "qk.north")
  arrow((1.4 + block-w / 2, 0), ((), "-|", "qk"), "qk.south")
  note((2.2, 0), [Kᵀ], side: "south")
  op((3.9, 0.6), "scale", "÷")
  arrow("qk.east", "scale.west")
  arrow((3.9, -0.5), "scale.south")
  note((3.9, -0.55), [√128], side: "north")
  block((5.1, 0.6), "soft", "Softmax")
  arrow("scale.east", "soft.west")
  op((6.3, 0.6), "av", "·")
  arrow("soft.east", "av.west")
  arrow("v.east", ((), "-|", "av"), "av.south")
  arrow("av.east", (7.4, 0.6))
  note((5.1, -0.35), [per head], side: "north")
})

// Flux rotates interleaved pairs (2i, 2i + 1): out = x cos + stack(−x₁, x₀) sin, one frequency per pair
#let rope-apply() = cetz.canvas({
  import cetz.draw: *
  setup()
  io((0, 0), "x", raw("x"), side: "east")
  block((1.5, -1.6), "unbind", [Unbind pairs], h: 1.9)
  arrow((0.7, 0), (0.7, -1.6), "unbind.west")
  let out = 1.5 + block-w / 2
  op((3.1, -0.8), "neg", "−", r: 0.2)
  arrow((out, -2), (2.5, -2), (2.5, -0.8), "neg.west")
  block((4, -1.7), "stack", "Stack", h: 1.3)
  arrow("neg.east", (3.6, -0.8), (3.6, -1.35), (4 - block-w / 2, -1.35))
  wire((out, -1.2), (2.2, -1.2))
  vwire(2.2, -1.2, -2.7, hops: (-2,), head: false)
  arrow((2.2, -2.7), (3.6, -2.7), (3.6, -2.05), (4 - block-w / 2, -2.05))
  note((out + 0.05, -1.2), [x#sub[0]], side: "south-west")
  note((out + 0.05, -2), [x#sub[1]], side: "south-west")
  op((5.2, -1.7), "msin", "×")
  arrow("stack.east", "msin.west")
  op((5.2, 0), "mcos", "×")
  arrow("x.east", "mcos.west")
  op((6.4, 0), "add", "+")
  arrow("mcos.east", "add.west")
  arrow("msin.east", ((), "-|", "add"), "add.south")
  arrow("add.east", (7.4, 0))
  arrow((5.2, 1.1), "mcos.north")
  note((5.2, 1.15), [cos], side: "south")
  arrow((5.2, -2.8), "msin.south")
  note((5.2, -2.85), [sin], side: "north")
})

// EmbedND: one rope per axis, d = 16, 56, 56 dims, then concatenated over the axes
#let embed-nd() = cetz.canvas({
  import cetz.draw: *
  setup()
  io((0, 0), "ids", [(idx, row, col)], side: "east")
  op((1.9, 0), "mul", "×")
  arrow("ids.east", "mul.west")
  arrow((1.9, -1.1), "mul.south")
  note((1.9, -1.15), [ω#sub[j] = 10000#super[−2j/d], j < d/2], side: "north")
  block((3.5, 0.9), "cos", "cos", h: 1)
  block((3.5, -0.9), "sin", "sin", h: 1)
  wire("mul.east", (2.9, 0))
  wire((2.9, 0.9), (2.9, -0.9))
  arrow((2.9, 0.9), "cos.west")
  arrow((2.9, -0.9), "sin.west")
  block((4.9, 0), "rot", [2×2 rotations], h: 2.2)
  arrow("cos.east", (4.4, 0.9), (4.4, 0.6), (4.9 - block-w / 2, 0.6))
  arrow("sin.east", (4.4, -0.9), (4.4, -0.6), (4.9 - block-w / 2, -0.6))
  op((6.3, 0), "cat", "CAT")
  arrow("rot.east", "cat.west")
  arrow("cat.east", (7.6, 0))
  note((7.2, 0), [P.E.])
  note((6.3, -0.4), [over axes], side: "north")
})

#let position-table() = {
  set text(size: 7pt)
  table(
    columns: 4,
    stroke: none,
    align: center,
    inset: (x: 6pt, y: 3pt),
    table.hline(stroke: 0.6pt),
    [token], [axis 0 (16)], [axis 1 (56)], [axis 2 (56)],
    table.hline(stroke: 0.4pt),
    [text], [0], [0], [0],
    [image (r, c)], [0], [r], [c],
    table.hline(stroke: 0.6pt),
  )
}
