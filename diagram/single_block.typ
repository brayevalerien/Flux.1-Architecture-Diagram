#import "lib.typ": *

// SingleStream block: attention and MLP run in parallel out of one fused Linear (Dehghani et al., 2023)
#let single-block() = cetz.canvas({
  import cetz.draw: *
  setup()

  let br = 2.6
  let mid = 1.3
  let r1 = 0.2
  let qkv-y = br + 2.6
  let d = 1.3
  let mlp-y = br - 0.6

  // residual stream
  io((-1.4, 0), "in", raw("x"), side: "east")
  io((17.2, 0), "out", raw("x"), side: "west")
  op((15.2, 0), "res", "+")
  arrow("in.east", "res.west")
  arrow("res.east", "out.west")

  block((1.8, br), "norm", "LayerNorm")
  arrow((0.8, 0), (0.8, br), "norm.west")
  op((3, br), "scale", "×")
  op((4, br), "shift", "+")
  arrow("norm.east", "scale.west")
  arrow("scale.east", "shift.west")
  block((5.2, br), "l1", "Linear")
  arrow("shift.east", "l1.west")
  block((6.5, br), "split", "split", h: 2.4)
  arrow("l1.east", "split.west")
  note((5.85, br), [21504])

  // attention path
  let out = 6.5 + block-w / 2
  wire((out, br + 0.8), (7.1, br + 0.8), (7.1, qkv-y))
  block((7.9, qkv-y), "chunk", "split", h: 2 * d + 0.8)
  arrow((7.1, qkv-y), "chunk.west")
  note((out + 0.05, br + 0.8), [qkv], side: "south-west")
  block((9.4, qkv-y + d), "qn", "RMSNorm", h: 1.15, size: 6.5pt)
  block((9.4, qkv-y), "kn", "RMSNorm", h: 1.15, size: 6.5pt)
  block((10.9, qkv-y), "attn", [RoPE\ Attention], h: 2 * d + 0.8, w: 0.75)
  let cout = 7.9 + block-w / 2
  for (dy, l, n) in ((d, "Q", "qn"), (0, "K", "kn")) {
    arrow((cout, qkv-y + dy), n + ".west")
    arrow(n + ".east", (10.9 - 0.375, qkv-y + dy))
    note((8.55, qkv-y + dy), l)
  }
  arrow((cout, qkv-y - d), (10.9 - 0.375, qkv-y - d))
  note((8.55, qkv-y - d), "V")
  io((10.9, qkv-y + d + 1.4), "pe", [P.E.], side: "south")
  arrow("pe.south", "attn.north")

  // mlp path
  block((9.4, mlp-y), "act", "GELU (tanh)", h: 1.9)
  arrow((out, br - 0.6), "act.west")
  note((out + 0.05, mlp-y), [mlp], side: "south-west")
  note((8.2, mlp-y - 0.05), [12288], side: "north")

  // join, attention first
  op((12.2, br + 0.6), "cat", "CAT")
  arrow("attn.east", (12.2, qkv-y), "cat.north")
  arrow("act.east", (12.2, mlp-y), "cat.south")
  block((13.8, br + 0.6), "l2", "Linear")
  arrow("cat.east", "l2.west")
  note((13, br + 0.6), [15360])
  op((15.2, mid), "gate", "×", r: r1)
  arrow("l2.east", (15.2, br + 0.6), (15.2, mid + r1))
  arrow((15.2, mid - r1), (15.2, 0.28))

  // modulation from below, outputs listed outward from the spine in code order
  let step = 0.5
  let ys = range(3).map(i => -0.9 - i * step)
  block((0, (ys.first() + ys.last()) / 2), "mod", "Modulation", h: 1.9)
  io((-1.4, (ys.first() + ys.last()) / 2), "vec", raw("vec"), side: "east")
  arrow("vec.east", "mod.west")
  let edge = block-w / 2
  for (i, kind) in ("shift", "scale", "gate").enumerate() {
    let v = ys.at(i)
    note((edge + 0.08, v + 0.02), kind, side: "south-west")
    if kind == "shift" {
      wire((edge, v), (4, v))
      vwire(4, v, br - 0.28, hops: (0,))
    } else if kind == "scale" {
      op((3, mid), "one", "+", r: r1)
      wire((edge, v), (3, v))
      vwire(3, v, mid - r1, hops: (0, v + step))
      arrow((3, mid + r1), (3, br - 0.28))
      arrow((2.2, mid), (3 - r1, mid))
      note((2.15, mid), "1", side: "east")
    } else {
      wire((edge, v), (15.9, v))
      vwire(15.9, v, mid, hops: (0,), head: false)
      arrow((15.9, mid), (15.2 + r1, mid))
    }
  }
})
