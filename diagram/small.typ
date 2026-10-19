#import "lib.typ": *

#let mlp-emb() = cetz.canvas({
  import cetz.draw: *
  setup()
  io((0, 0), "in", raw("x"), side: "east")
  block((1.3, 0), "l1", "Linear")
  block((2.6, 0), "act", "SiLU")
  block((3.9, 0), "l2", "Linear")
  io((5.2, 0), "out", none, side: "west")
  arrow("in.east", "l1.west")
  arrow("l1.east", "act.west")
  arrow("act.east", "l2.west")
  arrow("l2.east", (5.2, 0))
  note((4.55, 0), [3072])
})

// one per block (two in a DoubleStream block, one per stream), SiLU before the Linear
#let modulation() = cetz.canvas({
  import cetz.draw: *
  setup()
  io((0, 0), "in", raw("vec"), side: "east")
  block((1.3, 0), "act", "SiLU")
  block((2.6, 0), "l", "Linear")
  block((4.6, 0), "c", "chunk", h: 2.4)
  arrow("in.east", "act.west")
  arrow("act.east", "l.west")
  arrow("l.east", "c.west")
  note((3.6, 0), [6 × 3072])
  let names = ([shift#sub[1]], [scale#sub[1]], [gate#sub[1]], [shift#sub[2]], [scale#sub[2]], [gate#sub[2]])
  for (i, n) in names.enumerate() {
    let y = 1 - i * 0.4
    arrow((4.6 + block-w / 2, y), (5.7, y))
    note((5.75, y - 0.01), n, side: "west")
  }
})

#let last-layer() = cetz.canvas({
  import cetz.draw: *
  setup()
  io((0, 0), "in", raw("img"), side: "east")
  block((1.3, 0), "norm", "LayerNorm")
  op((2.7, 0), "scale", "×")
  op((3.9, 0), "shift", "+")
  block((5.1, 0), "l", "Linear")
  io((6.5, 0), "out", raw("v"), side: "west")
  arrow("in.east", "norm.west")
  arrow("norm.east", "scale.west")
  arrow("scale.east", "shift.west")
  arrow("shift.east", "l.west")
  arrow("l.east", "out.west")
  note((0.55, 0), [3072])
  note((5.85, 0), [64])

  io((-1.2, 3), "vec", raw("vec"), side: "east")
  block((0.1, 3), "act", "SiLU")
  block((1.3, 3), "m", "Linear")
  block((2.3, 3), "c", "chunk", h: 1.2)
  arrow("vec.east", "act.west")
  arrow("act.east", "m.west")
  arrow("m.east", "c.west")
  op((2.7, 1.2), "one", "+", r: 0.2)
  // chunk order is shift then scale, shift is the wire closest to the spine
  wire((2.3 + block-w / 2, 2.7), (3.9, 2.7))
  arrow((3.9, 2.7), "shift.north")
  wire((2.3 + block-w / 2, 3.3), (2.7, 3.3))
  vwire(2.7, 3.3, 1.4, hops: (2.7,))
  arrow("one.south", "scale.north")
  arrow((1.9, 1.2), "one.west")
  note((1.85, 1.2), "1", side: "east")
  note((3.95, 2.2), [shift], side: "west")
  note((2.75, 2.2), [scale], side: "west")
})
