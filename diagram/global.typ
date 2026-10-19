#import "lib.typ": *

#let global-structure() = cetz.canvas({
  import cetz.draw: *
  setup()

  let main-y = -8
  let up-y = main-y + 0.3
  let low-y = main-y - 0.3
  let vec-y = -5.6
  let pe-y = -10.8
  let ds = (13.6, 17, 22.2)
  let ss = (29.2, 32.6, 37.8)
  let last-x = 41.4
  let sw = 2.4
  let half = sw / 2

  // TIMESTEP AND GUIDANCE
  io((0, 0), "in-t", raw("timesteps"), side: "east")
  io((0, -2), "in-g", raw("guidance"), side: "east")
  block((2, -1), "sin", "Sinusoidal emb.", h: 3)
  arrow("in-t.east", (2 - block-w / 2, 0))
  arrow("in-g.east", (2 - block-w / 2, -2))
  block((4, 0), "mlp-t", "MLP Emb.")
  block((4, -2), "mlp-g", "MLP Emb.")
  arrow((2 + block-w / 2, 0), "mlp-t.west")
  arrow((2 + block-w / 2, -2), "mlp-g.west")
  note((3, 0), [256])
  note((3, -2), [256])
  note((2, -2.6), [t × 1000, (cos, sin)], side: "north")

  // PROMPT, the two frozen text encoders
  io((-6.4, -5.6), "prompt", raw("prompt"), side: "east")
  frozen((-3.6, -4), "clip", [CLIP ViT-L/14\ (frozen)], h: 2.3, w: 0.85)
  frozen((-3.6, -7.2), "t5", [T5 v1.1 XXL\ (frozen)], h: 2.3, w: 0.85)
  wire("prompt.east", (-4.8, -5.6))
  wire((-4.8, -4), (-4.8, -7.2))
  arrow((-4.8, -4), "clip.west")
  arrow((-4.8, -7.2), "t5.west")
  note((-3.6, -5.2), [pooled], side: "north")
  note((-3.6, -8.4), [last hidden state], side: "north")
  io((-1.4, -4), "y", raw("y"))
  io((-1.4, -7.2), "txt", raw("txt"))
  arrow("clip.east", "y.west")
  arrow("t5.east", "txt.west")
  block((4, -4), "mlp-y", "MLP Emb.")
  arrow("y.east", "mlp-y.west")
  note((0.6, -4), [768])

  // vec
  op((6.4, -2), "vec-sum", "+")
  arrow("mlp-t.east", (6.4, 0), "vec-sum.north")
  arrow("mlp-g.east", "vec-sum.west")
  arrow("mlp-y.east", (6.4, -4), "vec-sum.south")
  wire("vec-sum.east", (7.2, -2), (7.2, vec-y), (last-x, vec-y))
  note((8, vec-y), [vec])
  note((8.6, vec-y - 0.05), [3072], side: "north")

  // txt and img tokens
  block((4, -7.2), "txt-in", "Linear")
  arrow("txt.east", "txt-in.west")
  note((1.3, -7.2), [512 × 4096])
  wire("txt-in.east", (11, -7.2), (11, up-y))
  arrow((11, up-y), (ds.first() - half, up-y))
  note((8, -7.2), [512 × 3072])

  io((0, -9.6), "in-img", raw("img"), side: "east")
  block((4, -9.6), "img-in", "Linear")
  arrow("in-img.east", "img-in.west")
  note((1.3, -9.6), [HW × 64])
  wire("img-in.east", (11, -9.6), (11, low-y))
  arrow((11, low-y), (ds.first() - half, low-y))
  note((8, -9.6), [HW × 3072])

  // positions, text ids first as in the code
  io((0, -11.6), "in-txt-ids", raw("txt_ids"), side: "east")
  io((0, -13.4), "in-img-ids", raw("img_ids"), side: "east")
  op((2.4, -12.5), "ids-cat", "CAT")
  arrow("in-txt-ids.east", ((), "-|", "ids-cat"), "ids-cat.north")
  arrow("in-img-ids.east", ((), "-|", "ids-cat"), "ids-cat.south")
  block((4, -12.5), "embed", "EmbedND")
  arrow("ids-cat.east", "embed.west")
  wire("embed.east", (6, -12.5), (6, pe-y), (ss.last(), pe-y))
  note((7, pe-y), [P.E.])

  // DOUBLE STREAM STACK
  for (i, x) in ds.enumerate() {
    stage((x, main-y), "ds" + str(i), [DoubleStream\ Block], w: sw)
    arrow((x, vec-y), "ds" + str(i) + ".north")
    arrow((x, pe-y), "ds" + str(i) + ".south")
  }
  let ds-dots = 19.6
  dots((ds-dots, main-y))
  arrow((ds-dots, vec-y), (ds-dots, main-y + 0.65))
  arrow((ds-dots, pe-y), (ds-dots, main-y - 0.65))
  for (a, b) in ((ds.at(0) + half, ds.at(1) - half), (ds.at(1) + half, ds-dots - 0.6), (ds-dots + 0.6, ds.at(2) - half)) {
    arrow((a, up-y), (b, up-y))
    arrow((a, low-y), (b, low-y))
  }
  note(((ds.at(0) + ds.at(1)) / 2, up-y), [txt])
  note(((ds.at(0) + ds.at(1)) / 2, low-y - 0.05), [img], side: "north")
  content(((ds.first() + ds.last()) / 2, vec-y + 1.6), text(size: 15pt)[N = 19])

  // join the two streams, text first
  op((25, main-y), "cat", "CAT")
  arrow((ds.last() + half, up-y), (25, up-y), "cat.north")
  arrow((ds.last() + half, low-y), (25, low-y), "cat.south")
  note((24.1, up-y), [txt])
  note((24.1, low-y - 0.05), [img], side: "north")

  // SINGLE STREAM STACK
  for (i, x) in ss.enumerate() {
    stage((x, main-y), "ss" + str(i), [SingleStream\ Block], w: sw)
    arrow((x, vec-y), "ss" + str(i) + ".north")
    arrow((x, pe-y), "ss" + str(i) + ".south")
  }
  let ss-dots = 35.2
  dots((ss-dots, main-y))
  arrow((ss-dots, vec-y), (ss-dots, main-y + 0.65))
  arrow((ss-dots, pe-y), (ss-dots, main-y - 0.65))
  arrow("cat.east", "ss0.west")
  note(("cat.east", 50%, "ss0.west"), [(512 + HW) × 3072])
  arrow("ss0.east", "ss1.west")
  arrow("ss1.east", (ss-dots - 0.6, main-y))
  arrow((ss-dots + 0.6, main-y), "ss2.west")
  content(((ss.first() + ss.last()) / 2, vec-y + 1.6), text(size: 15pt)[M = 38])

  // tail, the text tokens are dropped
  let tail-x = 39.8
  block((last-x, main-y), "last", "LastLayer")
  arrow("ss2.east", "last.west")
  note((tail-x + 0.7, main-y), [img])
  arrow((tail-x, main-y), (tail-x, main-y - 1.3))
  content((tail-x, main-y - 1.55), text(size: 16pt)[×])
  note((tail-x - 0.08, main-y - 0.6), [txt], side: "east")
  arrow((last-x, vec-y), "last.north")
  io((43.8, main-y), "out", raw("v"), side: "west")
  arrow("last.east", "out.west")
  note((42.8, main-y), [HW × 64])
})
