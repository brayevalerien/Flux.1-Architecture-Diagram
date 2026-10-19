#import "lib.typ": *

// From pixels to the 64 channel tokens the transformer sees, and back
#let latent-space() = cetz.canvas({
  import cetz.draw: *
  setup()

  // encode
  io((0, 0), "in", raw("image"), side: "east")
  frozen((1.4, 0), "enc", [AE encoder\ (frozen)], h: 2.1, w: 0.85)
  op((3.9, 0), "sub", "−")
  op((4.9, 0), "mul", "×")
  block((6.2, 0), "pack", [Pack 2×2], h: 1.8)
  io((8.6, 0), "out", raw("img"), side: "west")
  arrow("in.east", "enc.west")
  arrow("enc.east", "sub.west")
  arrow("sub.east", "mul.west")
  arrow("mul.east", "pack.west")
  arrow("pack.east", "out.west")
  arrow((3.9, -1), "sub.south")
  arrow((4.9, -1), "mul.south")
  note((3.9, -1.05), [0.1159], side: "north")
  note((4.9, -1.05), [0.3611], side: "north")
  note((2.75, 0), [16 × 2H × 2W])
  note((1.4, -1.2), [mean of the moments], side: "north")
  note(("pack.east", 50%, "out.west"), [HW × 64])

  // decode
  let y = -3.2
  io((0, y), "z", [x#sub[0]], side: "east")
  block((1.4, y), "unpack", [Unpack 2×2], h: 1.8)
  op((2.7, y), "div", "÷")
  op((3.7, y), "add", "+")
  frozen((6.3, y), "dec", [AE decoder\ (frozen)], h: 2.1, w: 0.85)
  io((7.8, y), "img", raw("image"), side: "west")
  arrow("z.east", "unpack.west")
  arrow("unpack.east", "div.west")
  arrow("div.east", "add.west")
  arrow("add.east", "dec.west")
  arrow("dec.east", "img.west")
  arrow((2.7, y - 1), "div.south")
  arrow((3.7, y - 1), "add.south")
  note((2.7, y - 1.05), [0.3611], side: "north")
  note((3.7, y - 1.05), [0.1159], side: "north")
  note((5, y), [16 × 2H × 2W])
})

// One sampling step. The guidance is distilled into the model, so there is a single forward pass and no CFG
#let sampling() = cetz.canvas({
  import cetz.draw: *
  setup()

  io((0, 0), "x", [x#sub[i]], side: "east")
  stage((2.8, 0), "flux", [Flux.1 \[dev\]], w: 2.2, h: 1)
  wire("x.east", (0.9, 0))
  arrow((0.9, 0), "flux.west")
  io((2.2, 1.6), "pos", raw("prompt"), side: "south")
  io((3.4, 1.6), "g", raw("guidance"), side: "south")
  arrow("pos.south", (2.2, 0.5))
  arrow("g.south", (3.4, 0.5))
  io((2.8, -1.6), "t", [t#sub[i]], side: "north")
  arrow("t.north", (2.8, -0.5))
  op((5.3, 0), "dt", "×")
  arrow("flux.east", "dt.west")
  note((4.45, 0), [v])
  arrow((5.3, -1.1), "dt.south")
  note((5.3, -1.15), [t#sub[i+1] − t#sub[i]], side: "north")
  op((6.6, 0), "step", "+")
  arrow("dt.east", "step.west")
  wire((0.9, 0), (0.9, 2.6), (6.6, 2.6))
  arrow((6.6, 2.6), "step.north")
  io((7.9, 0), "next", [x#sub[i+1]], side: "west")
  arrow("step.east", "next.west")
})
