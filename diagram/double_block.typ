#import "lib.typ": *

// DoubleStream block. txt residual stream on top, img at the bottom, both branches point inward to
// the joint attention in the middle. Each stream has its own modulation, coming from outside its spine.
#let double-block() = cetz.canvas({
  import cetz.draw: *
  setup()

  let spine = 6.6
  let branch = 4.6
  let mid = 5.6
  let step = 0.5
  let r1 = 0.2

  // the img half is the txt half mirrored, s flips every y
  let stream(s, name) = {
    let y(v) = s * v
    let p = name + "-"

    // residual stream
    io((-1.4, y(spine)), p + "in", raw(name), side: "east")
    io((25.4, y(spine)), p + "out", raw(name), side: "west")
    op((14.6, y(spine)), p + "res1", "+")
    op((23.4, y(spine)), p + "res2", "+")
    arrow(p + "in.east", p + "res1.west")
    arrow(p + "res1.east", p + "res2.west")
    arrow(p + "res2.east", p + "out.west")

    // attention input
    block((1.8, y(branch)), p + "norm1", "LayerNorm")
    arrow((0.8, y(spine)), (0.8, y(branch)), p + "norm1.west")
    op((3, y(branch)), p + "scale1", "×")
    op((4, y(branch)), p + "shift1", "+")
    arrow(p + "norm1.east", p + "scale1.west")
    arrow(p + "scale1.east", p + "shift1.west")
    block((5.2, y(branch)), p + "qkv", "Linear", h: 3.4)
    arrow(p + "shift1.east", p + "qkv.west")
    let out = 5.2 + block-w / 2
    // Q is the line closest to the middle so the three wires never cross
    for (v, l, x) in ((branch - 1.3, "Q", 8.3), (branch, "K", 9), (branch + 1.3, "V", 9.7)) {
      note((5.8, y(v)), l)
      if l != "V" {
        block((6.8, y(v)), p + "norm-" + l, "RMSNorm", h: 1.15, size: 6.5pt)
        arrow((out, y(v)), p + "norm-" + l + ".west")
        wire(p + "norm-" + l + ".east", (x, y(v)))
      } else {
        wire((out, y(v)), (x, y(v)))
      }
      arrow((x, y(v)), (x, y(0.7)))
    }

    // back from the split: projection, gate, residual
    let split-out = 0.6
    wire((12.25, y(split-out)), (12.8, y(split-out)), (12.8, y(branch)))
    block((13.6, y(branch)), p + "proj", "Linear")
    arrow((12.8, y(branch)), p + "proj.west")
    note((12.85, y(split-out)), raw(name), side: if s > 0 { "south-west" } else { "north-west" })
    op((14.6, y(mid)), p + "gate1", "×", r: r1)
    arrow(p + "proj.east", (14.6, y(branch)), (14.6, y(mid - r1)))
    arrow((14.6, y(mid + r1)), (14.6, y(spine - 0.28)))

    // MLP
    block((16.6, y(branch)), p + "norm2", "LayerNorm")
    arrow((15.8, y(spine)), (15.8, y(branch)), p + "norm2.west")
    op((17.8, y(branch)), p + "scale2", "×")
    op((18.8, y(branch)), p + "shift2", "+")
    arrow(p + "norm2.east", p + "scale2.west")
    arrow(p + "scale2.east", p + "shift2.west")
    block((20, y(branch)), p + "fc1", "Linear")
    block((21.2, y(branch)), p + "act", "GELU (tanh)", h: 1.9)
    block((22.4, y(branch)), p + "fc2", "Linear")
    arrow(p + "shift2.east", p + "fc1.west")
    arrow(p + "fc1.east", p + "act.west")
    arrow(p + "act.east", p + "fc2.west")
    note((20.6, y(branch)), [12288])
    op((23.4, y(mid)), p + "gate2", "×", r: r1)
    arrow(p + "fc2.east", (23.4, y(branch)), (23.4, y(mid - r1)))
    arrow((23.4, y(mid + r1)), (23.4, y(spine - 0.28)))

    // MODULATION, outside the spine, outputs listed outward from the spine in code order
    let m-y0 = spine + 0.9
    let names = ("shift1", "scale1", "gate1", "shift2", "scale2", "gate2")
    let ys = range(6).map(i => m-y0 + i * step)
    block((0, y((ys.first() + ys.last()) / 2)), p + "mod", "Modulation", h: 3.1)
    io((-1.4, y((ys.first() + ys.last()) / 2)), p + "vec", raw("vec"), side: "east")
    arrow(p + "vec.east", p + "mod.west")
    let edge = block-w / 2
    for (i, n) in names.enumerate() {
      let v = ys.at(i)
      let kind = n.slice(0, n.len() - 1)
      let k = n.last()
      note((edge + 0.08, y(v) + 0.02), [#kind#sub(k)], side: "south-west")
      if kind == "shift" {
        let x = if k == "1" { 4 } else { 18.8 }
        wire((edge, y(v)), (x, y(v)))
        vwire(x, y(v), y(branch + 0.28), hops: (y(spine),))
      } else if kind == "scale" {
        let x = if k == "1" { 3 } else { 17.8 }
        op((x, y(mid)), p + "one" + k, "+", r: r1)
        wire((edge, y(v)), (x, y(v)))
        vwire(x, y(v), y(mid + r1), hops: (y(spine), y(v - step)))
        arrow((x, y(mid - r1)), (x, y(branch + 0.28)))
        arrow((x - 0.8, y(mid)), (x - r1, y(mid)))
        note((x - 0.85, y(mid)), "1", side: "east")
      } else {
        let x = if k == "1" { 15.2 } else { 24 }
        let target = if k == "1" { 14.6 } else { 23.4 }
        wire((edge, y(v)), (x, y(v)))
        vwire(x, y(v), y(mid), hops: (y(spine),), head: false)
        arrow((x, y(mid)), (target + r1, y(mid)))
      }
    }
  }

  stream(1, "txt")
  stream(-1, "img")

  // JOINT ATTENTION, text first in the sequence
  rect((7.9, -0.7), (10.1, 0.7), name: "cat")
  content("cat", normal[CAT])
  block((11.3, 0), "attn", [RoPE\ Attention], h: 2.4, w: 0.75)
  for (v, l) in ((0.45, "Q"), (0, "K"), (-0.45, "V")) {
    arrow((10.1, v), (11.3 - 0.375, v))
    note((10.5, v - 0.02), l)
  }
  io((11.3, 2.4), "pe", [P.E.], side: "south")
  arrow("pe.south", "attn.north")
  block((12.05, 0), "split", "split", h: 2)
  wire("attn.east", "split.west")
  note((11.3, -1.25), [24 heads × 128], side: "north")
})
