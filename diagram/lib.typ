#import "@preview/cetz:0.5.2"

#let font = ("Helvetica", "Arial")
#let line-stroke = 0.6pt + black

// Sizes in cm, close to the draw.io version of the Flux diagram
#let block-w = 0.42
#let block-h = 1.6
#let op-r = 0.28

#let small(body) = text(size: 6.5pt, body)
#let normal(body) = text(size: 8pt, body)

#let setup() = {
  import cetz.draw: *
  set-style(
    stroke: line-stroke,
    mark: (fill: black, scale: 0.55),
    content: (padding: 0),
  )
}

// Tall narrow block with a vertical label, like Linear, RMSNorm, SiLU...
#let block(pos, name, label, h: block-h, w: block-w, size: 8pt) = {
  import cetz.draw: *
  group(name: name, {
    rect((rel: (-w / 2, -h / 2), to: pos), (rel: (w, h)), name: "box")
    content(pos, angle: 90deg, align(center, text(size: size, label)))
    anchor("west", "box.west")
    anchor("east", "box.east")
    anchor("north", "box.north")
    anchor("south", "box.south")
    anchor("center", pos)
  })
}

// Frozen pretrained models we do not open (Qwen3.5, FLUX.2 VAE)
#let frozen(pos, name, label, h: block-h, w: 0.6) = {
  import cetz.draw: *
  group(name: name, {
    rect((rel: (-w / 2, -h / 2), to: pos), (rel: (w, h)), radius: 0.1, name: "box")
    content(pos, angle: 90deg, align(center, normal(label)))
    anchor("west", "box.west")
    anchor("east", "box.east")
    anchor("north", "box.north")
    anchor("south", "box.south")
    anchor("center", pos)
  })
}

// Wide block with a horizontal label, used for the repeated blocks of the global figure
#let stage(pos, name, label, w: 1.9, h: 1.3) = {
  import cetz.draw: *
  group(name: name, {
    rect((rel: (-w / 2, -h / 2), to: pos), (rel: (w, h)), name: "box")
    content(pos, align(center, text(size: 10pt, label)))
    anchor("west", "box.west")
    anchor("east", "box.east")
    anchor("north", "box.north")
    anchor("south", "box.south")
    anchor("center", pos)
  })
}

// Round operator node (+, ×, CAT...)
#let op(pos, name, sym, r: op-r) = {
  import cetz.draw: *
  circle(pos, radius: r, name: name)
  content(pos, if type(sym) == str and sym.len() > 1 { small(sym) } else { normal(sym) })
}

#let arrow(..pts) = {
  import cetz.draw: *
  line(..pts, mark: (end: "stealth"))
}

#let wire(..pts) = {
  import cetz.draw: *
  line(..pts)
}

// Plain text tensor name at the start or end of a wire
#let io(pos, name, label, side: "center") = {
  import cetz.draw: *
  content(pos, normal(label), name: name, anchor: side, padding: 0.08)
}

#let note(pos, label, side: "south") = {
  import cetz.draw: *
  content(pos, small(label), anchor: side, padding: 0.06)
}

#let dots(pos) = {
  import cetz.draw: *
  content(pos, text(size: 14pt, weight: "bold")[...])
}

// Vertical wire from y0 to y1 (either direction) that jumps over horizontal wires at each y in hops
#let vwire(x, y0, y1, hops: (), r: 0.15, head: true) = {
  import cetz.draw: *
  let dir = if y1 > y0 { 1 } else { -1 }
  let ys = hops.sorted()
  if dir < 0 { ys = ys.rev() }
  let cur = y0
  for c in ys {
    line((x, cur), (x, c - dir * r))
    arc((x, c - r), start: -90deg, stop: 90deg, radius: r)
    cur = c + dir * r
  }
  line((x, cur), (x, y1), mark: if head { (end: "stealth") } else { none })
}
