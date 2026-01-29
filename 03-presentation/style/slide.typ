#import "@preview/touying:0.6.1": *
#import "@preview/codelst:2.0.2": sourcecode
#import "@preview/pinit:0.2.2": *
#import themes.metropolis: *
#import "@preview/note-me:0.5.0": *
// #let the_date = datetime(year: 2025, month: 6,day: 24)
#let the_date = datetime.today()

#let myred = rgb(80%,0%,0%)
#let myblue = rgb(0%,40%,58%)
#let myorange = rgb(75%,31%,9%)
#let mygreen = rgb(24%,50%,19%)
#let mylightgreen = rgb(50%,70%,40%)
#let myyellow = rgb(100%,88%,26%)
#let myviolet = rgb(48%,47%,72%)
#let mypurple = rgb(69%,45%,69%)
#let mylightorange = rgb(96%,51%,22%)
#let mypink = rgb(85%,51%,72%)
#let almostblack = rgb("#23373B")
#let hydro-blue = rgb("#295b8c")

#let default_text_size = 24pt

#let get-config() = {
  return (
    config-colors(
      primary: rgb("#609ec5"),
      primary-light: rgb("#c1def0"),
      secondary: rgb("#295b8c"),
      // neutral-lightest: rgb("#609ec5"),
      neutral-dark: rgb("#295b8c"),
      neutral-darkest: rgb("#1c4065"),
    ),
    config-methods(
      init: (self: none, body) => {
        show strong: it => text(fill: almostblack, it)
        body
      },
    ),
    config-common(
      datetime-format: "[month repr:long] [day], [year]",
      // handout: true,
    ),

  )
}

#let title_style(body) = [
  #set text(size: 36pt, weight: "bold")
  #image("mse.png", width: 60%)
  // #place(
  //   top+right,
  //   dy: -2.5em,
  //   dx: 4em,
  //   float: false,
  //   clearance: 0.5em,
  //   image("/resources/img/logos/hydro.png", height: 1.5em)
  // )
  #body
]

#let myglobals(doc) = [
  // #set page(background: image("swiss_universities.png"))
  // #set text(font: "CMU Sans Serif", size: default_text_size)
  // #set text(font: "Ubuntu", size: default_text_size)
  #show footnote.entry: set text(size: 18pt)
  // #show link: underline
  // #show link: it => text(fill: myblue, it)
  #doc]

#let strong(body) = [#set text(weight: "bold")
  #body]

#let red(body) = [#set text(fill: myred)
  #body]

#let green(body) = [#set text(fill: mygreen)
  #body]

#let blue(body) = [#set text(fill: myblue)
  #body]

#let orange(body) = [#set text(fill: myorange)
  #body]

#let purple(body) = [#set text(fill: mypurple)
  #body]

#let code_frame = block.with(
  stroke: 0.1mm + rgb(70%, 70%, 80%),
  inset: 3mm,
  radius: 8pt,
  fill: rgb(94%, 94%, 94%),
)

// Usage example:
// #code(size:16pt,```c
// char name = "Jane";
// printf("Hey %s\n", name);
// ```)
#let code(code, size: 20pt, nb:none, highlight:()) = [#set text(size: size)
  #v(-5mm)
  #set par(justify: false)
  #sourcecode(numbering:nb, frame: code_frame, highlighted: highlight, code)]

// Usage example:
// #items(15mm)[
//   - Item 1
//   - Item 2
//   - Item 3
// ]
#let items(vspace, body) = [
  #par[
    #show list.item: it => {
      it
      v(vspace)
    }
    #body]]

// Usage examples:
// #par(al:right)[
//   blah blah
// ]
// #par(size:16pt,al:center)[
//   blah blah
// ]
#let par(size: default_text_size, al: left, body) = [
  #set align(al)
  #set text(size: size)
  #body
]
