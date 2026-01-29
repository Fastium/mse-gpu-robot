#import "style/slide.typ": *

#let HANDOUT = false

#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  config-info(
    title: title_style[GPU Robot],
    subtitle: [Robot NVIDIA jetson - GPU project - Grp. 5],
    author: [Rémi H. -- Yann S. -- Jonathan AD],
    date: datetime(year:2026, month:01, day:30),
    institution: [HES-SO Master],
  ),
  footer: self => [],
  ..get-config(),
  config-common(
    handout: HANDOUT,
    // show-notes-on-second-screen: if HANDOUT {none} else {right}
  )
)

#show: myglobals
#title-slide()

// 25 min présentation
// incl. démo 
// incl. 5min install

== Personal Challenge
- work without CUDA3
- no server pictures

== Dataset

taking a picture with the robot through PC-webviewer.go and with personal camera
#grid(
    columns: (1fr, 1fr, 1fr),
  align: center,
  image("img/Image_2025_0005_105_cible.jpg"),
  image("img/Image_2025_0005_215_nocible.jpg"),
  image("img/Image_2025_0005_293_cible.JPG")
  
)




== Cutting high res pictures
#image("diagrams/highrespic.drawio.png")


== Best model


  #grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 50pt,
  align: left,
  
  [
    - Top 5 models by Gemini
      - ResNet-18
      - MobileNet_v2
      // performance and inference speed
  ], [
    - Hyper-parameters
      - Batch size
      - Learning Rate
      - Epoch
  ], [
    - Metrics
      - Accuracy
      - BCE Loss
      - FPS
  ], [
    
  ]
)







// graph accuracy vs loss

// #table(
//   columns: (1fr, 1fr, 1fr),
//   align: center,
//   [*Metrics*], [*Description*], [*Strategy*],
//   [Accuracy], [Binary (True - False)], [Maximize],
//   [Loss BCE], [Probability (Quantity)], [Minimize]
// )



// 8 graphe wave01
#image("res/results-wave01.png")

#align(center + horizon)[
  #image("res/global-comparison.png")  
]


// tableau best model
#table(
  columns: (1fr, 1fr, 1fr, 1fr,),
  align: center,
  [*Model*], [*Batch size*], [*Learning rate*], [*Epochs*],
  
  [mobilenet_v2],
  [16],
  [0,001],
  [40]
)

#linebreak()

#align(center)[
#table(
  columns: (250pt, 250pt),
  align: center + horizon,
  [*Accuracy*], [*BCE *],
  [98.125 %], [0.0588]
)
  
]

== Inference jetson
// Arch: server, controller, viewer
// - No Jupyter Notebook
#image("diagrams/MachineLearningEmbarque.drawio.png")

== Step-up FPS
- ONNX too challenging
$->$ TensorRT $=> 18->30$ FPS

= Demo
- Use case 1 ✅
- Use case 2 🟠
- Use case 3 ❌


== Self-evaluation

$=>$ self-evaluation: 4,5 / 6

