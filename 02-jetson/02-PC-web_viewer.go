package main

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"

	zmq "github.com/pebbe/zmq4"
	"gocv.io/x/gocv"
)

// --- CONFIGURATION ---
const (
	JetsonIP   = "192.168.37.22"
	ZmqPort    = "5555"
	DatasetDir = "dataset_capture"
	Threshold  = 0.70
)

// --- GLOBAL STATE ---
var (
	// Stores the latest RAW jpeg bytes from Jetson (Clean image for photos)
	latestRawImage []byte
	imgMutex       sync.RWMutex

	// Channel to send processed images (with overlay) to the web stream
	streamChannel = make(chan []byte, 1)
)

// Data structure matches the JSON sent by Jetson
type VisionData struct {
	ProbTarget float64 `json:"prob_target"`
	ImageB64   string  `json:"image_b64"`
	FPS        float64 `json:"jetson_fps"`
}

// HTML Interface
const PAGE = `
<html>
<head>
    <title>Jetson Pilot</title>
    <style>
        body { background: #222; color: #eee; text-align: center; font-family: monospace; overflow: hidden; }
        #container { position: relative; display: inline-block; border: 2px solid #444; margin-top: 20px; }
        img { width: 672px; image-rendering: pixelated; display: block; }
        #flash {
            position: absolute; top: 0; left: 0; width: 100%; height: 100%;
            background: white; opacity: 0; pointer-events: none; transition: opacity 0.1s;
        }
        #log { margin-top: 15px; color: #4CAF50; font-weight: bold; height: 20px; }
    </style>
</head>
<body>
    <h1>Jetson Pilot</h1>
    <div style="color: #aaa;">[SPACE] or CLICK to Save Raw Photo</div>

    <div id="container" onclick="capture()">
        <div id="flash"></div>
        <img src="/video_feed" />
    </div>
    <div id="log">Recording video... Ready.</div>

    <script>
        function capture() {
            const flash = document.getElementById('flash');
            flash.style.opacity = 0.8;
            setTimeout(() => { flash.style.opacity = 0; }, 100);

            fetch('/capture_trigger')
                .then(response => response.text())
                .then(msg => {
                    const log = document.getElementById('log');
                    log.innerText = msg;
                    log.style.opacity = 1;
                    setTimeout(() => { log.style.opacity = 0.5; }, 2000);
                })
                .catch(err => console.error(err));
        }

        document.addEventListener('keydown', function(event) {
            if (event.code === 'Space') {
                event.preventDefault();
                capture();
            }
        });
    </script>
</body>
</html>
`

func main() {
	// 1. Prepare Directory
	if _, err := os.Stat(DatasetDir); os.IsNotExist(err) {
		os.MkdirAll(DatasetDir, 0755)
	}

	// 2. Start ZMQ Logic
	go zmqLoop()

	// 3. Start Web Server
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) { w.Write([]byte(PAGE)) })
	http.HandleFunc("/video_feed", streamHandler)
	http.HandleFunc("/capture_trigger", captureHandler)

	fmt.Println("[System] Collector running on :5000")
	fmt.Printf("[Info]  Photos -> '%s/'\n", DatasetDir)
	log.Fatal(http.ListenAndServe(":5000", nil))
}

func zmqLoop() {
	// Setup ZMQ
	zCtx, _ := zmq.NewContext()
	socket, _ := zCtx.NewSocket(zmq.SUB)
	defer socket.Close()

	addr := fmt.Sprintf("tcp://%s:%s", JetsonIP, ZmqPort)
	fmt.Printf("[Connect] Connecting to %s...\n", addr)
	socket.Connect(addr)
	socket.SetSubscribe("")
	socket.SetConflate(true)

	// Setup Video Writer (Using OpenCV - Real AVI)
	timestamp := time.Now().Format("20060102_150405")
	filename := fmt.Sprintf("recording_%s.avi", timestamp)

	// MJPG is a safe bet for AVI container without complex codec installs
	writer, err := gocv.VideoWriterFile(filename, "MJPG", 20, 224, 224, true)
	if err != nil {
		fmt.Printf("[Error] Could not open video writer: %v\n", err)
	} else {
		fmt.Printf("[Record] Saving video to %s\n", filename)
		defer writer.Close()
	}

	for {
		msg, err := socket.Recv(0)
		if err != nil {
			continue
		}

		var data VisionData
		if err := json.Unmarshal([]byte(msg), &data); err != nil {
			continue
		}

		// Decode Base64
		rawBytes, err := base64.StdEncoding.DecodeString(data.ImageB64)
		if err != nil {
			continue
		}

		// 1. Store RAW bytes for Photo Capture (Thread Safe)
		imgMutex.Lock()
		latestRawImage = rawBytes
		imgMutex.Unlock()

		// 2. Decode to OpenCV Mat for Processing
		img, err := gocv.IMDecode(rawBytes, gocv.IMReadColor)
		if err != nil {
			continue
		}
		if img.Empty() {
			continue
		}

		// --- DRAW OVERLAY (Using OpenCV - Clean Text) ---
		var label string
		var rectCol color.RGBA
		// var textCol color.RGBA

		if data.ProbTarget > Threshold {
			label = fmt.Sprintf("CIBLE:%.0f%%", data.ProbTarget*100)
			// textCol = color.RGBA{0, 255, 0, 0} // Green Text
			rectCol = color.RGBA{0, 50, 0, 0} // Dark Green BG
		} else {
			label = fmt.Sprintf("NOCIBLE:%.0f%%", data.ProbTarget*100)
			// textCol = color.RGBA{0, 0, 255, 0} // Red Text
			rectCol = color.RGBA{0, 0, 50, 0} // Dark Red BG
		}

		// Draw Top Bar Background
		gocv.Rectangle(&img, image.Rect(0, 0, 224, 20), rectCol, -1)

		// Draw Labels
		// FONT_HERSHEY_SIMPLEX ensures the ":" renders correctly
		gocv.PutText(&img, label, image.Pt(5, 15), gocv.FontHersheySimplex, 0.4, color.RGBA{200, 200, 200, 0}, 1)

		fpsText := fmt.Sprintf("FPS:%.1f", data.FPS)
		gocv.PutText(&img, fpsText, image.Pt(150, 15), gocv.FontHersheySimplex, 0.4, color.RGBA{200, 200, 200, 0}, 1)

		// 3. Write Frame to Video File
		if writer.IsOpened() {
			writer.Write(img)
		}

		// 4. Encode to JPEG for Web Stream
		buf, _ := gocv.IMEncode(".jpg", img)
		select {
		case streamChannel <- buf.GetBytes():
		default:
		}

		// Important: Release OpenCV memory manually
		img.Close()
	}
}

// --- HANDLERS ---

func captureHandler(w http.ResponseWriter, r *http.Request) {
	imgMutex.RLock()
	data := latestRawImage // Get the CLEAN bytes
	imgMutex.RUnlock()

	if data == nil {
		w.WriteHeader(http.StatusServiceUnavailable)
		return
	}

	// Generate filename
	files, _ := ioutil.ReadDir(DatasetDir)
	id := len(files)
	filename := fmt.Sprintf("%d.jpg", id)
	path := filepath.Join(DatasetDir, filename)

	// Save RAW data
	if err := ioutil.WriteFile(path, data, 0644); err != nil {
		w.Write([]byte("Error Saving"))
		return
	}

	msg := fmt.Sprintf("Saved: %s", filename)
	fmt.Printf("[Photo] %s\n", msg)
	w.Write([]byte(msg))
}

func streamHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "multipart/x-mixed-replace; boundary=frame")
	for jpgData := range streamChannel {
		fmt.Fprintf(w, "--frame\r\nContent-Type: image/jpeg\r\n\r\n%s\r\n", jpgData)
	}
}
