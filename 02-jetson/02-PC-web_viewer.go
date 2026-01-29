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
	JetsonIP     = "192.168.37.22"
	ZmqPort      = "5555"
	DatasetDir   = "dataset_capture"
	Threshold    = 0.70
	CamWidth     = 320
	CamHeight    = 224
	CropSize     = 224
	OffsetLeft   = 0
	OffsetCenter = 48
	OffsetRight  = 95
)

// --- GLOBAL STATE ---
var (
	// Stores the latest RAW jpeg bytes from Jetson (Clean image for photos)
	latestRawImage []byte
	imgMutex       sync.RWMutex

	// Channel to send processed images (with overlay) to the web stream
	streamChannel = make(chan []byte, 1)

	// Recording control
	isRecording     = false
	recordingMutex  sync.RWMutex
	recordingSignal = make(chan bool, 1) // Signal to start/stop recording
)

// Data structure matches the JSON sent by Jetson
type VisionData struct {
	Probs    map[string]float64 `json:"probs"` // "left", "center", "right"
	ImageB64 string             `json:"image_b64"`
	FPS      float64            `json:"jetson_fps"`
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
        .controls { margin-top: 20px; }
        button {
            background: #333; color: #eee; border: 1px solid #555; padding: 10px 20px;
            margin: 5px; cursor: pointer; font-size: 14px; border-radius: 4px;
        }
        button:hover { background: #444; }
        button.active { background: #4CAF50; }
        #recording-indicator {
            display: inline-block; width: 12px; height: 12px; border-radius: 50%;
            background: #666; margin-left: 10px; vertical-align: middle;
        }
        #recording-indicator.active { background: #ff4444; animation: blink 0.5s infinite; }
        @keyframes blink { 0%, 50% { opacity: 1; } 51%, 100% { opacity: 0.5; } }
    </style>
</head>
<body>
    <h1>Jetson Pilot</h1>
    <div style="color: #aaa;">[SPACE] or CLICK to Save Raw Photo</div>

    <div id="container" onclick="capture()">
        <div id="flash"></div>
        <img src="/video_feed" />
    </div>
    <div id="log">Ready.</div>

    <div class="controls">
        <button id="recordBtn" onclick="toggleRecording()">Start Recording</button>
        <span id="recording-indicator"></span>
    </div>

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

        function toggleRecording() {
            fetch('/toggle_recording')
                .then(response => response.text())
                .then(msg => {
                    const log = document.getElementById('log');
                    log.innerText = msg;
                    log.style.opacity = 1;
                    setTimeout(() => { log.style.opacity = 0.5; }, 2000);
                    updateRecordingButton();
                })
                .catch(err => console.error(err));
        }

        function updateRecordingButton() {
            fetch('/recording_status')
                .then(response => response.json())
                .then(data => {
                    const btn = document.getElementById('recordBtn');
                    const indicator = document.getElementById('recording-indicator');
                    if (data.recording) {
                        btn.innerText = 'Stop Recording';
                        btn.classList.add('active');
                        indicator.classList.add('active');
                    } else {
                        btn.innerText = 'Start Recording';
                        btn.classList.remove('active');
                        indicator.classList.remove('active');
                    }
                })
                .catch(err => console.error(err));
        }

        // Check recording status on page load and update periodically
        updateRecordingButton();
        setInterval(updateRecordingButton, 500);

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
	http.HandleFunc("/toggle_recording", toggleRecordingHandler)
	http.HandleFunc("/recording_status", recordingStatusHandler)

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

	var writer *gocv.VideoWriter
	var recordingStartTime time.Time

	offsets := map[string]int{
		"left":   OffsetLeft,
		"center": OffsetCenter,
		"right":  OffsetRight,
	}

	// Signal initial state
	recordingSignal <- false

	for {
		// Check for recording state change
		select {
		case shouldRecord := <-recordingSignal:
			recordingMutex.Lock()
			isRecording = shouldRecord
			recordingMutex.Unlock()

			if shouldRecord {
				// Start recording
				timestamp := time.Now().Format("20060102_150405")
				filename := fmt.Sprintf("recording_%s.avi", timestamp)
				recordingStartTime = time.Now()

				var err error
				writer, err = gocv.VideoWriterFile(filename, "MJPG", 20, CamWidth, CamHeight, true)
				if err != nil {
					fmt.Printf("[Error] Could not open video writer: %v\n", err)
				} else {
					fmt.Printf("[Record] Started recording to %s\n", filename)
				}
			} else {
				// Stop recording
				if writer != nil && writer.IsOpened() {
					writer.Close()
					duration := time.Since(recordingStartTime).Round(time.Second)
					fmt.Printf("[Record] Stopped recording (duration: %v)\n", duration)
				}
			}
		default:
		}

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

		// Draw Top Bar Background
		gocv.Rectangle(&img, image.Rect(0, 0, CamWidth, 20), rectCol, -1)

		for zone, prob := range data.Probs {
			xStart := offsets[zone]

			// Color: Green if detected, Grey otherwise
			var rectCol color.RGBA
			thickness := 1

			if prob > 0.60 { // Visual threshold
				rectCol = color.RGBA{0, 255, 0, 0} // Bright green
				thickness = 1
			} else {
				rectCol = color.RGBA{100, 100, 100, 0} // Discreet grey
			}

			// Draw the zone frame
			rect := image.Rect(xStart, 20, xStart+CropSize, CamHeight-thickness)
			gocv.Rectangle(&img, rect, rectCol, thickness)

			// Small text with probability at the top of the zone
			label := fmt.Sprintf("%.0f%%", prob*100)
			gocv.PutText(&img, label, image.Pt(xStart+5, 15), gocv.FontHersheySimplex, 0.4, rectCol, 1)
		}

		// Draw Labels
		gocv.PutText(&img, label, image.Pt(5, 15), gocv.FontHersheySimplex, 0.4, color.RGBA{200, 200, 200, 0}, 1)

		fpsText := fmt.Sprintf("FPS:%.1f", data.FPS)
		gocv.PutText(&img, fpsText, image.Pt(CamWidth-74, 15), gocv.FontHersheySimplex, 0.4, color.RGBA{200, 200, 200, 0}, 1)

		// 3. Write Frame to Video File (only if recording)
		recordingMutex.RLock()
		if isRecording && writer != nil && writer.IsOpened() {
			writer.Write(img)
		}
		recordingMutex.RUnlock()

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

	// 1. Decode raw bytes to OpenCV Mat to allow manipulation
	img, err := gocv.IMDecode(data, gocv.IMReadColor)
	if err != nil {
		fmt.Printf("[Error] Failed to decode image for cropping: %v\n", err)
		w.Write([]byte("Error Decoding"))
		return
	}
	// Important: Free memory when function exits
	defer img.Close()

	// 2. Define the Center Region of Interest (ROI)
	// We use the constants defined at the top of your file
	// x: 48, y: 0, w: 224, h: 224
	rect := image.Rect(OffsetCenter, 0, OffsetCenter+CropSize, CamHeight)

	// Create a new Mat that points to that specific region
	cropped := img.Region(rect)
	defer cropped.Close()

	// 3. Generate filename
	files, _ := ioutil.ReadDir(DatasetDir)
	id := len(files)
	filename := fmt.Sprintf("%d.jpg", id)
	path := filepath.Join(DatasetDir, filename)

	// 4. Save the CROPPED image to disk
	if success := gocv.IMWrite(path, cropped); !success {
		w.Write([]byte("Error Saving"))
		return
	}

	msg := fmt.Sprintf("Saved Center Crop: %s", filename)
	fmt.Printf("[Photo] %s\n", msg)
	w.Write([]byte(msg))
}

func streamHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "multipart/x-mixed-replace; boundary=frame")
	for jpgData := range streamChannel {
		fmt.Fprintf(w, "--frame\r\nContent-Type: image/jpeg\r\n\r\n%s\r\n", jpgData)
	}
}

func toggleRecordingHandler(w http.ResponseWriter, r *http.Request) {
	recordingMutex.Lock()
	defer recordingMutex.Unlock()

	newState := !isRecording
	select {
	case recordingSignal <- newState:
	default:
	}

	var msg string
	if newState {
		msg = "Recording started"
		fmt.Println("[Record] Recording started via web interface")
	} else {
		msg = "Recording stopped"
		fmt.Println("[Record] Recording stopped via web interface")
	}
	w.Write([]byte(msg))
}

func recordingStatusHandler(w http.ResponseWriter, r *http.Request) {
	recordingMutex.RLock()
	status := isRecording
	recordingMutex.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]bool{"recording": status})
}
