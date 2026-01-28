package main

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"log"
	"net/http"
	"time"

	zmq "github.com/pebbe/zmq4"
	"gocv.io/x/gocv"
)

// --- CONFIG ---
const (
	JetsonIP  = "192.168.37.22"
	ZmqPort   = "5555"
	Threshold = 0.70
)

// Global channel to pass frames from ZMQ loop to HTTP handler
var frameStream = make(chan []byte, 1) // Buffer of 1 (Conflate behavior)

// Data structure matches the JSON sent by C++
type VisionData struct {
	ProbTarget float64 `json:"prob_target"`
	ImageB64   string  `json:"image_b64"`
	FPS        float64 `json:"jetson_fps"`
}

func main() {
	// Start ZMQ Receiver in a Goroutine
	go zmqReceiver()

	// Start HTTP Server
	http.HandleFunc("/", indexHandler)
	http.HandleFunc("/video_feed", streamHandler)

	fmt.Println("[System] Starting Go Web Server on :5000...")
	log.Fatal(http.ListenAndServe(":5000", nil))
}

func zmqReceiver() {
	// Setup ZMQ Subscriber
	zCtx, _ := zmq.NewContext()
	socket, _ := zCtx.NewSocket(zmq.SUB)
	defer socket.Close()

	address := fmt.Sprintf("tcp://%s:%s", JetsonIP, ZmqPort)
	fmt.Printf("[Connect] Connecting to Jetson at %s...\n", address)
	socket.Connect(address)
	socket.SetSubscribe("")
	// Conflate to keep only latest message
	socket.SetConflate(true)

	// Video Writer Setup
	timestamp := time.Now().Format("20060102_150405")
	filename := fmt.Sprintf("jetson_recording_%s.avi", timestamp)
	writer, err := gocv.VideoWriterFile(filename, "MJPG", 20, 224, 224, true)
	if err != nil {
		fmt.Printf("[Error] Could not open video writer: %v\n", err)
	} else {
		fmt.Printf("[Record] Saving to %s\n", filename)
		defer writer.Close()
	}

	for {
		msg, err := socket.Recv(0)
		if err != nil {
			continue
		}

		// Parse JSON
		var data VisionData
		json.Unmarshal([]byte(msg), &data)

		// Decode Image
		imgBytes, _ := base64.StdEncoding.DecodeString(data.ImageB64)
		img, err := gocv.IMDecode(imgBytes, gocv.IMReadColor)
		if err != nil {
			continue
		}

		// --- DRAW OVERLAY ---
		label := fmt.Sprintf("NOCIBLE:%0.0f%%", data.ProbTarget*100)
		col := color.RGBA{0, 0, 255, 0} // Red
		rectCol := color.RGBA{0, 0, 50, 0}

		if data.ProbTarget > Threshold {
			label = fmt.Sprintf("CIBLE:%0.0f%%", data.ProbTarget*100)
			col = color.RGBA{0, 255, 0, 0} // Green
			rectCol = color.RGBA{0, 50, 0, 0}
		}

		// Draw Top Bar
		gocv.Rectangle(&img, image.Rect(0, 0, 224, 20), rectCol, -1)

		// Draw Text
		gocv.PutText(&img, label, image.Pt(5, 15), gocv.FontHersheySimplex, 0.4, col, 1)
		fpsText := fmt.Sprintf("FPS:%.1f", data.FPS)
		// GoCV doesn't have GetTextSize easily accessible in all versions, hardcoding pos for 224px
		gocv.PutText(&img, fpsText, image.Pt(150, 15), gocv.FontHersheySimplex, 0.4, color.RGBA{200, 200, 200, 0}, 1)

		// Save to disk
		if writer.IsOpened() {
			writer.Write(img)
		}

		// Re-encode to JPG for Browser
		buf, _ := gocv.IMEncode(".jpg", img)

		// Push to channel (non-blocking drop if full)
		select {
		case frameStream <- buf.GetBytes():
		default:
		}

		img.Close()
	}
}

func indexHandler(w http.ResponseWriter, r *http.Request) {
	html := `<html><body style="background:#222;text-align:center;">
			 <h1 style="color:#eee;font-family:monospace;">JetsonPilot Go Viewer</h1>
			 <img src="/video_feed" style="border:2px solid #444;width:672px;image-rendering:pixelated;"/>
			 </body></html>`
	w.Write([]byte(html))
}

func streamHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "multipart/x-mixed-replace; boundary=frame")
	for jpgData := range frameStream {
		fmt.Fprintf(w, "--frame\r\nContent-Type: image/jpeg\r\n\r\n%s\r\n", jpgData)
	}
}
