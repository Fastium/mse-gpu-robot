package main

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"time"

	zmq "github.com/pebbe/zmq4"
)

// --- CONFIGURATION ---
const (
	JetsonIP     = "192.168.37.22" // <--- CHECK IP
	ZmqPort      = "5555"
	DatasetDir   = "dataset_capture"

	// Set to true for automatic capture every X seconds
	// Set to false to capture only when pressing ENTER (Recommended)
	AutoMode     = false
	AutoInterval = 5 * time.Second
)

// Data structure matches the JSON sent by Jetson
type VisionData struct {
	ImageB64 string `json:"image_b64"`
}

func main() {
	// 1. Create Output Directory
	if _, err := os.Stat(DatasetDir); os.IsNotExist(err) {
		fmt.Printf("[Init] Creating directory: %s\n", DatasetDir)
		os.MkdirAll(DatasetDir, 0755)
	}

	// 2. Setup ZMQ Subscriber
	zCtx, _ := zmq.NewContext()
	socket, _ := zCtx.NewSocket(zmq.SUB)
	defer socket.Close()

	addr := fmt.Sprintf("tcp://%s:%s", JetsonIP, ZmqPort)
	fmt.Printf("[Connect] Connecting to Jetson at %s...\n", addr)
	socket.Connect(addr)
	socket.SetSubscribe("")
	// We use Conflate to always get the freshest frame available
	socket.SetConflate(true)

	// 3. Prepare Logic
	counter := countExistingFiles(DatasetDir)
	fmt.Printf("[Init] Resuming counter at: %d.jpg\n", counter)

	// Channel to signal a save request
	saveTrigger := make(chan bool)

	// --- INPUT LOOP (Goroutine) ---
	go func() {
		if AutoMode {
			fmt.Printf("[Mode] AUTO: Capturing every %v\n", AutoInterval)
			for {
				time.Sleep(AutoInterval)
				saveTrigger <- true
			}
		} else {
			fmt.Println("[Mode] MANUAL: Press [ENTER] to save a frame.")
			fmt.Println("       (Press Ctrl+C to stop)")
			var input string
			for {
				fmt.Scanln(&input) // Wait for Enter key
				saveTrigger <- true
			}
		}
	}()

	// --- MAIN RECEIVE LOOP ---
	for {
		// Non-blocking check for save trigger
		shouldSave := false
		select {
		case <-saveTrigger:
			shouldSave = true
		default:
			// No trigger, continue receiving
		}

		// Receive ZMQ message
		msg, err := socket.Recv(0)
		if err != nil {
			continue
		}

		// Only parse JSON if we are about to save (Optimization)
		if shouldSave {
			var data VisionData
			if err := json.Unmarshal([]byte(msg), &data); err != nil {
				log.Printf("[Error] Bad JSON: %v\n", err)
				continue
			}

			saveImage(data.ImageB64, counter)
			counter++

			if !AutoMode {
				fmt.Print("Ready > ") // Prompt for next input
			}
		}
	}
}

func saveImage(b64String string, id int) {
	// Decode Base64 to raw bytes
	data, err := base64.StdEncoding.DecodeString(b64String)
	if err != nil {
		log.Printf("[Error] Base64 decode failed: %v\n", err)
		return
	}

	filename := fmt.Sprintf("%d.jpg", id)
	path := filepath.Join(DatasetDir, filename)

	// Write raw bytes to disk (It is already a JPEG, no need to re-encode)
	err = ioutil.WriteFile(path, data, 0644)
	if err != nil {
		log.Printf("[Error] Write failed: %v\n", err)
		return
	}

	fmt.Printf("[Saved] %s (%d KB)\n", path, len(data)/1024)
}

// Helper to find the next available filename index
func countExistingFiles(dir string) int {
	files, _ := ioutil.ReadDir(dir)
	// Simplified logic: just count files to restart somewhat correctly.
	return len(files)
}
