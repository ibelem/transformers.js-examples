import { AutoModel, AutoProcessor, RawImage, env } from "@huggingface/transformers";

// Reduce log noise from ONNX Runtime
env.backends.onnx.logSeverityLevel = 0;

// DOM elements
const videoElement = document.getElementById("video");
const canvasElement = document.getElementById("canvas");
const overlayElement = document.getElementById("overlay");
const statusElement = document.getElementById("status");
const fpsElement = document.getElementById("fps");
const confidenceSlider = document.getElementById("confidence");
const confidenceValue = document.getElementById("confidence-value");

// Configuration
const MODEL_ID = "webnn/yolov8m";
let confidenceThreshold = 0.25;
let isProcessing = false;
let lastFrameTime = 0;

// Initialize sliders
confidenceSlider.value = confidenceThreshold;
confidenceValue.textContent = confidenceThreshold;
confidenceSlider.addEventListener("input", () => {
  confidenceThreshold = parseFloat(confidenceSlider.value);
  confidenceValue.textContent = confidenceThreshold.toFixed(2);
});

// Colors for bounding boxes
const COLORS = [
  "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231", 
  "#48F90A", "#92CC17", "#3DDB86", "#1A9334", "#00D4BB", 
  "#2C99A8", "#00C2FF", "#344593", "#6473FF", "#0018EC", 
  "#8438FF", "#520085", "#CB38FF", "#FF95C8", "#FF37C7"
];

async function initializeModel() {
  statusElement.textContent = "Loading YOLOv8m model...";
  
  try {
    // Detect best available backend
    const urlParams = new URLSearchParams(window.location.search);
    const provider = urlParams.get('provider') || 'webgpu';
    
    console.log(`Using ${provider} backend with fp16 precision`);
    
    // Load model and processor
    const model = await AutoModel.from_pretrained(MODEL_ID, {
      device: provider.toLowerCase(),
      dtype: "fp16",
    });
    
    const processor = await AutoProcessor.from_pretrained(MODEL_ID);
    
    // Configure processor to match model's expected input size (640x640)
    processor.feature_extractor.size = { width: 640, height: 640 };
    
    // Log the class names from the model config
    console.log("Model config:", model.config);
    console.log("Class labels:", model.config.id2label);
    
    statusElement.textContent = "Model loaded! Starting camera...";
    return { model, processor };
  } catch (error) {
    statusElement.textContent = `Error loading model: ${error.message}`;
    console.error("Model initialization error:", error);
    throw error;
  }
}

function setupCamera() {
  return new Promise((resolve, reject) => {
    navigator.mediaDevices.getUserMedia({ 
      video: { 
        facingMode: "environment",
        width: { ideal: 1280 },
        height: { ideal: 720 }
      } 
    })
    .then(stream => {
      videoElement.srcObject = stream;
      
      videoElement.onloadedmetadata = () => {
        // Set canvas size to match video
        canvasElement.width = videoElement.videoWidth;
        canvasElement.height = videoElement.videoHeight;
        overlayElement.style.width = `${canvasElement.width}px`;
        overlayElement.style.height = `${canvasElement.height}px`;
        
        // Start video playback
        videoElement.play();
        resolve();
      };
    })
    .catch(error => {
      statusElement.textContent = `Camera error: ${error.message}`;
      reject(error);
    });
  });
}

function processDetections(outputs, imageWidth, imageHeight, classLabels) {
  // Clear previous detections
  overlayElement.innerHTML = "";
  
  // Process YOLOv8 outputs (shape: [1, 84, 8400])
  // For each of the 8400 predictions, we have 84 values:
  // - First 4 are bounding box coordinates (x, y, width, height)
  // - Remaining 80 are class confidences for COCO dataset
  
  const predictions = outputs.tolist()[0];  // Get the first batch
  const numClasses = predictions.length - 4;  // Subtract 4 for bbox coordinates
  const numPredictions = predictions[0].length;  // Number of predictions (8400)
  
  let detections = [];
  
  // Process each prediction
  for (let i = 0; i < numPredictions; i++) {
    // Get bbox coordinates (center_x, center_y, width, height)
    const x = predictions[0][i];
    const y = predictions[1][i];
    const w = predictions[2][i];
    const h = predictions[3][i];
    
    // Find the class with highest confidence
    let maxScore = 0;
    let maxClassIndex = -1;
    
    for (let c = 0; c < numClasses; c++) {
      const score = predictions[c + 4][i];
      if (score > maxScore) {
        maxScore = score;
        maxClassIndex = c;
      }
    }
    
    // Skip if below threshold
    if (maxScore < confidenceThreshold) continue;
    
    // Convert from center coordinates to corner coordinates
    const xmin = (x - w/2) / 640 * imageWidth;
    const ymin = (y - h/2) / 640 * imageHeight;
    const width = w / 640 * imageWidth;
    const height = h / 640 * imageHeight;
    
    detections.push({
      bbox: [xmin, ymin, width, height],
      score: maxScore,
      class: maxClassIndex
    });
  }
  
  // Draw detections
  detections.forEach(detection => {
    const [x, y, width, height] = detection.bbox;
    const className = classLabels[detection.class];
    const color = COLORS[detection.class % COLORS.length];
    const score = detection.score;
    
    // Create box element
    const boxElement = document.createElement("div");
    boxElement.className = "detection-box";
    boxElement.style.left = `${x}px`;
    boxElement.style.top = `${y}px`;
    boxElement.style.width = `${width}px`;
    boxElement.style.height = `${height}px`;
    boxElement.style.borderColor = color;
    
    // Create label element
    const labelElement = document.createElement("div");
    labelElement.className = "detection-label";
    labelElement.style.backgroundColor = color;
    labelElement.textContent = `${className} ${(score * 100).toFixed(1)}%`;
    
    boxElement.appendChild(labelElement);
    overlayElement.appendChild(boxElement);
  });
  
  return detections.length;
}

async function detectFrame(model, processor, ctx) {
  if (isProcessing) return;
  isProcessing = true;
  
  try {
    const startTime = performance.now();
    
    // Capture current frame from video
    ctx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
    const imageData = ctx.getImageData(0, 0, canvasElement.width, canvasElement.height);
    
    // Convert to RawImage format for transformers.js
    const image = new RawImage(imageData.data, canvasElement.width, canvasElement.height, 4);
    
    // Process image and run model
    const inputs = await processor(image);
    const { outputs } = await model(inputs);
    
    // Extract class labels from model config
    const classLabels = {};
    for (const [id, label] of Object.entries(model.config.id2label)) {
      classLabels[id] = label;
    }
    
    // Process and display detections
    const detectionCount = processDetections(outputs, canvasElement.width, canvasElement.height, classLabels);
    
    // Calculate FPS
    const endTime = performance.now();
    const frameTime = endTime - startTime;
    const fps = 1000 / (endTime - lastFrameTime);
    lastFrameTime = endTime;
    
    // Update status
    statusElement.textContent = `Detected: ${detectionCount} objects`;
    fpsElement.textContent = `FPS: ${fps.toFixed(1)} | Processing: ${frameTime.toFixed(0)}ms`;
  } catch (error) {
    console.error("Detection error:", error);
    statusElement.textContent = `Error: ${error.message}`;
  } finally {
    isProcessing = false;
  }
}

async function startDetection() {
  try {
    // Initialize model and camera
    const { model, processor } = await initializeModel();
    await setupCamera();
    
    const ctx = canvasElement.getContext("2d");
    
    // Main detection loop
    function detectionLoop() {
      detectFrame(model, processor, ctx).finally(() => {
        requestAnimationFrame(detectionLoop);
      });
    }
    
    // Start detection loop
    detectionLoop();
    
  } catch (error) {
    console.error("Application error:", error);
    statusElement.textContent = `Failed to start: ${error.message}`;
  }
}

// Start the application when the page is loaded
window.onload = startDetection;