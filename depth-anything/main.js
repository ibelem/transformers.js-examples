import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

import { pipeline, RawImage, env } from "@huggingface/transformers";

env.backends.onnx.logSeverityLevel = 0;

// Constants
const EXAMPLE_URL =
  "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/bread_small.png";
const DEFAULT_SCALE = 0.75;

// Reference the elements that we will need
const status = document.getElementById("status");
const fileUpload = document.getElementById("upload");
const imageContainer = document.getElementById("container");
const example = document.getElementById("example");

const urlParams = new URLSearchParams(window.location.search);
const provider = urlParams.get('provider');

let deviceType = 'webgpu';
if (provider) {
  deviceType = provider.toLowerCase();
}

document.querySelector('#log').innerHTML = deviceType + ' + fp16';

const depth_estimator = await pipeline(
  "depth-estimation",
  "webnn/depth-anything-v2-small-518",
  {
    device: deviceType,
    dtype: "fp16",
    session_options: {
      logSeverityLevel: 0
    }
  }
);
status.textContent = "Ready";

example.addEventListener("click", (e) => {
  e.preventDefault();
  predict(EXAMPLE_URL);
});

fileUpload.addEventListener("change", function (e) {
  const file = e.target.files[0];
  if (!file) {
    return;
  }

  const reader = new FileReader();

  // Set up a callback when the file is loaded
  reader.onload = (e2) => predict(e2.target.result);

  reader.readAsDataURL(file);
});

let onSliderChange;

// Predict depth map for the given image
async function predict(imageUrl) {
  imageContainer.innerHTML = "";
  const origImg = await RawImage.fromURL(imageUrl);
  console.log(`Image - Original: ${origImg.width} x ${origImg.height}`);
  
  // Set up scene with original dimensions
  const { canvas: displayCanvas, setDisplacementMap } = setupScene(
    imageUrl,
    origImg.width,
    origImg.height
  );
  imageContainer.append(displayCanvas);
  
  // Resize to 518x518 for depth_estimator
  const requiredSize = 518;
  const resizeCanvas = new OffscreenCanvas(requiredSize, requiredSize);
  const resizeCtx = resizeCanvas.getContext('2d');
  
  // Convert the RawImage to a format compatible with drawImage
  const imgBlob = await origImg.toBlob();
  const imgBlobUrl = URL.createObjectURL(imgBlob);
  const htmlImg = new Image();
  await new Promise(resolve => {
    htmlImg.onload = resolve;
    htmlImg.src = imgBlobUrl;
  });
  
  // Draw and resize to 518x518 square (depth_estimator requirement)
  resizeCtx.drawImage(htmlImg, 0, 0, requiredSize, requiredSize);
  
  // Convert back to a format usable by depth_estimator
  const processedBlob = await resizeCanvas.convertToBlob({ type: 'image/png' });
  const processedImg = await RawImage.fromBlob(processedBlob);
  
  // Clean up
  URL.revokeObjectURL(imgBlobUrl);
  console.log(`Image - Resized for depth estimation: ${processedImg.width} x ${processedImg.height}`);
  
  status.textContent = "Analysing...";
  const { depth } = await depth_estimator(processedImg);
  
  // Get depth map as canvas
  const depthCanvas = depth.toCanvas();
  console.log(`Depth map size: ${depthCanvas.width} x ${depthCanvas.height}`);
  
  // Restore depth map canvas to original aspect ratio
  const aspectRatio = origImg.width / origImg.height;
  const outputWidth = origImg.width;
  const outputHeight = origImg.height;
  
  const finalCanvas = document.createElement('canvas');
  finalCanvas.width = outputWidth;
  finalCanvas.height = outputHeight;
  const finalCtx = finalCanvas.getContext('2d');
  finalCtx.drawImage(depthCanvas, 0, 0, outputWidth, outputHeight);
  console.log(`Output Canvas - Restored to: ${finalCanvas.width} x ${finalCanvas.height}`);
  
  // Set the displacement map with corrected aspect ratio
  setDisplacementMap(finalCanvas);
  status.textContent = "";
  
  // Add slider control
  const depthSlider = document.createElement("input");
  depthSlider.type = "range";
  depthSlider.min = 0;
  depthSlider.max = 1;
  depthSlider.step = 0.01;
  depthSlider.addEventListener("input", (e) => {
    onSliderChange(parseFloat(e.target.value));
  });
  depthSlider.defaultValue = DEFAULT_SCALE;
  imageContainer.append(depthSlider);
}

function setupScene(url, w, h) {
  // Create new scene
  const canvas = document.createElement("canvas");
  const width = (canvas.width = imageContainer.offsetWidth);
  const height = (canvas.height = imageContainer.offsetHeight);

  const scene = new THREE.Scene();

  // Create camera and add it to the scene
  const camera = new THREE.PerspectiveCamera(30, width / height, 0.01, 10);
  camera.position.z = 2;
  scene.add(camera);

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(window.devicePixelRatio);

  // Add ambient light
  const light = new THREE.AmbientLight(0xffffff, 2);
  scene.add(light);

  // Load depth texture
  const image = new THREE.TextureLoader().load(url);
  image.colorSpace = THREE.SRGBColorSpace;
  const material = new THREE.MeshStandardMaterial({
    map: image,
    side: THREE.DoubleSide,
  });
  material.displacementScale = DEFAULT_SCALE;

  const setDisplacementMap = (canvas) => {
    material.displacementMap = new THREE.CanvasTexture(canvas);
    material.needsUpdate = true;
  };

  const setDisplacementScale = (scale) => {
    material.displacementScale = scale;
    material.needsUpdate = true;
  };
  onSliderChange = setDisplacementScale;

  // Create plane and rescale it so that max(w, h) = 1
  const [pw, ph] = w > h ? [1, h / w] : [w / h, 1];
  const geometry = new THREE.PlaneGeometry(pw, ph, w, h);
  const plane = new THREE.Mesh(geometry, material);
  scene.add(plane);

  // Add orbit controls
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  renderer.setAnimationLoop(() => {
    renderer.render(scene, camera);
    controls.update();
  });

  window.addEventListener(
    "resize",
    () => {
      const width = imageContainer.offsetWidth;
      const height = imageContainer.offsetHeight;

      camera.aspect = width / height;
      camera.updateProjectionMatrix();

      renderer.setSize(width, height);
    },
    false,
  );

  return {
    canvas: renderer.domElement,
    setDisplacementMap,
  };
}
