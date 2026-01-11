// ================= CONFIG =================
const HISTORY_LENGTH = 16;
const STATIC_CONF_TH = 0.6;
const DYNAMIC_CONF_TH = 0.6;

// ================= STATE =================
let pointHistory = [];
let staticHistory = [];
let dynamicHistory = [];

let staticModel = null;
let dynamicModel = null;
let KEYPOINT_LABELS = [];
let POINT_HISTORY_LABELS = [];

// ================= LOADERS =================
async function loadJSON(path) {
  const res = await fetch(path);
  return res.json();
}

async function loadModels() {
  staticModel = await tf.loadLayersModel("models/keypoint/model.json");
  dynamicModel = await tf.loadLayersModel("models/point_history/model.json");
}

// ================= PREPROCESS =================
function preprocessLandmarks(landmarks) {
  const baseX = landmarks[0].x;
  const baseY = landmarks[0].y;

  let processed = landmarks.map(lm => [
    lm.x - baseX,
    lm.y - baseY
  ]).flat();

  const maxVal = Math.max(...processed.map(v => Math.abs(v)));
  return processed.map(v => v / maxVal);
}

function preprocessPointHistory(history) {
  const baseX = history[0][0];
  const baseY = history[0][1];

  return history.map(p => [
    p[0] - baseX,
    p[1] - baseY
  ]).flat();
}

// ================= UTILS =================
function majorityVote(arr) {
  if (arr.length === 0) return null;
  const freq = {};
  arr.forEach(v => freq[v] = (freq[v] || 0) + 1);
  return Object.entries(freq).sort((a,b)=>b[1]-a[1])[0][0];
}

// ================= MEDIAPIPE =================
const hands = new Hands({
  locateFile: file =>
    `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
});

hands.setOptions({
  maxNumHands: 1,
  minDetectionConfidence: 0.7,
  minTrackingConfidence: 0.5
});

hands.onResults(async (results) => {
  if (!results.multiHandLandmarks) return;

  const landmarks = results.multiHandLandmarks[0];

  // ---------- STATIC ----------
  const staticInput = preprocessLandmarks(landmarks);
  const staticTensor = tf.tensor([staticInput]);
  const staticPred = staticModel.predict(staticTensor);
  const staticData = await staticPred.data();
  staticTensor.dispose();

  const staticId = staticData.indexOf(Math.max(...staticData));
  if (staticData[staticId] > STATIC_CONF_TH) {
    staticHistory.push(staticId);
  }

  const staticResult = majorityVote(staticHistory);
  const staticLabel = staticResult !== null ? KEYPOINT_LABELS[staticResult] : "-";

  // ---------- DYNAMIC ----------
  const tip = landmarks[8];
  pointHistory.push([tip.x, tip.y]);

  if (pointHistory.length === HISTORY_LENGTH) {
    const dynInput = preprocessPointHistory(pointHistory);
    const dynTensor = tf.tensor([dynInput]);
    const dynPred = dynamicModel.predict(dynTensor);
    const dynData = await dynPred.data();
    dynTensor.dispose();

    const dynId = dynData.indexOf(Math.max(...dynData));
    if (dynData[dynId] > DYNAMIC_CONF_TH) {
      dynamicHistory.push(dynId);
    }
  }

  document.getElementById("output").innerText =
    `STATIC: ${staticLabel}`;
});

// ================= CAMERA =================
function startCamera() {
  const video = document.getElementById("video");

  const camera = new Camera(video, {
    onFrame: async () => {
      await hands.send({ image: video });
    },
    width: 640,
    height: 480
  });

  camera.start();
}

// ================= INIT =================
(async () => {
  KEYPOINT_LABELS = await loadJSON("labels/keypoint_labels.json");
  POINT_HISTORY_LABELS = await loadJSON("labels/point_history_labels.json");

  await loadModels();
  startCamera();
})();