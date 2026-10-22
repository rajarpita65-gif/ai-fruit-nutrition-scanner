import { useCallback, useEffect, useRef, useState } from "react";
import "./App.css";
import { getScanHistory, saveScanResult } from "./firebase";

const API = "http://localhost:5000";
const HISTORY_CACHE_KEY = "nutrition_scan_history_cache";
const CAMERA_SAVE_COOLDOWN_MS = 15000;

const FOOD_EMOJI = {
  apple: "🍎",
  banana: "🍌",
  orange: "🍊",
  mango: "🥭",
  grape: "🍇",
  strawberry: "🍓",
  pineapple: "🍍",
  watermelon: "🍉",
  kiwi: "🥝",
  peach: "🍑",
  pear: "🍐",
  cherry: "🍒",
  blueberry: "🫐",
  lemon: "🍋",
  coconut: "🥥",
  tomato: "🍅",
  avocado: "🥑",
  corn: "🌽",
  carrot: "🥕",
  potato: "🥔",
  pepper: "🫑",
  plum: "🍑",
};

const getFoodEmoji = (name = "") => {
  const lowered = name.toLowerCase();
  return Object.entries(FOOD_EMOJI).find(([key]) => lowered.includes(key))?.[1] ?? "🥗";
};

const confidenceColor = (confidence) => {
  if (confidence >= 90) return "#00b894";
  if (confidence >= 70) return "#f6ad55";
  return "#fc8181";
};

const buildCameraErrorMessage = (error) => {
  if (error?.message === "UnsupportedBrowser") {
    return "This browser does not support camera access. Use a recent Chrome, Edge, or Firefox browser.";
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    return "This browser does not support camera access. Use a recent Chrome, Edge, or Firefox browser.";
  }

  if (!window.isSecureContext) {
    return "Camera access needs a secure page. Open the app on localhost or https, then try again.";
  }

  switch (error?.name) {
    case "NotAllowedError":
    case "PermissionDeniedError":
      return "Camera permission is blocked in this browser. Allow camera access for this site in the browser address bar or site settings, then try again.";
    case "NotFoundError":
    case "DevicesNotFoundError":
      return "No camera was found. Reconnect the USB camera and refresh the page.";
    case "NotReadableError":
    case "TrackStartError":
      return "The camera is already being used by another app. Close other camera apps and try again.";
    case "OverconstrainedError":
    case "ConstraintNotSatisfiedError":
      return "This camera does not support the selected settings. The app will retry with basic camera settings.";
    case "SecurityError":
      return "The browser blocked camera access for security reasons. Use localhost or https and allow camera access.";
    default:
      return `Camera access failed${error?.name ? ` (${error.name})` : ""}. Check browser camera permission and make sure no other app is using it.`;
  }
};

const getHistoryTimestampMs = (entry) => {
  if (!entry?.timestamp) return 0;
  if (typeof entry.timestamp?.toDate === "function") return entry.timestamp.toDate().getTime();
  if (typeof entry.timestamp === "number") return entry.timestamp;
  if (typeof entry.timestamp === "string") {
    const parsed = Date.parse(entry.timestamp);
    return Number.isFinite(parsed) ? parsed : 0;
  }
  if (typeof entry.timestamp?.seconds === "number") return entry.timestamp.seconds * 1000;
  if (typeof entry.timestamp?.timestampMs === "number") return entry.timestamp.timestampMs;
  return 0;
};

const dedupeHistoryEntries = (entries) => {
  const sorted = [...entries].sort((a, b) => getHistoryTimestampMs(b) - getHistoryTimestampMs(a));
  const latestByKey = new Map();
  const deduped = [];

  for (const entry of sorted) {
    const key = `${entry.food_name || ""}|${entry.weight || ""}`;
    const currentTime = getHistoryTimestampMs(entry);
    const previousTime = latestByKey.get(key);

    if (previousTime && previousTime - currentTime < CAMERA_SAVE_COOLDOWN_MS) {
      continue;
    }

    latestByKey.set(key, currentTime);
    deduped.push(entry);
  }

  return deduped;
};

const toCacheableHistory = (entries) => entries.map((entry) => ({
  ...entry,
  timestamp: getHistoryTimestampMs(entry),
}));

export default function App() {
  const loadCachedHistory = () => {
    try {
      const cached = localStorage.getItem(HISTORY_CACHE_KEY);
      if (!cached) return [];
      const parsed = JSON.parse(cached);
      return Array.isArray(parsed) ? dedupeHistoryEntries(parsed) : [];
    } catch {
      return [];
    }
  };

  const [activeTab, setActiveTab] = useState("upload");
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [weight, setWeight] = useState("");

  const [cameraActive, setCameraActive] = useState(false);
  const [videoDevices, setVideoDevices] = useState([]);
  const [deviceId, setDeviceId] = useState("");
  const [cameraStatus, setCameraStatus] = useState("idle");
  const [autoScanMessage, setAutoScanMessage] = useState("");

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [inputError, setInputError] = useState("");
  const [dragging, setDragging] = useState(false);
  const [backendStatus, setBackendStatus] = useState("checking");

  const [scanHistory, setScanHistory] = useState(loadCachedHistory);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [showHistory, setShowHistory] = useState(false);

  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const previewUrlRef = useRef("");
  const autoScanBusyRef = useRef(false);
  const lastSavedScanRef = useRef("");
  const lastSavedScanAtRef = useRef(0);

  const hasValidWeight = Number.isFinite(parseFloat(weight)) && parseFloat(weight) > 0;

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraActive(false);
    setCameraStatus("idle");
    setAutoScanMessage("");
  }, []);

  useEffect(() => {
    const checkBackend = async () => {
      try {
        const response = await fetch(`${API}/health`, {
          signal: AbortSignal.timeout(4000),
        });
        setBackendStatus(response.ok ? "online" : "offline");
      } catch {
        setBackendStatus("offline");
      }
    };

    void checkBackend();
    const intervalId = setInterval(checkBackend, 10000);
    return () => clearInterval(intervalId);
  }, []);

  useEffect(() => () => stopCamera(), [stopCamera]);

  useEffect(() => () => {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
  }, []);

  useEffect(() => {
    let cancelled = false;

    const refreshHistory = async () => {
      try {
        const history = dedupeHistoryEntries(await getScanHistory());
        if (cancelled) return;
        setScanHistory(history);
        try {
          localStorage.setItem(HISTORY_CACHE_KEY, JSON.stringify(toCacheableHistory(history)));
        } catch {
          // Ignore cache failures and keep using live data.
        }
      } catch {
        // Keep cached history if Firebase is temporarily slow.
      }
    };

    void refreshHistory();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!cameraActive || !streamRef.current || !videoRef.current) return;

    const video = videoRef.current;
    video.srcObject = streamRef.current;
    video.muted = true;

    const startPreview = async () => {
      try {
        await video.play();
        setCameraStatus("streaming");
      } catch {
        setCameraStatus("error");
        setInputError("Camera connected, but the browser could not start the live preview.");
      }
    };

    void startPreview();
  }, [cameraActive]);

  const applyFile = useCallback((file, { clearResult = true } = {}) => {
    if (!file || !file.type.startsWith("image/")) return;

    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    const nextUrl = URL.createObjectURL(file);
    previewUrlRef.current = nextUrl;

    setSelectedFile(file);
    setImagePreview(nextUrl);
    if (clearResult) setResult(null);
    setInputError("");
  }, []);

  const handleFileChange = (event) => applyFile(event.target.files[0]);
  const handleDragOver = (event) => {
    event.preventDefault();
    setDragging(true);
  };
  const handleDragLeave = () => setDragging(false);
  const handleDrop = (event) => {
    event.preventDefault();
    setDragging(false);
    applyFile(event.dataTransfer.files[0]);
  };

  const pickPreferredCamera = useCallback((cameras) => {
    if (!cameras.length) return "";
    const preferred = cameras.find((camera) => /usb|webcam|external|hd/i.test(camera.label || ""));
    return preferred?.deviceId || cameras[cameras.length - 1].deviceId;
  }, []);

  const buildVideoConstraints = useCallback((selectedDeviceId, { exact = true } = {}) => {
    if (selectedDeviceId) {
      return {
        deviceId: exact ? { exact: selectedDeviceId } : { ideal: selectedDeviceId },
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30, max: 30 },
      };
    }

    return {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      frameRate: { ideal: 30, max: 30 },
      facingMode: "environment",
    };
  }, []);

  const enumerateCameras = useCallback(async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const cameras = devices.filter((device) => device.kind === "videoinput");
      setVideoDevices(cameras);
      if (cameras.length) {
        setDeviceId((current) => current || pickPreferredCamera(cameras));
      }
    } catch {
      setInputError("Could not read the available cameras from your browser.");
    }
  }, [pickPreferredCamera]);

  useEffect(() => {
    if (!navigator.mediaDevices?.addEventListener) return undefined;

    const handleDeviceChange = () => {
      void enumerateCameras();
    };

    navigator.mediaDevices.addEventListener("devicechange", handleDeviceChange);
    return () => navigator.mediaDevices.removeEventListener("devicechange", handleDeviceChange);
  }, [enumerateCameras]);

  const startCamera = useCallback(async (nextDeviceId = deviceId) => {
    setInputError("");
    setCameraStatus("starting");

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("UnsupportedBrowser");
      }

      streamRef.current?.getTracks().forEach((track) => track.stop());

      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: buildVideoConstraints(nextDeviceId, { exact: true }),
        });
      } catch (primaryError) {
        if (primaryError?.name === "NotAllowedError" || primaryError?.name === "PermissionDeniedError") {
          throw primaryError;
        }

        stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: buildVideoConstraints(nextDeviceId, { exact: false }),
        });
      }

      streamRef.current = stream;
      setCameraActive(true);

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.muted = true;
        try {
          await videoRef.current.play();
        } catch {
          // The effect above retries after the camera view renders.
        }
      }

      const devices = await navigator.mediaDevices.enumerateDevices();
      const cameras = devices.filter((device) => device.kind === "videoinput");
      setVideoDevices(cameras);
      if (!nextDeviceId && cameras.length) {
        setDeviceId(pickPreferredCamera(cameras));
      }
      setAutoScanMessage("Live camera ready. Hold one fruit steady for automatic analysis.");
    } catch (error) {
      setCameraStatus("error");
      setInputError(buildCameraErrorMessage(error));
    }
  }, [buildVideoConstraints, deviceId, pickPreferredCamera]);

  const captureFrameFile = useCallback(async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2 || !video.videoWidth || !video.videoHeight) {
      return null;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);

    const blob = await new Promise((resolve) => {
      canvas.toBlob(resolve, "image/jpeg", 0.92);
    });

    if (!blob) return null;
    return new File([blob], `camera-frame-${Date.now()}.jpg`, { type: "image/jpeg" });
  }, []);

  const saveSuccessfulScan = useCallback((data, { auto = false } = {}) => {
    const saveKey = `${data.food_name}|${data.weight}|${Math.round(data.confidence)}`;
    const baseKey = `${data.food_name}|${data.weight}`;
    const now = Date.now();

    if (auto && lastSavedScanRef.current === baseKey && now - lastSavedScanAtRef.current < CAMERA_SAVE_COOLDOWN_MS) {
      return;
    }
    if (!auto && lastSavedScanRef.current === saveKey) return;

    lastSavedScanRef.current = auto ? baseKey : saveKey;
    lastSavedScanAtRef.current = now;

    const localEntry = {
      id: `local-${now}`,
      ...data,
      timestamp: now,
    };

    setScanHistory((current) => {
      const next = dedupeHistoryEntries([localEntry, ...current]).slice(0, 12);
      try {
        localStorage.setItem(HISTORY_CACHE_KEY, JSON.stringify(toCacheableHistory(next)));
      } catch {
        // Ignore cache failures and keep UI responsive.
      }
      return next;
    });

    void saveScanResult(data).catch((error) => {
      console.warn("Firebase save failed:", error);
    });
  }, []);

  const analyzeImage = useCallback(async (file, { auto = false } = {}) => {
    if (!file) {
      if (!auto) setInputError("Please upload an image or use the live camera.");
      return null;
    }

    const parsedWeight = parseFloat(weight);
    if (!Number.isFinite(parsedWeight) || parsedWeight <= 0) {
      if (auto) setAutoScanMessage("Enter the fruit weight to enable live camera analysis.");
      else setInputError("Enter a valid weight in grams, for example 150.");
      return null;
    }

    if (autoScanBusyRef.current) return null;
    autoScanBusyRef.current = true;

    if (!auto || !result) setLoading(true);
    setInputError("");
    if (!auto) setResult(null);
    applyFile(file, { clearResult: !auto });

    const formData = new FormData();
    formData.append("image", file);
    formData.append("weight", String(parsedWeight));
    formData.append("source", auto ? "camera" : "upload");

    const controller = new AbortController();
    const timeoutMs = auto ? 20000 : 60000;
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const response = await fetch(`${API}/predict`, {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });
      clearTimeout(timer);

      const data = await response.json().catch(() => ({
        status: "error",
        message: "Backend returned an unreadable response.",
      }));

      if (auto && data.status === "no_detection") {
        setAutoScanMessage(data.message || "No fruit detected yet. Hold one fruit clearly in front of the camera.");
        return data;
      }

      if (!response.ok) {
        if (!auto || !result) setResult(data);
        if (auto) setAutoScanMessage(data.message || "Waiting for a clearer fruit view from the live camera.");
        return data;
      }

      setResult(data);
      if (data.status === "success") {
        saveSuccessfulScan(data, { auto });
        if (auto) setAutoScanMessage(`Live analysis active: ${data.food_name} detected.`);
      } else if (auto) {
        setAutoScanMessage(data.message || "Live camera is on. Hold one fruit steady in the frame.");
      }

      return data;
    } catch (error) {
      clearTimeout(timer);
      const message = error.name === "AbortError"
        ? "Live analysis timed out. Keeping the camera on and trying again."
        : "Cannot connect to backend. Make sure the Flask server is running on port 5000.";
      if (!auto || !result) setResult({ status: "error", message });
      if (auto) setAutoScanMessage(message);
      return null;
    } finally {
      autoScanBusyRef.current = false;
      setLoading(false);
    }
  }, [applyFile, result, saveSuccessfulScan, weight]);

  useEffect(() => {
    if (activeTab !== "camera" || !cameraActive) return undefined;
    if (backendStatus !== "online" || !hasValidWeight) return undefined;

    const scanLiveFrame = async () => {
      const file = await captureFrameFile();
      if (!file) return;
      await analyzeImage(file, { auto: true });
    };

    void scanLiveFrame();
    const intervalId = setInterval(() => {
      void scanLiveFrame();
    }, 2500);

    return () => clearInterval(intervalId);
  }, [activeTab, analyzeImage, backendStatus, cameraActive, captureFrameFile, hasValidWeight]);

  const cameraStatusMessage = activeTab === "camera" && cameraActive
    ? backendStatus !== "online"
      ? "Backend offline. Start Flask to enable live camera analysis."
      : !hasValidWeight
      ? "Enter the fruit weight to start live automatic analysis."
      : autoScanMessage || "Live automatic analysis is on. Hold one fruit steady in front of the camera."
    : autoScanMessage;

  const handleAnalyze = async (event) => {
    event.preventDefault();
    await analyzeImage(selectedFile, { auto: false });
  };

  const handleLoadHistory = async () => {
    if (showHistory) {
      setShowHistory(false);
      return;
    }

    setShowHistory(true);
    setHistoryLoading(true);
    try {
      const history = dedupeHistoryEntries(await getScanHistory());
      setScanHistory(history);
      try {
        localStorage.setItem(HISTORY_CACHE_KEY, JSON.stringify(toCacheableHistory(history)));
      } catch {
        // Ignore cache failures and keep using network data.
      }
    } catch {
      console.error("Failed to load history");
    } finally {
      setHistoryLoading(false);
    }
  };

  const resetAll = () => {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = "";
    }

    setSelectedFile(null);
    setImagePreview(null);
    setWeight("");
    setResult(null);
    setInputError("");
    setAutoScanMessage("");
    lastSavedScanRef.current = "";
    lastSavedScanAtRef.current = 0;
    stopCamera();
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const fmtTimestamp = (timestamp) => {
    try {
      const date = timestamp?.toDate ? timestamp.toDate() : new Date(timestamp.seconds * 1000);
      return date.toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch {
      return "";
    }
  };

  return (
    <div className="ns-app">
      <header className="ns-header">
        <div className="ns-header__inner">
          <div className="ns-header__brand">
            <span className="ns-header__logo">🥦</span>
            <div>
              <h1 className="ns-header__title">AI Edge Nutrition Scanner</h1>
              <p className="ns-header__sub">Upload a fruit image or use your USB camera for live nutrition analysis.</p>
            </div>
          </div>
          <span className="ns-badge">YOLO World + OpenCLIP · Firebase</span>
        </div>
      </header>

      {backendStatus === "offline" && (
        <div className="ns-backend-warn">
          Flask server is offline — start it with <code>python app.py</code> in the backend folder, then this banner will disappear.
        </div>
      )}
      {backendStatus === "checking" && (
        <div className="ns-backend-check">Connecting to backend...</div>
      )}

      <main className="ns-grid">
        <section className="ns-panel">
          <div className="ns-tabs">
            <button
              className={`ns-tab ${activeTab === "upload" ? "ns-tab--on" : ""}`}
              onClick={() => {
                setActiveTab("upload");
                stopCamera();
              }}
            >
              📁 Upload Image
            </button>
            <button
              className={`ns-tab ${activeTab === "camera" ? "ns-tab--on" : ""}`}
              onClick={() => {
                setActiveTab("camera");
                void enumerateCameras();
              }}
            >
              📷 Use Camera
            </button>
          </div>

          {activeTab === "upload" && (
            <div
              className={`ns-dropzone ${dragging ? "ns-dropzone--drag" : ""} ${selectedFile ? "ns-dropzone--filled" : ""}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              {imagePreview ? (
                <img src={imagePreview} alt="Preview" className="ns-preview" />
              ) : (
                <div className="ns-dropzone__ph">
                  <span className="ns-dropzone__icon">📸</span>
                  <p>Drag and drop your fruit image</p>
                  <span className="ns-dropzone__sub">or click to browse</span>
                  <span className="ns-chip">JPG · PNG · WEBP</span>
                </div>
              )}
              <input ref={fileInputRef} type="file" accept="image/*" onChange={handleFileChange} hidden />
            </div>
          )}

          {activeTab === "camera" && (
            <div className="ns-camera">
              {videoDevices.length > 0 && (
                <div className="ns-device-row">
                  <label htmlFor="cam-sel">Camera:</label>
                  <select
                    id="cam-sel"
                    value={deviceId}
                    onChange={(event) => {
                      const nextId = event.target.value;
                      setDeviceId(nextId);
                      if (cameraActive) void startCamera(nextId);
                    }}
                  >
                    {videoDevices.map((device, index) => (
                      <option key={device.deviceId} value={device.deviceId}>
                        {device.label || `Camera ${index + 1}`}
                      </option>
                    ))}
                  </select>
                </div>
              )}

              {cameraActive ? (
                <>
                  <div className="ns-camera__live">
                    <video ref={videoRef} autoPlay playsInline muted className="ns-camera__video" />
                    <span className="ns-camera__live-badge">LIVE</span>
                  </div>
                  <div className="ns-camera__btns">
                    <button className="ns-btn ns-btn--primary" onClick={() => void startCamera(deviceId)}>
                      ↻ Restart Camera
                    </button>
                    <button className="ns-btn ns-btn--ghost" onClick={stopCamera}>
                      Stop
                    </button>
                  </div>
                  <div className="ns-camera__status">
                    <strong>Camera status:</strong> {cameraStatus === "streaming" ? "Streaming" : "Starting..."}
                    <br />
                    {cameraStatusMessage || "Live camera is ready."}
                  </div>
                </>
              ) : (
                <div className="ns-camera__ph">
                  <span className="ns-camera__ph-icon">🎥</span>
                  <p>Start your external USB camera to analyze fruit automatically.</p>
                  <button className="ns-btn ns-btn--primary" style={{ marginTop: "1rem" }} onClick={() => void startCamera()}>
                    Start Camera
                  </button>
                </div>
              )}
              <canvas ref={canvasRef} hidden />
            </div>
          )}

          <div className="ns-field">
            <label htmlFor="ns-weight"><span>⚖️</span> Food Weight (grams)</label>
            <input
              id="ns-weight"
              type="number"
              value={weight}
              onChange={(event) => setWeight(event.target.value)}
              placeholder="e.g. 150"
              min="1"
              max="5000"
            />
          </div>

          {inputError && <div className="ns-alert ns-alert--err">⚠️ {inputError}</div>}

          <div className="ns-actions">
            <button
              className="ns-btn ns-btn--analyze"
              onClick={handleAnalyze}
              disabled={loading || backendStatus !== "online" || (activeTab === "camera" && cameraActive)}
            >
              {loading ? (
                <>
                  <span className="ns-spinner" /> Analyzing...
                </>
              ) : backendStatus === "offline" ? (
                "Backend Offline"
              ) : backendStatus === "checking" ? (
                "Connecting..."
              ) : activeTab === "camera" && cameraActive ? (
                "Live Auto Analysis Running"
              ) : (
                "Analyze Nutrition"
              )}
            </button>
            {(selectedFile || result || cameraActive) && (
              <button className="ns-btn ns-btn--ghost" onClick={resetAll}>
                Reset
              </button>
            )}
          </div>

          <div className="ns-tips">
            <p className="ns-tips__title">💡 Tips for best results</p>
            <ul>
              <li>Use a well-lit, clear image of a single fruit or food</li>
              <li>Ensure the fruit fills most of the frame</li>
              <li>Keep the fruit steady for 2-3 seconds in camera mode</li>
            </ul>
          </div>
        </section>

        <section className="ns-panel ns-results">
          {loading && (
            <div className="ns-card ns-card--loading">
              <div className="ns-pulse" />
              <span className="ns-loading-icon">🔬</span>
              <h3>Analyzing food...</h3>
              <p>AI model is processing your image</p>
              <p className="ns-loading-hint">First request may take 10-30 s while the detection models warm up.</p>
            </div>
          )}

          {!loading && !result && (
            <div className="ns-card ns-card--empty">
              <span className="ns-empty-icon">🥗</span>
              <h3>Ready to Scan</h3>
              <p>Upload a fruit image or start the camera to get detailed nutrition info.</p>
              <ul className="ns-features">
                <li>✓ AI fruit detection with YOLO and OpenCLIP</li>
                <li>✓ Live USB camera auto-analysis</li>
                <li>✓ Calories, protein, carbs, and fat</li>
                <li>✓ Scan history saved to Firebase</li>
              </ul>
            </div>
          )}

          {!loading && result?.status === "low_confidence" && (
            <div className="ns-card ns-card--warn">
              <span className="ns-warn-icon">⚠️</span>
              <h3>Low Confidence Detection</h3>
              <p>{result.message}</p>
              <div className="ns-detected">
                Detected: <strong>{result.food_name}</strong>
                <span className="ns-chip-conf" style={{ background: confidenceColor(result.confidence) }}>
                  {result.confidence}%
                </span>
              </div>
            </div>
          )}

          {!loading && result?.status === "not_found" && (
            <div className="ns-card ns-card--warn">
              <span className="ns-warn-icon">🔍</span>
              <h3>Nutrition Data Not Found</h3>
              <p>{result.message}</p>
            </div>
          )}

          {!loading && result?.status === "error" && (
            <div className="ns-card ns-card--err">
              <span className="ns-warn-icon">❌</span>
              <h3>Error</h3>
              <p>{result.message}</p>
            </div>
          )}

          {!loading && result?.status === "success" && (
            <>
              <div className="ns-card ns-food-header">
                {result.demo_mode && (
                  <div className="ns-demo-notice">
                    🧪 Fallback recognition — {result.message || "This result used a fallback before nutrition calculation."}
                  </div>
                )}
                <div className="ns-food-left">
                  <span className="ns-food-emoji">{getFoodEmoji(result.food_name)}</span>
                  <div>
                    <h2 className="ns-food-name">{result.food_name}</h2>
                    <div className="ns-food-meta">
                      <span className="ns-food-sub">{result.weight}g analyzed</span>
                      {result.recognition_model && <span className="ns-food-sub">{result.recognition_model}</span>}
                      {result.calorie_class && (
                        <span className={`ns-cal-badge ns-cal-badge--${result.calorie_class}`}>
                          {result.calorie_class.toUpperCase()} CALORIE
                        </span>
                      )}
                    </div>
                  </div>
                </div>
                {imagePreview && <img src={imagePreview} alt={result.food_name} className="ns-result-thumb" />}
                <div className="ns-conf">
                  <div className="ns-conf__row">
                    <span>AI Confidence</span>
                    <strong style={{ color: confidenceColor(result.confidence) }}>{result.confidence}%</strong>
                  </div>
                  <div className="ns-conf__track">
                    <div
                      className="ns-conf__fill"
                      style={{
                        width: `${result.confidence}%`,
                        background: confidenceColor(result.confidence),
                      }}
                    />
                  </div>
                </div>
              </div>

              <div className="ns-nutr-grid">
                <div className="ns-nutr ns-nutr--cal">
                  <span className="ns-nutr__icon">🔥</span>
                  <div className="ns-nutr__val">{result.nutrition.calories}</div>
                  <div className="ns-nutr__unit">kcal</div>
                  <div className="ns-nutr__lbl">Calories</div>
                </div>
                <div className="ns-nutr ns-nutr--pro">
                  <span className="ns-nutr__icon">💪</span>
                  <div className="ns-nutr__val">{result.nutrition.protein}</div>
                  <div className="ns-nutr__unit">g</div>
                  <div className="ns-nutr__lbl">Protein</div>
                </div>
                <div className="ns-nutr ns-nutr--carb">
                  <span className="ns-nutr__icon">🌾</span>
                  <div className="ns-nutr__val">{result.nutrition.carbohydrates}</div>
                  <div className="ns-nutr__unit">g</div>
                  <div className="ns-nutr__lbl">Carbohydrates</div>
                </div>
                <div className="ns-nutr ns-nutr--fat">
                  <span className="ns-nutr__icon">🥑</span>
                  <div className="ns-nutr__val">{result.nutrition.fat}</div>
                  <div className="ns-nutr__unit">g</div>
                  <div className="ns-nutr__lbl">Fat</div>
                </div>
              </div>

              {(result.nutrition.fiber > 0 || result.nutrition.sugar > 0) && (
                <div className="ns-card ns-extra">
                  <h4>More Nutrients</h4>
                  <div className="ns-extra__row">
                    {result.nutrition.fiber > 0 && (
                      <div className="ns-extra__item">
                        <span>🌿</span>
                        <strong>{result.nutrition.fiber}g</strong>
                        <span>Fiber</span>
                      </div>
                    )}
                    {result.nutrition.sugar > 0 && (
                      <div className="ns-extra__item">
                        <span>🍬</span>
                        <strong>{result.nutrition.sugar}g</strong>
                        <span>Sugar</span>
                      </div>
                    )}
                  </div>
                  <p className="ns-per100">
                    Per 100g — Cal: {result.nutrition_per_100g?.calories} kcal · Protein: {result.nutrition_per_100g?.protein}g ·
                    Carbs: {result.nutrition_per_100g?.carbohydrates}g · Fat: {result.nutrition_per_100g?.fat}g
                  </p>
                </div>
              )}
            </>
          )}
        </section>
      </main>

      <section className="ns-history">
        <div className="ns-history__hdr">
          <h2>📋 Scan History</h2>
          <button className="ns-btn ns-btn--outline" onClick={handleLoadHistory} disabled={historyLoading}>
            {historyLoading ? "Loading..." : showHistory ? "Hide History" : "Show History"}
          </button>
        </div>

        {showHistory && (
          <div className="ns-history__grid">
            {scanHistory.length === 0 ? (
              <p className="ns-no-hist">No scans yet. Start analyzing fruit.</p>
            ) : (
              scanHistory.map((scan) => (
                <div key={scan.id} className="ns-hcard">
                  <div className="ns-hcard__emoji">{getFoodEmoji(scan.food_name)}</div>
                  <div className="ns-hcard__name">{scan.food_name}</div>
                  <div className="ns-hcard__row">
                    <span>{scan.weight}g</span>
                    <span>{scan.nutrition?.calories} kcal</span>
                  </div>
                  <div className="ns-hcard__conf" style={{ color: confidenceColor(scan.confidence) }}>
                    {scan.confidence}% confident
                  </div>
                  <div className="ns-hcard__time">{fmtTimestamp(scan.timestamp)}</div>
                </div>
              ))
            )}
          </div>
        )}
      </section>

      <footer className="ns-footer">
        AI Edge Nutrition Scanner · YOLO World · OpenCLIP · Firebase
      </footer>
    </div>
  );
}
