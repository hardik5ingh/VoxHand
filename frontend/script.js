// const API_BASE = "http://127.0.0.1:5000"; 
const API_BASE = "https://voxhand-backend.onrender.com";
let GOOGLE_CLIENT_ID = "";

// Generate or retrieve unique device session ID so different browsers do not share sentences
let sessionId = sessionStorage.getItem("voxhand_session_id");
if (!sessionId) {
  sessionId = "sess_" + Math.random().toString(36).substring(2, 9) + "_" + Date.now();
  sessionStorage.setItem("voxhand_session_id", sessionId);
}

const video = document.getElementById("webcam");
const canvas = document.getElementById("hiddenCanvas");
const ctx = canvas.getContext("2d");
const placeholder = document.getElementById("cameraPlaceholder");
const toggleCamBtn = document.getElementById("toggleCamBtn");
const camStatusIndicator = document.getElementById("camStatusIndicator");
const cameraSelect = document.getElementById("cameraSelect");

let mediaStream = null;
let isCameraRunning = false;
let loopTimeout = null;
let lastSentence = "";
let lastWord = "";
let currentSuggestionsCache = "";
let selectedDeviceId = "";
let toastTimer = null;
let googleBtnRendered = false;

// ==========================================
// PAUSABLE GUEST TRIAL TIMER (2 MINUTES)
// ==========================================
const TRIAL_DURATION_MS = 2 * 60 * 1000;
let trialTimer = null;
let trialStartTime = 0;
let remainingTrialMs = TRIAL_DURATION_MS;
let isTimerPaused = false;

function hasUsedTrial() {
  return localStorage.getItem("voxhand_guest_trial_used") === "true";
}

function markTrialUsed() {
  localStorage.setItem("voxhand_guest_trial_used", "true");
}

function startTrialCountdown() {
  if (trialTimer) clearTimeout(trialTimer);
  if (hasUsedTrial() || isUserLoggedIn()) return;

  trialStartTime = Date.now();
  isTimerPaused = false;

  trialTimer = setTimeout(() => {
    if (!isUserLoggedIn() && isCameraRunning) {
      markTrialUsed();
      stopCamera();
      openTrialModal();
    }
  }, remainingTrialMs);
}

function pauseTrialCountdown() {
  if (!trialTimer || isTimerPaused || isUserLoggedIn() || !isCameraRunning) return;

  clearTimeout(trialTimer);
  trialTimer = null;

  const elapsed = Date.now() - trialStartTime;
  remainingTrialMs = Math.max(0, remainingTrialMs - elapsed);
  isTimerPaused = true;
}

function resumeTrialCountdown() {
  if (!isTimerPaused || isUserLoggedIn() || !isCameraRunning) return;

  if (remainingTrialMs <= 0) {
    markTrialUsed();
    stopCamera();
    openTrialModal();
    return;
  }

  startTrialCountdown();
}

function resetTrialState() {
  if (trialTimer) clearTimeout(trialTimer);
  trialTimer = null;
  remainingTrialMs = TRIAL_DURATION_MS;
  isTimerPaused = false;
}

// ==========================================
// CLIENT-SIDE AUDIO FEEDBACK (WEB AUDIO API)
// ==========================================
let audioCtx = null;

function initAudio() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (audioCtx.state === 'suspended') {
    audioCtx.resume();
  }
}

function playTone(frequency, durationMs) {
  initAudio();
  try {
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();

    osc.type = 'sine';
    osc.frequency.setValueAtTime(frequency, audioCtx.currentTime);

    gain.gain.setValueAtTime(0.12, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + (durationMs / 1000));

    osc.connect(gain);
    gain.connect(audioCtx.destination);

    osc.start();
    osc.stop(audioCtx.currentTime + (durationMs / 1000));
  } catch (e) {
    console.error("Audio error:", e);
  }
}

function playFeedbackSound(type) {
  if (type === "confirm") {
    playTone(1000, 70);
  } else if (type === "delete") {
    playTone(400, 90);
    setTimeout(() => playTone(350, 90), 100);
  } else if (type === "autocomplete") {
    playTone(1200, 60);
    setTimeout(() => playTone(1500, 80), 80);
  }
}

// ==========================================
// GOOGLE AUTH INITIALIZATION & PROFILE MENU
// ==========================================
function isUserLoggedIn() {
  const user = localStorage.getItem("voxhand_user");
  return user !== null && user !== undefined;
}

function promptLoginRequired() {
  const profileContainer = document.getElementById("userProfile");
  const toast = document.getElementById("authPromptToast");

  if (toast) {
    toast.classList.add("visible");
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      toast.classList.remove("visible");
    }, 3500);
  }

  if (profileContainer) {
    profileContainer.classList.add("open");
    profileContainer.classList.add("auth-attention-pulse");
    setTimeout(() => {
      profileContainer.classList.remove("auth-attention-pulse");
    }, 1600);

    if (!googleBtnRendered) {
      waitForGoogleSdkAndRender();
    }
  }

  const promptText = document.getElementById("menuPromptText");
  if (promptText) {
    const originalText = promptText.innerText;
    promptText.innerText = "⚠️ Please sign in with Google to continue";
    promptText.style.color = "#f59e0b";
    setTimeout(() => {
      promptText.innerText = originalText;
      promptText.style.color = "";
    }, 4500);
  }
}

async function initAuthSystem() {
  const savedUser = localStorage.getItem("voxhand_user");
  if (savedUser) {
    displayUserProfile(JSON.parse(savedUser));
    return;
  }

  displayLoggedOutState();

  if (!GOOGLE_CLIENT_ID) {
    try {
      const res = await fetch(`${API_BASE}/api/config`);
      if (res.ok) {
        const config = await res.json();
        if (config.google_client_id) {
          GOOGLE_CLIENT_ID = config.google_client_id;
          waitForGoogleSdkAndRender();
        }
      }
    } catch (err) {
      console.warn("Could not fetch Google Client ID from backend:", err);
    }
  } else {
    waitForGoogleSdkAndRender();
  }
}

function waitForGoogleSdkAndRender() {
  const checkSdk = () => {
    if (window.google && window.google.accounts && window.google.accounts.id && GOOGLE_CLIENT_ID) {
      renderGoogleButton();
    } else {
      setTimeout(checkSdk, 60);
    }
  };
  checkSdk();
}

function renderGoogleButton() {
  if (!GOOGLE_CLIENT_ID) return;

  google.accounts.id.initialize({
    client_id: GOOGLE_CLIENT_ID,
    callback: handleCredentialResponse,
    auto_select: false
  });

  const btnWrapper = document.getElementById("googleBtnWrapper");
  if (btnWrapper) {
    btnWrapper.innerHTML = "";
    google.accounts.id.renderButton(
      btnWrapper,
      {
        theme: "filled_black",
        size: "medium",
        type: "standard",
        shape: "pill",
        text: "signin_with"
      }
    );
  }

  const modalWrapper = document.getElementById("modalGoogleBtnWrapper");
  if (modalWrapper) {
    modalWrapper.innerHTML = "";
    google.accounts.id.renderButton(
      modalWrapper,
      {
        theme: "filled_black",
        size: "large",
        type: "standard",
        shape: "pill",
        text: "signin_with"
      }
    );
  }

  googleBtnRendered = true;
}

function decodeJwt(token) {
  const base64Url = token.split('.')[1];
  const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
  const jsonPayload = decodeURIComponent(atob(base64).split('').map(function(c) {
    return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
  }).join(''));

  return JSON.parse(jsonPayload);
}

async function handleCredentialResponse(response) {
  const userPayload = decodeJwt(response.credential);
  
  const userData = {
    name: userPayload.name,
    email: userPayload.email,
    picture: userPayload.picture
  };

  try {
    await fetch(`${API_BASE}/api/auth/google`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token: response.credential })
    });
  } catch (err) {
    console.warn("Backend auth verification warning:", err);
  }

  localStorage.removeItem("voxhand_guest_trial_used");
  localStorage.setItem("voxhand_user", JSON.stringify(userData));
  displayUserProfile(userData);
  closeTrialModal();
  resetTrialState();

  const profileContainer = document.getElementById("userProfile");
  if (profileContainer) profileContainer.classList.remove("open");
  
  const toast = document.getElementById("authPromptToast");
  if (toast) toast.classList.remove("visible");
}

function displayUserProfile(user) {
  const container = document.getElementById("userProfile");
  const avatarImg = document.getElementById("userAvatar");
  const defaultIcon = document.getElementById("defaultAvatarIcon");
  const loginSec = document.getElementById("loginSection");
  const logoutSec = document.getElementById("logoutSection");
  const tooltipTitle = document.getElementById("tooltipTitle");
  const tooltipSubtitle = document.getElementById("tooltipSubtitle");
  const toast = document.getElementById("authPromptToast");

  if (toast) toast.classList.remove("visible");

  container.classList.remove("logged-out");
  container.classList.add("logged-in");

  defaultIcon.style.display = "none";
  avatarImg.src = user.picture || "https://lh3.googleusercontent.com/a/default-user";
  avatarImg.style.display = "block";

  tooltipTitle.innerText = user.name || "User";
  tooltipSubtitle.innerText = user.email || "";

  loginSec.style.display = "none";
  logoutSec.style.display = "block";
}

function displayLoggedOutState() {
  const container = document.getElementById("userProfile");
  const avatarImg = document.getElementById("userAvatar");
  const defaultIcon = document.getElementById("defaultAvatarIcon");
  const loginSec = document.getElementById("loginSection");
  const logoutSec = document.getElementById("logoutSection");
  const tooltipTitle = document.getElementById("tooltipTitle");
  const tooltipSubtitle = document.getElementById("tooltipSubtitle");

  container.classList.remove("logged-in");
  container.classList.add("logged-out");

  avatarImg.style.display = "none";
  avatarImg.src = "";
  defaultIcon.style.display = "flex";

  tooltipTitle.innerText = "Sign In";
  tooltipSubtitle.innerText = "Click to connect Google";

  loginSec.style.display = "flex";
  logoutSec.style.display = "none";
}

function toggleProfileMenu(event) {
  if (event) event.stopPropagation();
  const profileContainer = document.getElementById("userProfile");
  if (profileContainer) {
    const willOpen = !profileContainer.classList.contains("open");
    profileContainer.classList.toggle("open");

    if (willOpen && !isUserLoggedIn() && !googleBtnRendered) {
      waitForGoogleSdkAndRender();
    }
  }
}

document.addEventListener("click", (e) => {
  const profileContainer = document.getElementById("userProfile");
  if (profileContainer && profileContainer.classList.contains("open") && !profileContainer.contains(e.target)) {
    profileContainer.classList.remove("open");
  }
});

function handleSignOut(event) {
  if (event) event.stopPropagation();
  localStorage.removeItem("voxhand_user");
  const profile = document.getElementById("userProfile");
  if (profile) profile.classList.remove("open");

  if (isCameraRunning) {
    stopCamera();
  }

  resetTrialState();
  googleBtnRendered = false;
  displayLoggedOutState();
  renderGoogleButton();
}

// ==========================================
// STATUS INDICATOR BADGE UPDATE
// ==========================================
function setServerStatus(online) {
  const dot = document.getElementById("statusDot");
  const text = document.getElementById("statusText");
  if (!dot || !text) return;
  
  if (online) {
    dot.classList.remove("offline");
    dot.classList.add("online");
    text.classList.remove("offline");
    text.classList.add("online");
    text.innerText = "● AI Ready to Translate";
  } else {
    dot.classList.remove("online");
    dot.classList.add("offline");
    text.classList.remove("online");
    text.classList.add("offline");
    text.innerText = "● Connecting to AI Translator...";
  }
}

// ==========================================
// GESTURE GUIDE & TRIAL MODAL CONTROLS
// ==========================================
function openGuideModal() {
  const modal = document.getElementById("guideModal");
  if (modal) modal.classList.add("active");
  pauseTrialCountdown();
}

function closeGuideModal() {
  const modal = document.getElementById("guideModal");
  if (modal) modal.classList.remove("active");
  resumeTrialCountdown();
}

function closeGuideModalOnOutsideClick(event) {
  if (event.target.id === "guideModal") {
    closeGuideModal();
  }
}

function openTrialModal() {
  const modal = document.getElementById("trialExpiredModal");
  if (modal) modal.classList.add("active");
  if (!googleBtnRendered) {
    waitForGoogleSdkAndRender();
  }
}

function closeTrialModal() {
  const modal = document.getElementById("trialExpiredModal");
  if (modal) modal.classList.remove("active");
}

function closeTrialModalOnOutsideClick(event) {
  if (event.target.id === "trialExpiredModal") {
    closeTrialModal();
  }
}

function switchGuideTab(tabName) {
  const tabs = ['alphabets', 'numbers', 'special'];
  
  tabs.forEach(t => {
    const btn = document.getElementById(`tab${t.charAt(0).toUpperCase() + t.slice(1)}`);
    const panel = document.getElementById(`guide${t.charAt(0).toUpperCase() + t.slice(1)}`);
    
    if (btn && panel) {
      if (t === tabName) {
        btn.classList.add("active");
        panel.classList.add("active");
      } else {
        btn.classList.remove("active");
        panel.classList.remove("active");
      }
    }
  });
}

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    closeGuideModal();
    closeTrialModal();
  }
});

// ==========================================
// CAMERA ENUMERATION & SELECTION
// ==========================================
async function getCameraDevices() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(device => device.kind === "videoinput");

    cameraSelect.innerHTML = "";

    if (videoDevices.length === 0) {
      cameraSelect.innerHTML = `<option value="">No Camera Found</option>`;
      return;
    }

    videoDevices.forEach((device, index) => {
      const option = document.createElement("option");
      option.value = device.deviceId;
      option.text = device.label || `Camera ${index + 1}`;
      if (device.deviceId === selectedDeviceId) {
        option.selected = true;
      }
      cameraSelect.appendChild(option);
    });

    if (!selectedDeviceId && videoDevices.length > 0) {
      selectedDeviceId = videoDevices[0].deviceId;
    }
  } catch (err) {
    console.error("Error listing camera devices:", err);
  }
}

cameraSelect.addEventListener("change", async (e) => {
  selectedDeviceId = e.target.value;
  if (isCameraRunning) {
    stopCamera();
    await startCamera();
  }
});

navigator.mediaDevices.addEventListener("devicechange", () => {
  getCameraDevices();
});

// ==========================================
// CAMERA CONTROLS & TRIAL TRIGGER
// ==========================================
async function toggleCamera() {
  initAudio();

  if (isCameraRunning) {
    stopCamera();
    return;
  }

  if (isUserLoggedIn()) {
    await startCamera();
    return;
  }

  if (hasUsedTrial()) {
    openTrialModal();
    return;
  }

  await startCamera();
  startTrialCountdown();
}

async function startCamera() {
  try {
    const constraints = {
      video: {
        facingMode: selectedDeviceId ? undefined : "user",
        deviceId: selectedDeviceId ? { exact: selectedDeviceId } : undefined,
        width: { ideal: 640 },
        height: { ideal: 480 }
      }
    };

    mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = mediaStream;
    isCameraRunning = true;

    placeholder.style.display = "none";
    toggleCamBtn.className = "toggle-cam-btn turn-off";
    toggleCamBtn.innerHTML = "⏹ Stop Camera";
    camStatusIndicator.className = "cam-indicator live";
    camStatusIndicator.innerText = "● LIVE";

    await getCameraDevices();

    video.onloadedmetadata = () => {
      sendFrame();
    };
  } catch (err) {
    console.error("Camera access error:", err);
    alert("Could not access the selected camera. Please check permissions.");
  }
}

function stopCamera() {
  if (mediaStream) {
    mediaStream.getTracks().forEach(track => track.stop());
    mediaStream = null;
  }
  if (loopTimeout) {
    clearTimeout(loopTimeout);
  }
  isCameraRunning = false;
  video.srcObject = null;

  pauseTrialCountdown();

  placeholder.style.display = "flex";
  toggleCamBtn.className = "toggle-cam-btn turn-on";
  toggleCamBtn.innerHTML = "▶ Start Camera";
  camStatusIndicator.className = "cam-indicator off";
  camStatusIndicator.innerText = "● CAMERA OFF";
  document.getElementById("gestureValue").innerText = "--";
}

// ==========================================
// FRAME TRANSMISSION LOOP
// ==========================================
let isProcessingFrame = false;

async function sendFrame() {
  if (!isCameraRunning) return;

  if (!isProcessingFrame && video.readyState === video.HAVE_ENOUGH_DATA) {
    isProcessingFrame = true;

    const vWidth = video.videoWidth || 640;
    const vHeight = video.videoHeight || 480;
    const aspect = vWidth / vHeight;

    if (aspect < 1) {
      canvas.width = 240;
      canvas.height = Math.round(240 / aspect);
    } else {
      canvas.width = 320;
      canvas.height = Math.round(320 / aspect);
    }

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const base64Image = canvas.toDataURL("image/jpeg", 0.4);

    try {
      const response = await fetch(`${API_BASE}/process_frame`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          image: base64Image,
          session_id: sessionId 
        })
      });

      if (response.ok) {
        const data = await response.json();
        updateUI(data);
        setServerStatus(true);
      }
    } catch (err) {
      setServerStatus(false);
    } finally {
      isProcessingFrame = false;
    }
  }

  loopTimeout = setTimeout(sendFrame, 120);
}

function updateUI(data) {
  if (data.action === "confirm") {
    playFeedbackSound("confirm");
  } else if (data.action === "delete") {
    playFeedbackSound("delete");
  } else if (data.action === "speak") {
    playFeedbackSound("confirm");
    const textToSpeak = (data.sentence || "").trim();
    if (textToSpeak) {
      speakText(textToSpeak);
    }
  }

  document.getElementById("gestureValue").innerText = data.gesture || "--";
  document.getElementById("sentenceText").innerText = data.sentence || "";
  document.getElementById("currentWord").innerText = data.current_word || "";

  lastSentence = data.sentence || "";
  lastWord = data.current_word || "";

  const container = document.getElementById("chipsContainer");
  const suggestionsKey = (data.suggestions || []).join("|");

  if (suggestionsKey !== currentSuggestionsCache) {
    currentSuggestionsCache = suggestionsKey;

    if (data.suggestions && data.suggestions.length > 0) {
      container.innerHTML = data.suggestions
        .map(word => `<button class="chip-btn" onclick="chooseWord('${word}')">${word}</button>`)
        .join("");
    } else {
      container.innerHTML = `<span class="no-sugg">Type signs to see suggestions...</span>`;
    }
  }
}

async function setMode(modeKey) {
  try {
    await fetch(`${API_BASE}/set_mode`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ 
        mode: modeKey,
        session_id: sessionId 
      })
    });
    document.querySelectorAll(".mode-btn").forEach((btn, index) => {
      const modeKeys = ['1', '2', '0'];
      btn.classList.toggle("active", modeKey === modeKeys[index]);
    });
  } catch (err) {
    console.error("Mode switch error:", err);
  }
}

async function chooseWord(word) {
  playFeedbackSound("autocomplete");
  try {
    await fetch(`${API_BASE}/autocomplete`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ 
        word: word,
        session_id: sessionId 
      })
    });
  } catch (err) {
    console.error("Autocomplete error:", err);
  }
}

async function clearText() {
  try {
    await fetch(`${API_BASE}/clear`, { 
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId })
    });
    document.getElementById("sentenceText").innerText = "";
    document.getElementById("currentWord").innerText = "";
    currentSuggestionsCache = "";
    lastSentence = "";
    lastWord = "";
  } catch (err) {
    console.error("Clear error:", err);
  }
}

function copyText() {
  const text = (lastSentence + lastWord).trim();
  if (text) {
    navigator.clipboard.writeText(text);
    alert("Copied to clipboard!");
  }
}

function speakText(customText) {
  const text = (typeof customText === 'string' && customText.length > 0) 
    ? customText 
    : (lastSentence + lastWord).trim();
    
  if (!text) return;

  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 0.95;
  window.speechSynthesis.speak(utterance);
}

window.addEventListener("DOMContentLoaded", () => {
  initAuthSystem();
  getCameraDevices();
});