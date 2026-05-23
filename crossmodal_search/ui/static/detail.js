const shell = document.querySelector(".detail-shell");
const cockpitGrid = document.getElementById("cockpit-grid");
const frameTitle = document.getElementById("frame-title");
const frameSubtitle = document.getElementById("frame-subtitle");
const intentEl = document.getElementById("intent");
const captionEl = document.getElementById("caption");
const tagsEl = document.getElementById("tags");
const speedEl = document.getElementById("speed");
const yawEl = document.getElementById("yaw");
const lightEl = document.getElementById("light");
const frameCountEl = document.getElementById("frame-count");
const rangeEl = document.getElementById("frame-range");
const prevButton = document.getElementById("prev-frame");
const nextButton = document.getElementById("next-frame");
const scrubberLabel = document.getElementById("scrubber-label");
const lightbox = document.getElementById("lightbox");
const lightboxImage = document.getElementById("lightbox-image");
const closeLightbox = document.getElementById("close-lightbox");

let detail = null;

function imageUrl(frameName, camera) {
  return `/api/image/${encodeURIComponent(frameName)}/${encodeURIComponent(camera)}`;
}

function formatNumber(value, suffix, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${Number(value).toFixed(digits)}${suffix}`;
}

function chip(label) {
  return `<span class="chip">${label}</span>`;
}

function gridSlot(camera) {
  const slots = {
    FRONT_LEFT: "grid-column: 1; grid-row: 1;",
    FRONT: "grid-column: 2; grid-row: 1;",
    FRONT_RIGHT: "grid-column: 3; grid-row: 1;",
    SIDE_LEFT: "grid-column: 1; grid-row: 2;",
    SIDE_RIGHT: "grid-column: 3; grid-row: 2;",
    REAR_LEFT: "grid-column: 1; grid-row: 3;",
    REAR: "grid-column: 2; grid-row: 3;",
    REAR_RIGHT: "grid-column: 3; grid-row: 3;"
  };
  return slots[camera] || "";
}

function renderCockpit(frameName) {
  const cameras = window.COCKPIT_ORDER || [];
  const firstFour = cameras.slice(0, 5);
  const lastThree = cameras.slice(5);
  const tiles = cameras.map((camera) => `
    <button class="camera-tile" data-camera="${camera}" style="${gridSlot(camera)}" type="button">
      <img src="${imageUrl(frameName, camera)}" alt="${camera}">
      <span>${camera.replaceAll("_", " ")}</span>
    </button>
  `);
  tiles.splice(5, 0, `<div class="camera-tile empty-center" style="grid-column: 2; grid-row: 2;">EGO</div>`);
  cockpitGrid.innerHTML = tiles.join("");

  cockpitGrid.querySelectorAll("button.camera-tile").forEach((button) => {
    button.addEventListener("click", () => {
      const camera = button.dataset.camera;
      lightboxImage.src = imageUrl(frameName, camera);
      lightboxImage.alt = camera;
      lightbox.hidden = false;
    });
  });
}

function currentFrameByPosition(position) {
  if (!detail || !detail.segment_frames) return null;
  return detail.segment_frames.find((frame) => Number(frame.segment_frame_position) === Number(position));
}

function updateScrubber() {
  const position = Number(detail.segment_frame_position || 0);
  const count = Number(detail.segment_frame_count || 1);
  rangeEl.min = 0;
  rangeEl.max = Math.max(0, count - 1);
  rangeEl.value = position;
  prevButton.disabled = position <= 0;
  nextButton.disabled = position >= count - 1;
  scrubberLabel.textContent = `frame ${position + 1} / ${count}`;
}

function renderDetail(payload) {
  detail = payload;
  frameTitle.textContent = payload.frame_name;
  frameSubtitle.textContent = `segment ${payload.segment_id || ""}`;
  intentEl.textContent = payload.intent_name || "-";
  captionEl.textContent = payload.caption || "";
  tagsEl.innerHTML = Array.from(new Set([payload.motion_state].concat(payload.tags || [])))
    .filter((tag) => tag && tag !== "unknown")
    .slice(0, 8)
    .map(chip)
    .join("");
  speedEl.textContent = formatNumber(payload.past_speed_now, " m/s");
  yawEl.textContent = formatNumber(payload.past_yaw_change, " rad", 2);
  lightEl.textContent = formatNumber(payload.image_brightness, "", 0);
  frameCountEl.textContent = `${Number(payload.segment_frame_position || 0) + 1}/${payload.segment_frame_count || 1}`;
  renderCockpit(payload.frame_name);
  updateScrubber();
}

async function loadFrame(frameName, replaceHistory = true) {
  const response = await fetch(`/api/frame/${encodeURIComponent(frameName)}`);
  if (!response.ok) return;
  const payload = await response.json();
  renderDetail(payload);
  if (replaceHistory) {
    history.replaceState({ frameName }, "", `/frame/${encodeURIComponent(frameName)}`);
  }
}

rangeEl.addEventListener("input", () => {
  const frame = currentFrameByPosition(rangeEl.value);
  if (frame && frame.frame_name !== detail.frame_name) {
    loadFrame(frame.frame_name);
  }
});

prevButton.addEventListener("click", () => {
  const position = Math.max(0, Number(detail.segment_frame_position || 0) - 1);
  const frame = currentFrameByPosition(position);
  if (frame) loadFrame(frame.frame_name);
});

nextButton.addEventListener("click", () => {
  const position = Math.min(Number(detail.segment_frame_count || 1) - 1, Number(detail.segment_frame_position || 0) + 1);
  const frame = currentFrameByPosition(position);
  if (frame) loadFrame(frame.frame_name);
});

closeLightbox.addEventListener("click", () => {
  lightbox.hidden = true;
  lightboxImage.removeAttribute("src");
});

lightbox.addEventListener("click", (event) => {
  if (event.target === lightbox) {
    lightbox.hidden = true;
    lightboxImage.removeAttribute("src");
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    lightbox.hidden = true;
    lightboxImage.removeAttribute("src");
  }
});

if (window.INDEX_READY) {
  loadFrame(window.INITIAL_FRAME, false);
}
