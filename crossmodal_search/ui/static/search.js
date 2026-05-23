const form = document.getElementById("search-form");
const queryInput = document.getElementById("query");
const resultsEl = document.getElementById("results");
const statusLine = document.getElementById("status-line");

function imageUrl(frameName, camera) {
  return `/api/image/${encodeURIComponent(frameName)}/${encodeURIComponent(camera)}`;
}

function chip(label) {
  return `<span class="chip">${label}</span>`;
}

function mosaic(frameName, cameraOrder) {
  return `
    <div class="mosaic">
      ${cameraOrder.map((camera) => `
        <div class="mosaic-cell">
          <img src="${imageUrl(frameName, camera)}" alt="${camera}">
          <span>${camera.replaceAll("_", " ")}</span>
        </div>
      `).join("")}
    </div>
  `;
}

function shortSegment(segmentId) {
  if (!segmentId) return "";
  return segmentId.length > 12 ? `${segmentId.slice(0, 12)}...` : segmentId;
}

function renderResults(results, cameraOrder, emptyMessage = "No matching frames") {
  if (!results.length) {
    resultsEl.innerHTML = `<div class="empty-state">${emptyMessage}</div>`;
    return;
  }

  resultsEl.innerHTML = results.map((result) => {
    const tags = [result.intent_name, result.motion_state]
      .concat(result.tags || [])
      .filter((tag) => tag && tag !== "unknown");
    const uniqueTags = Array.from(new Set(tags)).slice(0, 5);
    const frameNumber = result.frame_display || result.frame_index || 0;
    const frameCount = result.segment_frame_count || 1;
    return `
      <a class="result-card" href="/frame/${encodeURIComponent(result.frame_name)}">
        ${mosaic(result.frame_name, cameraOrder)}
        <div class="card-body">
          <p class="card-caption">${result.caption || ""}</p>
          <div class="chips">${uniqueTags.map(chip).join("")}</div>
          <div class="card-meta">
            <span>segment ${shortSegment(result.segment_id)}</span>
            <span>frame ${frameNumber} of ${frameCount}</span>
          </div>
        </div>
      </a>
    `;
  }).join("");
}

async function runSearch() {
  if (!window.INDEX_READY) return;
  const query = queryInput.value.trim();
  statusLine.textContent = "Searching...";
  const response = await fetch(`/api/search?q=${encodeURIComponent(query)}&k=12`);
  if (!response.ok) {
    statusLine.textContent = "Search index unavailable";
    return;
  }
  const payload = await response.json();
  const warning = payload.warnings && payload.warnings.length ? payload.warnings[0] : "";
  renderResults(payload.results, payload.camera_order || window.CAMERA_ORDER, warning || "No matching frames");
  statusLine.textContent = warning || `${payload.results.length} segment results (${payload.mode})`;
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  runSearch();
});

if (window.INDEX_READY) {
  runSearch();
}
