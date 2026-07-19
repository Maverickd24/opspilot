const API_BASE = ""; // same-origin: backend serves this file too

const state = {
  sessionId: null,
  documents: [],
};

const els = {
  docList: document.getElementById("doc-list"),
  docEmpty: document.getElementById("doc-empty"),
  docCount: document.getElementById("doc-count"),
  uploader: document.getElementById("uploader"),
  fileInput: document.getElementById("file-input"),
  logBody: document.getElementById("log-body"),
  composer: document.getElementById("composer"),
  messageInput: document.getElementById("message-input"),
  sendBtn: document.getElementById("send-btn"),
  statusLine: document.getElementById("status-line"),
  sessionIdDisplay: document.getElementById("session-id-display"),
  newSessionBtn: document.getElementById("new-session-btn"),
};

init();

async function init() {
  await ensureSession();
  bindEvents();
}

async function ensureSession() {
  const existing = localStorage.getItem("opspilot_session_id");
  if (existing) {
    state.sessionId = existing;
    try {
      const res = await fetch(`${API_BASE}/api/sessions/${existing}`);
      if (res.ok) {
        const data = await res.json();
        state.documents = data.documents;
        renderDocs();
        data.history.forEach((turn) => {
          appendEntry(turn.role === "user" ? "user" : "assistant", turn.content, []);
        });
        updateStatus();
        els.sessionIdDisplay.textContent = `Session ${existing.slice(0, 8)}`;
        return;
      }
    } catch (e) {
      // fall through to creating a new session
    }
  }
  await createSession();
}

async function createSession() {
  const res = await fetch(`${API_BASE}/api/sessions`, { method: "POST" });
  const data = await res.json();
  state.sessionId = data.session_id;
  state.documents = [];
  localStorage.setItem("opspilot_session_id", state.sessionId);
  els.sessionIdDisplay.textContent = `Session ${state.sessionId.slice(0, 8)}`;
  renderDocs();
  updateStatus();
}

function bindEvents() {
  els.uploader.addEventListener("click", (e) => {
    // label already triggers input click; nothing extra needed
  });

  els.fileInput.addEventListener("change", async (e) => {
    const files = Array.from(e.target.files || []);
    if (files.length) await uploadFiles(files);
    els.fileInput.value = "";
  });

  ["dragover", "dragenter"].forEach((evt) =>
    els.uploader.addEventListener(evt, (e) => {
      e.preventDefault();
      els.uploader.style.borderColor = "var(--signal)";
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    els.uploader.addEventListener(evt, (e) => {
      e.preventDefault();
      els.uploader.style.borderColor = "";
    })
  );
  els.uploader.addEventListener("drop", async (e) => {
    const files = Array.from(e.dataTransfer.files || []).filter(
      (f) => f.type === "application/pdf"
    );
    if (files.length) await uploadFiles(files);
  });

  els.composer.addEventListener("submit", async (e) => {
    e.preventDefault();
    const message = els.messageInput.value.trim();
    if (!message) return;
    els.messageInput.value = "";
    autoGrow();
    await sendMessage(message);
  });

  els.messageInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      els.composer.requestSubmit();
    }
  });
  els.messageInput.addEventListener("input", autoGrow);

  els.newSessionBtn.addEventListener("click", async () => {
    if (!confirm("Clear this session? Uploaded documents and chat history will be lost.")) return;
    if (state.sessionId) {
      try {
        await fetch(`${API_BASE}/api/sessions/${state.sessionId}`, { method: "DELETE" });
      } catch (e) {
        /* best effort */
      }
    }
    localStorage.removeItem("opspilot_session_id");
    els.logBody.innerHTML = "";
    await createSession();
  });
}

function autoGrow() {
  els.messageInput.style.height = "auto";
  els.messageInput.style.height = Math.min(els.messageInput.scrollHeight, 140) + "px";
}

async function uploadFiles(files) {
  const tooMany = files.length > 5;
  if (tooMany) {
    appendEntry("error", "You can upload at most 5 files at a time. Please select fewer files.", []);
    return;
  }

  const pendingIds = files.map((f) => `pending-${f.name}-${Date.now()}`);
  files.forEach((f, i) => addPendingDocRow(pendingIds[i], f.name));

  const formData = new FormData();
  formData.append("session_id", state.sessionId);
  files.forEach((f) => formData.append("files", f));

  updateStatus("Uploading and indexing documents…");

  try {
    const res = await fetch(`${API_BASE}/api/documents/upload`, {
      method: "POST",
      body: formData,
    });
    const data = await res.json();

    if (!res.ok) {
      pendingIds.forEach((id) => markPendingError(id, data.detail || "Upload failed."));
      updateStatus();
      return;
    }

    state.documents = data.documents;
    pendingIds.forEach((id) => removeRow(id));
    renderDocs();
    updateStatus();
    els.messageInput.disabled = false;
    els.sendBtn.disabled = false;
    els.messageInput.placeholder = "Ask about the loaded documents…";
  } catch (err) {
    pendingIds.forEach((id) => markPendingError(id, "Network error — is the backend reachable?"));
    updateStatus();
  }
}

async function sendMessage(message) {
  appendEntry("user", message, []);
  const typingId = appendTyping();

  els.sendBtn.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.sessionId, message }),
    });
    const data = await res.json();
    removeRow(typingId);

    if (!res.ok) {
      appendEntry("error", data.detail || "Something went wrong answering that.", []);
    } else {
      appendEntry("assistant", data.answer, data.citations || []);
    }
  } catch (err) {
    removeRow(typingId);
    appendEntry("error", "Network error — couldn't reach the assistant. Please try again.", []);
  } finally {
    els.sendBtn.disabled = false;
    els.messageInput.focus();
  }
}

function renderDocs() {
  els.docList.innerHTML = "";
  els.docEmpty.style.display = state.documents.length ? "none" : "block";
  els.docCount.textContent = `${state.documents.length} loaded`;

  state.documents.forEach((doc) => {
    const li = document.createElement("li");
    li.className = "doc-item";
    li.innerHTML = `
      <span class="doc-name">${escapeHtml(doc.filename)}</span>
      <span class="doc-meta">${doc.page_count} pages · ${doc.chunk_count} chunks</span>
    `;
    els.docList.appendChild(li);
  });

  els.messageInput.disabled = state.documents.length === 0;
  els.sendBtn.disabled = state.documents.length === 0;
}

function addPendingDocRow(id, filename) {
  const li = document.createElement("li");
  li.className = "doc-item pending";
  li.id = id;
  li.innerHTML = `
    <span class="doc-name">${escapeHtml(filename)}</span>
    <span class="doc-meta">Indexing…</span>
  `;
  els.docEmpty.style.display = "none";
  els.docList.appendChild(li);
}

function markPendingError(id, message) {
  const row = document.getElementById(id);
  if (!row) return;
  row.classList.remove("pending");
  row.classList.add("error");
  row.querySelector(".doc-meta").textContent = message;
}

function removeRow(id) {
  const row = document.getElementById(id);
  if (row) row.remove();
}

function appendTyping() {
  const id = `typing-${Date.now()}`;
  const div = document.createElement("div");
  div.className = "log-entry assistant";
  div.id = id;
  div.innerHTML = `
    <div class="stamp">OPSPILOT</div>
    <div class="entry-content typing-indicator">Searching the manifest…</div>
  `;
  els.logBody.appendChild(div);
  els.logBody.scrollTop = els.logBody.scrollHeight;
  return id;
}

function appendEntry(role, text, citations) {
  const div = document.createElement("div");
  div.className = `log-entry ${role}`;

  const stampLabel = role === "user" ? "YOU" : role === "error" ? "ERROR" : "OPSPILOT";

  const citationHtml =
    citations && citations.length
      ? `<div class="citations">${citations
          .map(
            (c) =>
              `<span class="citation-tag" title="${escapeHtml(c.snippet)}">${escapeHtml(
                c.filename
              )} · p.${c.page}</span>`
          )
          .join("")}</div>`
      : "";

  div.innerHTML = `
    <div class="stamp">${stampLabel}</div>
    <div class="entry-content">
      <p>${escapeHtml(text).replace(/\n/g, "<br>")}</p>
      ${citationHtml}
    </div>
  `;
  els.logBody.appendChild(div);
  els.logBody.scrollTop = els.logBody.scrollHeight;
}

function updateStatus(overrideText) {
  if (overrideText) {
    els.statusLine.textContent = overrideText;
    return;
  }
  els.statusLine.textContent = state.documents.length
    ? `${state.documents.length} document${state.documents.length === 1 ? "" : "s"} loaded — ready for questions`
    : "Waiting for documents…";
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
