document.addEventListener("DOMContentLoaded", () => {
  const userInput = document.getElementById("userInput");
  const sendBtn = document.getElementById("sendBtn");
  const voiceBtn = document.getElementById("voiceBtn");
  const chatArea = document.getElementById("chatArea");
  const chatList = document.getElementById("chatList");
  const sideMenu = document.getElementById("side--menu");
  const toggleButton = document.getElementById("sidebar-toggle");

  // Model selector elements
  const activeModel = document.getElementById("activeModel");
  const modelToggle = document.getElementById("modelToggle");
  const modelSelectorDiv = document.getElementById("modelSelectorDiv");
  const modelOptions = document.querySelectorAll("#modelDropdown li");

  let currentChat = [];
  let currentSessionFilename = null;
  let mediaRecorder;
  let audioChunks = [];
  let isRecording = false;
  let currentSelectedModel = "banglat5"; // Default model

  // --- Model Selection Logic ---
  modelToggle.addEventListener("click", () => {
    modelSelectorDiv.classList.toggle("open");
  });

  modelOptions.forEach((option) => {
    option.addEventListener("click", () => {
      currentSelectedModel = option.getAttribute("data-model");
      activeModel.innerText = option.innerText;
      modelSelectorDiv.classList.remove("open");
    });
  });

  // Close dropdown if clicked outside
  document.addEventListener("click", (e) => {
    if (!modelSelectorDiv.contains(e.target)) {
      modelSelectorDiv.classList.remove("open");
    }
  });

  function addMessage(text, type) {
    const div = document.createElement("div");
    div.classList.add("message", type);
    div.innerText = text;
    chatArea.appendChild(div);
    chatArea.scrollTop = chatArea.scrollHeight;
    return div;
  }

  // Core Message Sending
  async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    userInput.value = "";
    userInput.style.height = "auto";

    addMessage(text, "user");
    currentChat.push({ role: "user", content: text });

    // loading placeholder
    const botLoadingMsg = addMessage("...", "bot");

    try {
      const response = await fetch("http://127.0.0.1:8000/process_text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, model: currentSelectedModel }),
      });
      const data = await response.json();

      botLoadingMsg.innerText = data.response;
      currentChat.push({ role: "bot", content: data.response });
      await saveSession();
    } catch (error) {
      botLoadingMsg.innerText = "Error connecting to server.";
    }
  }

  // Voice Logic
  async function toggleRecording() {
    if (!isRecording) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: true,
        });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
        mediaRecorder.onstop = async () => {
          const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
          await sendVoiceMessage(audioBlob);
        };
        mediaRecorder.start();
        isRecording = true;
        voiceBtn.classList.add("recording");
      } catch (err) {
        alert("Mic access denied.");
      }
    } else {
      mediaRecorder.stop();
      isRecording = false;
      voiceBtn.classList.remove("recording");
      mediaRecorder.stream.getTracks().forEach((t) => t.stop());
    }
  }

  async function sendVoiceMessage(audioBlob) {
    const userMsgPlaceholder = addMessage("🎙️ Transcribing...", "user");
    const botLoadingMsg = addMessage("...", "bot");

    const formData = new FormData();
    formData.append("audio", audioBlob);
    formData.append("model", currentSelectedModel);

    try {
      const response = await fetch("http://127.0.0.1:8000/process_voice", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();

      userMsgPlaceholder.innerText = data.transcribed_text;
      botLoadingMsg.innerText = data.response;

      currentChat.push({ role: "user", content: data.transcribed_text });
      currentChat.push({ role: "bot", content: data.response });
      await saveSession();
    } catch (error) {
      userMsgPlaceholder.innerText = "Voice error.";
      botLoadingMsg.remove();
    }
  }

  // JSON File / History Logic
  async function saveSession() {
    if (currentChat.length === 0) return;
    const title = currentChat[0].content.substring(0, 20) + "...";
    const res = await fetch("http://127.0.0.1:8000/save_session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title,
        messages: currentChat,
        filename: currentSessionFilename,
      }),
    });
    const data = await res.json();
    currentSessionFilename = data.filename;
    loadSessions();
  }

  async function loadSessions() {
    const res = await fetch("http://127.0.0.1:8000/get_sessions");
    const sessions = await res.json();
    chatList.innerHTML = `<div class="history-item" style="font-weight:bold" id="newChatBtn">➕ Start New Chat</div>`;
    document.getElementById("newChatBtn").onclick = startNewChat;

    sessions.forEach((s) => {
      const div = document.createElement("div");
      div.className = "history-item";
      div.innerText = s.title;
      div.onclick = () => loadChat(s.filename);
      chatList.appendChild(div);
    });
  }

  async function loadChat(filename) {
    const res = await fetch(`http://127.0.0.1:8000/get_session/${filename}`);
    const data = await res.json();
    currentChat = data.messages;
    currentSessionFilename = filename;
    chatArea.innerHTML = "";
    data.messages.forEach((m) => addMessage(m.content, m.role));
  }

  function startNewChat() {
    currentChat = [];
    currentSessionFilename = null;
    chatArea.innerHTML = "";
  }

  sendBtn.onclick = sendMessage;
  voiceBtn.onclick = toggleRecording;
  userInput.onkeydown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };
  toggleButton.onclick = () => sideMenu.classList.toggle("open");

  loadSessions();
});
