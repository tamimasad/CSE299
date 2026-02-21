document.addEventListener("DOMContentLoaded", () => {
  //UI & DESIGN LOGIC

  const userInput = document.getElementById("userInput");
  const sendBtn = document.getElementById("sendBtn");
  const voiceBtn = document.getElementById("voiceBtn");
  const chatArea = document.getElementById("chatArea");
  const chatList = document.getElementById("chatList");

  // Auto-resize textarea
  function autoResize() {
    userInput.style.height = "auto";
    userInput.style.height = userInput.scrollHeight + "px";
  }
  userInput.addEventListener("input", autoResize);

  // Sidebar toggle
  const toggleButton = document.getElementById("sidebar-toggle");
  const toggleImg = document.getElementById("toggle-img");
  const sideMenu = document.getElementById("side--menu");

  toggleButton.addEventListener("click", () => {
    sideMenu.classList.toggle("open");
    toggleImg.src = sideMenu.classList.contains("open")
      ? "/Asset/sidebar_cl.png"
      : "/Asset/sidebar.png";
  });

  // Model selector dropdown
  const modelSelector = document.querySelector(".model-selector");
  const modelToggle = document.getElementById("modelToggle");
  const modelDropdown = document.getElementById("modelDropdown");
  const activeModel = document.getElementById("activeModel");
  let currentSelectedModel = "standard"; // Track selected model for API

  modelToggle.addEventListener("click", (e) => {
    e.stopPropagation();
    modelSelector.classList.toggle("open");
  });

  modelDropdown.addEventListener("click", (e) => {
    if (e.target.tagName === "LI") {
      activeModel.textContent = e.target.textContent;
      modelSelector.classList.remove("open");
      currentSelectedModel = e.target.dataset.model;
    }
  });

  document.addEventListener("click", () => {
    modelSelector.classList.remove("open");
  });

  //CHAT FUNCTIONALITY LOGIC

  let currentChat = [];
  let currentSessionIndex = null; // Tracks if we are adding to an existing session

  // Send Message Trigger
  sendBtn.addEventListener("click", sendMessage);

  userInput.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    // Show user message in UI
    addMessage(text, "user");
    currentChat.push({ role: "user", content: text });

    // Reset Input
    userInput.value = "";
    userInput.style.height = "auto";
    userInput.focus();

    // Send to backend API
    try {
      const response = await fetch("http://127.0.0.1:8000/process_text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          model: currentSelectedModel, // Pass model choice to backend
        }),
      });

      const data = await response.json();

      // Show bot response
      addMessage(data.response, "bot");
      currentChat.push({ role: "bot", content: data.response });

      saveSession();
    } catch (error) {
      addMessage("Network Error: Could not connect to the server.", "bot");
    }
  }

  // Add Message to UI
  function addMessage(text, type) {
    const div = document.createElement("div");
    div.classList.add("message", type);
    div.innerText = text;
    chatArea.appendChild(div);

    // Auto scroll to bottom
    chatArea.scrollTop = chatArea.scrollHeight;
  }

  // Save Chat Session (LocalStorage)
  function saveSession() {
    if (currentChat.length === 0) return;

    // Generate title from first message
    const titleText = currentChat[0].content;
    const title =
      titleText.substring(0, 20) + (titleText.length > 20 ? "..." : "");

    let sessions = JSON.parse(localStorage.getItem("sessions")) || [];

    if (currentSessionIndex !== null) {
      // Update existing session
      sessions[currentSessionIndex].messages = currentChat;
    } else {
      // Create new session
      const session = { title: title, messages: currentChat };
      sessions.push(session);
      currentSessionIndex = sessions.length - 1; // Mark as active session
    }

    localStorage.setItem("sessions", JSON.stringify(sessions));
    loadSessions();
  }

  // Load Sessions to Sidebar
  function loadSessions() {
    chatList.innerHTML = "";
    let sessions = JSON.parse(localStorage.getItem("sessions")) || [];

    sessions.forEach((session, index) => {
      const div = document.createElement("div");
      div.classList.add("history-item");
      div.innerText = session.title;
      div.onclick = () => loadChat(index);
      chatList.appendChild(div);
    });
  }

  // Load Selected Chat
  function loadChat(index) {
    let sessions = JSON.parse(localStorage.getItem("sessions")) || [];
    const session = sessions[index];

    currentChat = session.messages;
    currentSessionIndex = index;

    chatArea.innerHTML = "";

    session.messages.forEach((msg) => {
      addMessage(msg.content, msg.role);
    });

    // Auto close sidebar on mobile (optional UX)
    if (window.innerWidth < 768) {
      sideMenu.classList.remove("open");
      toggleImg.src = "/Asset/sidebar.png";
    }
  }

  //VOICE RECORDING

  let mediaRecorder;
  let audioChunks = [];
  let isRecording = false;

  voiceBtn.addEventListener("click", async () => {
    if (isRecording) return; // Prevent multiple recording instances
    isRecording = true;

    // Add visual recording queue
    voiceBtn.classList.add("recording");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.start();
      audioChunks = [];

      mediaRecorder.addEventListener("dataavailable", (event) => {
        audioChunks.push(event.data);
      });

      mediaRecorder.addEventListener("stop", async () => {
        isRecording = false;
        voiceBtn.classList.remove("recording");

        const audioBlob = new Blob(audioChunks, { type: "audio/wav" });

        // UI feedback
        addMessage("🎙️ Audio message processing...", "user");

        // Send audio to backend
        const formData = new FormData();
        formData.append("file", audioBlob);
        formData.append("model", currentSelectedModel);

        try {
          const response = await fetch("http://127.0.0.1:8000/process_audio", {
            method: "POST",
            body: formData,
          });

          const data = await response.json();
          addMessage(data.response, "bot");

          // Save placeholder info to history
          currentChat.push({ role: "user", content: "🎙️ [Voice Audio]" });
          currentChat.push({ role: "bot", content: data.response });
          saveSession();
        } catch (error) {
          addMessage("Error processing audio on the server.", "bot");
        }

        // Close mic tracks to release permissions
        stream.getTracks().forEach((track) => track.stop());
      });

      // Stop recording after 5 seconds
      setTimeout(() => {
        if (mediaRecorder.state === "recording") {
          mediaRecorder.stop();
        }
      }, 5000);
    } catch (err) {
      isRecording = false;
      voiceBtn.classList.remove("recording");
      alert("Microphone access denied or unavailable.");
    }
  });

  // Initial Load
  loadSessions();
});
