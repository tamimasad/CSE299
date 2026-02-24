document.addEventListener("DOMContentLoaded", () => {
  // --------------------------------------------------------
  // 1. UI ELEMENTS & DESIGN LOGIC
  // --------------------------------------------------------
  const userInput = document.getElementById("userInput");
  const sendBtn = document.getElementById("sendBtn");
  const voiceBtn = document.getElementById("voiceBtn");
  const chatArea = document.getElementById("chatArea");
  const chatList = document.getElementById("chatList");
  const sideMenu = document.getElementById("side--menu");
  const toggleButton = document.getElementById("sidebar-toggle");
  const toggleImg = document.getElementById("toggle-img");

  // Auto-resize textarea as user types
  function autoResize() {
    userInput.style.height = "auto";
    userInput.style.height = userInput.scrollHeight + "px";
  }
  userInput.addEventListener("input", autoResize);

  // Sidebar toggle logic
  toggleButton.addEventListener("click", () => {
    sideMenu.classList.toggle("open");
    toggleImg.src = sideMenu.classList.contains("open")
      ? "/Asset/sidebar_cl.png"
      : "/Asset/sidebar.png";
  });

  // Model selector dropdown logic
  const modelSelector = document.querySelector(".model-selector");
  const modelToggle = document.getElementById("modelToggle");
  const modelDropdown = document.getElementById("modelDropdown");
  const activeModel = document.getElementById("activeModel");
  let currentSelectedModel = "standard";

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

  // --------------------------------------------------------
  // 2. CHAT FUNCTIONALITY LOGIC
  // --------------------------------------------------------
  let currentChat = [];
  let currentSessionIndex = null;

  // Send Message Triggers
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

    // 1. Show user message in the main Chat Area
    addMessage(text, "user");
    currentChat.push({ role: "user", content: text });

    // 2. Clear input box
    userInput.value = "";
    userInput.style.height = "auto";
    userInput.focus();

    // 3. Send to backend API
    try {
      const response = await fetch("http://127.0.0.1:8000/process_text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          model: currentSelectedModel,
        }),
      });

      const data = await response.json();

      // 4. Show bot response in the main Chat Area
      addMessage(data.response, "bot");
      currentChat.push({ role: "bot", content: data.response });

      // 5. Save to history (Local Storage)
      saveSession();
    } catch (error) {
      addMessage("Network Error: Could not connect to the server.", "bot");
    }
  }

  // Helper to add message bubbles to the UI
  function addMessage(text, type) {
    const div = document.createElement("div");
    div.classList.add("message", type);
    div.innerText = text;
    chatArea.appendChild(div);

    // Auto-scroll to the latest message
    chatArea.scrollTop = chatArea.scrollHeight;
  }

  // Save conversation to LocalStorage
  function saveSession() {
    if (currentChat.length === 0) return;

    // Use the first user message as the title
    const titleText = currentChat[0].content;
    const title =
      titleText.substring(0, 20) + (titleText.length > 20 ? "..." : "");

    let sessions = JSON.parse(localStorage.getItem("sessions")) || [];

    if (currentSessionIndex !== null) {
      // Update the current active session
      sessions[currentSessionIndex].messages = currentChat;
    } else {
      // Create a brand new session
      const session = { title: title, messages: currentChat };
      sessions.push(session);
      currentSessionIndex = sessions.length - 1;
    }

    localStorage.setItem("sessions", JSON.stringify(sessions));
    loadSidebar(); // Refresh the sidebar list
  }

  // Render the Sidebar History
  function loadSidebar() {
    chatList.innerHTML = "";
    let sessions = JSON.parse(localStorage.getItem("sessions")) || [];

    // Add "New Chat" button at the top
    const newChatBtn = document.createElement("div");
    newChatBtn.classList.add("history-item");
    newChatBtn.style.background = "rgba(255, 255, 255, 0.15)";
    newChatBtn.style.fontWeight = "bold";
    newChatBtn.innerText = "➕ New Chat";
    newChatBtn.onclick = startNewChat;
    chatList.appendChild(newChatBtn);

    // Add existing sessions
    sessions.forEach((session, index) => {
      const div = document.createElement("div");
      div.classList.add("history-item");
      if (index === currentSessionIndex)
        div.style.borderLeft = "4px solid #F8E3B4";
      div.innerText = session.title;
      div.onclick = () => loadChat(index);
      chatList.appendChild(div);
    });
  }

  // Load a specific chat from history into the main area
  function loadChat(index) {
    let sessions = JSON.parse(localStorage.getItem("sessions")) || [];
    const session = sessions[index];

    currentChat = session.messages;
    currentSessionIndex = index;

    chatArea.innerHTML = ""; // Clear screen before loading history

    session.messages.forEach((msg) => {
      addMessage(msg.content, msg.role);
    });

    // Close sidebar on small screens after selection
    if (window.innerWidth < 768) {
      sideMenu.classList.remove("open");
      toggleImg.src = "/Asset/sidebar.png";
    }
  }

  // Start a fresh conversation
  function startNewChat() {
    currentChat = [];
    currentSessionIndex = null;
    chatArea.innerHTML = "";
    loadSidebar();
  }

  // --------------------------------------------------------
  // 3. VOICE RECORDING LOGIC
  // --------------------------------------------------------
  let mediaRecorder;
  let audioChunks = [];
  let isRecording = false;

  voiceBtn.addEventListener("click", async () => {
    if (isRecording) return;
    isRecording = true;
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
        addMessage("🎙️ Audio message processing...", "user");

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

          currentChat.push({ role: "user", content: "🎙️ [Audio Message]" });
          currentChat.push({ role: "bot", content: data.response });
          saveSession();
        } catch (error) {
          addMessage("Error processing audio on the server.", "bot");
        }

        stream.getTracks().forEach((track) => track.stop());
      });

      // Automatically stop recording after 5 seconds
      setTimeout(() => {
        if (mediaRecorder.state === "recording") mediaRecorder.stop();
      }, 5000);
    } catch (err) {
      isRecording = false;
      voiceBtn.classList.remove("recording");
      alert("Microphone access denied.");
    }
  });

  // Run on page load
  loadSidebar();
});
