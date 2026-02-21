document.addEventListener("DOMContentLoaded", () => {
  // Auto-resize textarea + submit

  const textarea = document.querySelector(".chat__message.input");
  const form = document.querySelector(".chat");

  function autoResize() {
    textarea.style.height = "auto";
    textarea.style.height = textarea.scrollHeight + "px";
  }

  textarea.addEventListener("input", autoResize);

  textarea.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit();
    }
  });

  // Sidebar toggle

  const toggleButton = document.getElementById("sidebar-toggle");
  const toggleImg = document.getElementById("toggle-img");
  const sideMenu = document.getElementById("side--menu");

  toggleButton.addEventListener("click", () => {
    sideMenu.classList.toggle("open");

    toggleImg.src = sideMenu.classList.contains("open")
      ? "Asset/sidebar_cl.png"
      : "Asset/sidebar.png";
  });

  // Model selector dropdown

  const modelSelector = document.querySelector(".model-selector");
  const modelToggle = document.getElementById("modelToggle");
  const modelDropdown = document.getElementById("modelDropdown");
  const activeModel = document.getElementById("activeModel");

  modelToggle.addEventListener("click", (e) => {
    e.stopPropagation();
    modelSelector.classList.toggle("open");
  });

  modelDropdown.addEventListener("click", (e) => {
    if (e.target.tagName === "LI") {
      activeModel.textContent = e.target.textContent;
      modelSelector.classList.remove("open");

      const selectedModel = e.target.dataset.model;
      console.log("Selected model:", selectedModel);
    }
  });

  document.addEventListener("click", () => {
    modelSelector.classList.remove("open");
  });
});
