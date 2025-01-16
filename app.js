const sendButton = document.getElementById("sendButton");
const userInput = document.getElementById("userInput");
const chatOutput = document.getElementById("chat-output");

function addMessage(content, sender) {
  const message = document.createElement("div");
  message.classList.add(sender);
  message.textContent = content;
  chatOutput.appendChild(message);
  chatOutput.scrollTop = chatOutput.scrollHeight;
}

sendButton.addEventListener("click", async () => {
  const text = userInput.value.trim();
  if (!text) return;

  addMessage(text, "user");
  userInput.value = "";

  addMessage("Thinking...", "ai");

  try {
    const response = await fetch("http://localhost:5000/predict", {  // Update this URL after deployment to Render
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const data = await response.json();
    chatOutput.lastChild.textContent = data.response || "No response available.";
  } catch (error) {
    chatOutput.lastChild.textContent = "Error: Unable to connect.";
  }
});
