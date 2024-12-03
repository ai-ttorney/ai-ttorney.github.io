// Sidebar toggle functionality
const sidebarToggle = document.getElementById('sidebarToggle');
const sidebar = document.getElementById('sidebar');
const mainContainer = document.querySelector('.main-container');

sidebarToggle.addEventListener('click', () => {
    sidebar.classList.toggle('open');
    mainContainer.classList.toggle('shifted');
});

// Message Input and Send Functionality
const messageInput = document.getElementById('message');
const sendBtn = document.getElementById('sendBtn');
const chatContainer = document.getElementById('chatContainer');

sendBtn.addEventListener('click', sendMessage);
messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

function sendMessage() {
    const message = messageInput.value.trim();
    if (message === "") return;

    // Create and display the user message
    const userMessage = document.createElement('div');
    userMessage.className = 'user-message';
    userMessage.textContent = message;
    chatContainer.appendChild(userMessage);

    // Clear the input and scroll chat to bottom
    messageInput.value = "";
    chatContainer.scrollTop = chatContainer.scrollHeight;

    // Simulate bot response
    setTimeout(receiveBotMessage, 500);
}

function receiveBotMessage() {
    const botMessage = document.createElement('div');
    botMessage.className = 'bot-message';
    botMessage.innerHTML = "<p>*AITTORNEY Bot answers the question.*</p>";
    chatContainer.appendChild(botMessage);
    
    // Scroll chat to bottom after bot response
    chatContainer.scrollTop = chatContainer.scrollHeight;
}
