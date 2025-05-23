
const sidebarToggle = document.getElementById('sidebarToggle');
const sidebar = document.getElementById('sidebar');
const mainContainer = document.querySelector('.main-container');

sidebarToggle.addEventListener('click', () => {
    sidebar.classList.toggle('open');
    mainContainer.classList.toggle('shifted');
});


const messageInput = document.getElementById('message');
const sendBtn = document.getElementById('sendBtn');
const chatContainer = document.getElementById('chatContainer');

sendBtn.addEventListener('click', sendMessage);
messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});


async function sendMessage() {
    const message = messageInput.value.trim();
    if (message === "") return;


    const userMessage = document.createElement('div');
    userMessage.className = 'user-message';
    userMessage.textContent = message;
    chatContainer.appendChild(userMessage);

   
    messageInput.value = "";
    chatContainer.scrollTop = chatContainer.scrollHeight;

    
    try {
        const response = await fetch("http://127.0.0.1:5000/generate", { 
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ prompt: message }),
        });

        if (!response.ok) {
            throw new Error("Failed to fetch response from the server");
        }

        const data = await response.json();

       
        const botMessage = document.createElement('div');
        botMessage.className = 'bot-message';
        botMessage.innerHTML = `<p>${data.response}</p>`;
        chatContainer.appendChild(botMessage);

        
        chatContainer.scrollTop = chatContainer.scrollHeight;
    } catch (error) {
        console.error("Error:", error);

        
        const botMessage = document.createElement('div');
        botMessage.className = 'bot-message';
        botMessage.innerHTML = "<p>Sorry, an error occurred while fetching the response.</p>";
        chatContainer.appendChild(botMessage);

        
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
}
