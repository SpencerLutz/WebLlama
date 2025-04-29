// Import global styles
import './styles/global.css';

// Import layout styles
import layoutStyles from './styles/layout.module.css';

// Import HTML template and app
import { WebLlamaApp } from './app';
import chatTemplate from './templates/chat-template.html?raw';

// Get the app container and add layout class
const appContainer = document.querySelector<HTMLDivElement>('#app')!;
appContainer.className = layoutStyles.app;

// Inject template into the DOM
appContainer.innerHTML = chatTemplate;

// Apply container style to the chat container
document.getElementById('chatContainer')!.className = layoutStyles.chatContainer;
document.getElementById('chatHeader')!.className = layoutStyles.header;
document.getElementById('chatTitle')!.className = layoutStyles.title;

// Initialize the application
const app = new WebLlamaApp();