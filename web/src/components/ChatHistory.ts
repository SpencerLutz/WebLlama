import { ChatMessage, MessageType } from './ChatMessage';
import styles from '../styles/components/ChatHistory.module.css';

export class ChatHistory {
    private element: HTMLDivElement;
    
    constructor(elementId: string) {
        const foundElement = document.getElementById(elementId);
        if (!foundElement) {
            throw new Error(`Element with id ${elementId} not found`);
        }
        this.element = foundElement as HTMLDivElement;
        
        // Apply CSS module class
        this.element.className = styles.chatHistory;
    }
    
    public addUserMessage(message: string): void {
        const messageComponent = new ChatMessage(message, MessageType.USER);
        this.element.appendChild(messageComponent.getElement());
        this.scrollToBottom();
    }
    
    public addBotMessage(message: string): void {
        const messageComponent = new ChatMessage(message, MessageType.BOT);
        this.element.appendChild(messageComponent.getElement());
        this.scrollToBottom();
    }

    public createBotMessage(): ChatMessage {
        const messageComponent = new ChatMessage('', MessageType.BOT);
        this.element.appendChild(messageComponent.getElement());
        this.scrollToBottom();
        return messageComponent;
    }
    
    private scrollToBottom(): void {
        this.element.scrollTop = this.element.scrollHeight;
    }
    
    public clear(): void {
        this.element.innerHTML = '';
    }
}