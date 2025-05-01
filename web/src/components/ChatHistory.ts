import { ChatMessage, MessageType } from './ChatMessage';
import styles from '../styles/components/ChatHistory.module.css';

interface MessageData {
    text: string;
    type: MessageType;
}

export class ChatHistory {
    private element: HTMLDivElement;
    private messages: MessageData[] = []; // Add this to track messages
    
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
        // Track the message in memory
        this.messages.push({ text: message, type: MessageType.USER });
        
        const messageComponent = new ChatMessage(message, MessageType.USER);
        this.element.appendChild(messageComponent.getElement());
        this.scrollToBottom();
    }
    
    public addBotMessage(message: string): void {
        // Track the message in memory
        this.messages.push({ text: message, type: MessageType.BOT });
        
        const messageComponent = new ChatMessage(message, MessageType.BOT);
        this.element.appendChild(messageComponent.getElement());
        this.scrollToBottom();
    }

    public createBotMessage(): ChatMessage {
        // Create empty message component
        const messageComponent = new ChatMessage('', MessageType.BOT);
        this.element.appendChild(messageComponent.getElement());
        this.scrollToBottom();
        
        // Add empty message to track it (will be updated later)
        this.messages.push({ text: '', type: MessageType.BOT });
        
        return messageComponent;
    }
    
    // Method to update a bot message's content in the message array
    public updateLastBotMessage(message: string): void {
        if (this.messages.length > 0 && 
            this.messages[this.messages.length - 1].type === MessageType.BOT) {
            this.messages[this.messages.length - 1].text = message;
        }
    }
    
    // Method to get complete chat history as a formatted string
    public getFormattedHistory(): string {
        let formattedHistory = '';
        
        for (const message of this.messages) {
            const prefix = message.type === MessageType.USER ? 'User: ' : 'Assistant: ';
            formattedHistory += `${prefix}${message.text}\n`;
        }
        
        return formattedHistory;
    }
    
    private scrollToBottom(): void {
        this.element.scrollTop = this.element.scrollHeight;
    }
    
    public clear(): void {
        this.element.innerHTML = '';
        this.messages = []; // Clear the messages array as well
    }
}