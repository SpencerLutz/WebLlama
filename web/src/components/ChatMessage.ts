import styles from '../styles/components/ChatMessage.module.css';

export enum MessageType {
    USER = 'user',
    BOT = 'bot'
}

export class ChatMessage {
    private element: HTMLDivElement;
    
    constructor(
        private message: string,
        private type: MessageType
    ) {
        this.element = document.createElement('div');
        this.render();
    }
    
    public getElement(): HTMLDivElement {
        return this.element;
    }

    public updateMessage(newText: string): void {
        this.message = newText;
        this.element.textContent = newText;
    }
    
    private render(): void {
        // Apply CSS module classes
        this.element.className = styles.message;
        if (this.type === MessageType.USER) {
            this.element.classList.add(styles.userMessage);
        } else {
            this.element.classList.add(styles.botMessage);
        }
        
        this.element.textContent = this.message;
    }
}