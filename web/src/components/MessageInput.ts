import styles from '../styles/components/MessageInput.module.css';

export class MessageInput {
    private inputElement: HTMLInputElement;
    private sendButtonElement: HTMLButtonElement;
    private containerElement: HTMLDivElement;
    
    constructor(inputId: string, buttonId: string, private onSendCallback: (message: string) => void) {
        const inputElement = document.getElementById(inputId);
        const buttonElement = document.getElementById(buttonId);
        
        if (!inputElement || !buttonElement) {
            throw new Error(`Elements with ids ${inputId} or ${buttonId} not found`);
        }
        
        this.inputElement = inputElement as HTMLInputElement;
        this.sendButtonElement = buttonElement as HTMLButtonElement;
        
        // Get container element (parent of input)
        this.containerElement = this.inputElement.parentElement as HTMLDivElement;
        
        // Apply CSS module classes
        this.containerElement.className = styles.inputContainer;
        this.inputElement.className = styles.userInput;
        this.sendButtonElement.className = styles.sendButton;
        
        this.setupEventListeners();
    }
    
    private setupEventListeners(): void {
        this.sendButtonElement.addEventListener('click', () => {
            this.handleSend();
        });
        
        this.inputElement.addEventListener('keypress', (event) => {
            if (event.key === 'Enter') {
                this.handleSend();
            }
        });
    }
    
    private handleSend(): void {
        const message = this.inputElement.value.trim();
        if (message === '') return;
        
        this.onSendCallback(message);
        this.inputElement.value = '';
    }
    
    public focus(): void {
        this.inputElement.focus();
    }
    
    public setEnabled(enabled: boolean): void {
        if (enabled) {
            this.inputElement.removeAttribute('disabled');
            this.sendButtonElement.removeAttribute('disabled');
        } else {
            this.inputElement.setAttribute('disabled', 'true');
            this.sendButtonElement.setAttribute('disabled', 'true');
        }
    }
}