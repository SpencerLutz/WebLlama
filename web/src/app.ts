import { Model } from '../../src/Model';
import { ChatHistory } from './components/ChatHistory';
import { MessageInput } from './components/MessageInput';
import { ModelStatus } from './components/ModelStatus';
import { ModelControls } from './components/ModelControls';

export class WebLlamaApp {
    private model?: Model;
    private chatHistory: ChatHistory;
    private modelStatus: ModelStatus;
    private messageInput: MessageInput;
    
    constructor() {
        // Initialize components
        this.chatHistory = new ChatHistory('chatHistory');
        this.modelStatus = new ModelStatus('modelStatus');
        this.messageInput = new MessageInput('userInput', 'sendButton', this.handleUserInput.bind(this));
        
        // Initialize model controls
        new ModelControls('initButton', this.initializeModel.bind(this));
        
        // Disable input until model is initialized
        this.messageInput.setEnabled(false);
    }
    
    private async initializeModel(): Promise<void> {
        this.modelStatus.setStatus('Initializing model...');
        
        try {
            this.model = new Model();
            await this.model.init();
            
            this.modelStatus.setStatus('Model initialized and ready');
            
            // Enable user input after model is initialized
            this.messageInput.setEnabled(true);
            this.messageInput.focus();
        } catch (error) {
            console.error('Error initializing model:', error);
            this.modelStatus.setStatus(`Error initializing model: ${error}`);
        }
    }
    
    private async handleUserInput(message: string): Promise<void> {
        if (!this.model) return;
        
        // Add user message to chat
        this.chatHistory.addUserMessage(message);
        
        // Disable input while processing
        this.messageInput.setEnabled(false);
        this.modelStatus.setStatus('Thinking...');
        
        try {
            // Create an empty bot message that we'll update
            const botMessage = this.chatHistory.createBotMessage();
            
            // Get the entire formatted chat history
            const chatHistory = this.chatHistory.getFormattedHistory();
            
            // Implement inference logic with complete chat history
            const tokGenerator = this.model.generate(chatHistory, 50);
            let response = "";
            
            // Update the message as new tokens arrive
            for await (const next_tok of tokGenerator) {
                response += next_tok;
                botMessage.updateMessage(response);
                this.chatHistory.updateLastBotMessage(response);
            }
            
            this.modelStatus.setStatus('Ready');
        } catch (error) {
            console.error('Error generating response:', error);
            this.chatHistory.addBotMessage(`Error: ${error}`);
            this.modelStatus.setStatus(`Error: ${error}`);
        } finally {
            // Re-enable input
            this.messageInput.setEnabled(true);
            this.messageInput.focus();
        }
    }
}