import styles from '../styles/components/ModelControls.module.css';

export class ModelControls {
    private initButtonElement: HTMLButtonElement;
    private containerElement: HTMLDivElement;
    
    constructor(initButtonId: string, private onInitCallback: () => Promise<void>) {
        const initButtonElement = document.getElementById(initButtonId);
        
        if (!initButtonElement) {
            throw new Error(`Element with id ${initButtonId} not found`);
        }
        
        this.initButtonElement = initButtonElement as HTMLButtonElement;
        this.containerElement = this.initButtonElement.parentElement as HTMLDivElement;
        
        // Apply CSS module classes
        this.containerElement.className = styles.controls;
        this.initButtonElement.className = styles.initButton;
        
        this.setupEventListeners();
    }
    
    private setupEventListeners(): void {
        this.initButtonElement.addEventListener('click', async () => {
            await this.onInitCallback();
        });
    }
}