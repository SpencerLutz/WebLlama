import styles from '../styles/components/ModelStatus.module.css';

export class ModelStatus {
    private element: HTMLElement;
    
    constructor(elementId: string) {
        const foundElement = document.getElementById(elementId);
        if (!foundElement) {
            throw new Error(`Element with id ${elementId} not found`);
        }
        this.element = foundElement;
        
        // Apply CSS module class
        this.element.className = styles.modelStatus;
    }
    
    public setStatus(status: string): void {
        this.element.textContent = status;
    }
}