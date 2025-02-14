export class WaveVisualizer {
    constructor(playerState) {
        this.playerState = playerState;
        this.wave = null;
        this.progressBar = null;
        this.container = document.getElementById('wave-container');
        
        this.setupWave();
        this.setupProgressBar();
        this.setupStateSubscription();
    }

    setupWave() {
        this.wave = new SiriWave({
            container: this.container,
            style: 'ios9',
            width: this.container.clientWidth,
            height: 100,  // Increased height
            autostart: false,
            amplitude: 1,
            speed: 0.1
        });

        // Handle window resize
        window.addEventListener('resize', () => {
            if (this.wave) {
                this.wave.width = this.container.clientWidth;
            }
        });
    }

    setupProgressBar() {
        this.progressBar = document.createElement('progress');
        this.progressBar.max = 100;
        this.progressBar.value = 0;
        this.progressBar.className = 'generation-progress';
        // Insert inside wave-container at the bottom
        this.container.appendChild(this.progressBar);
        this.progressBar.style.display = 'none';
    }

    setupStateSubscription() {
        this.playerState.subscribe(state => {
            // Handle generation progress
            if (state.isGenerating) {
                this.progressBar.style.display = 'block';
                this.progressBar.value = state.progress;
            } else if (state.progress >= 100) {
                // Hide progress bar after completion
                setTimeout(() => {
                    this.progressBar.style.display = 'none';
                    this.progressBar.value = 0;
                }, 500);
            }

            // Only animate when playing, stop otherwise
            if (state.isPlaying) {
                this.wave.start();
            } else {
                this.wave.stop();
            }
        });
    }

    updateProgress(receivedChunks, totalChunks) {
        if (!totalChunks) return;
        
        // Calculate progress percentage based on chunks
        const progress = Math.min((receivedChunks / totalChunks) * 100, 99);
        
        // Always update on 0 progress or when progress increases
        if (receivedChunks === 0 || progress > this.progressBar.value) {
            this.progressBar.style.display = 'block';
            this.progressBar.value = progress;
            this.playerState.setProgress(receivedChunks, totalChunks);
        }
    }

    cleanup() {
        if (this.wave) {
            this.wave.stop();
            this.wave.dispose();
            this.wave = null;
        }
        
        if (this.progressBar) {
            this.progressBar.style.display = 'none';
            this.progressBar.value = 0;
            if (this.progressBar.parentNode) {
                this.progressBar.parentNode.removeChild(this.progressBar);
            }
            this.progressBar = null;
        }
        
        // Re-setup wave and progress bar
        this.setupWave();
        this.setupProgressBar();
        
        if (this.playerState) {
            this.playerState.setProgress(0, 1); // Reset progress in state
        }
    }
}

export default WaveVisualizer;
