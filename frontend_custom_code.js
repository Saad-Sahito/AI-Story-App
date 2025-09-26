// Simplified WeWeb WebSocket Implementation for Story Generation

// 1. Global WebSocket Manager (Add this as a global JavaScript function in WeWeb)
window.StoryWebSocketManager = class {
    constructor() {
        this.ws = null;
        this.isConnected = false;
        this.currentDecisionIndex = -1;
    }

    async connect(userId, storyId) {
        return new Promise((resolve, reject) => {
            try {
                // Close existing connection if any
                if (this.ws) {
                    this.disconnect();
                }

                // New URL with params
                const wsUrl = `ws://api.whimsera.com/ws/next_chapter/${userId}/${storyId}`;
                this.ws = new WebSocket(wsUrl);

                // Connection opened
                this.ws.onopen = async () => {
                    console.log('WebSocket connected');
                    this.isConnected = true;
                    resolve(true); // ✅ no initData needed
                };

                // Handle incoming messages
                this.ws.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        console.log('Received WebSocket message:', data);

                        // Get current storySegments array
                        const storySegments = wwLib.wwVariable.getValue('storySegments') || [];
                        // ... rest of your logic ...
                    } catch (error) {
                        console.error('Error parsing WebSocket message:', error);
                    }
                };

                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    this.isConnected = false;
                    reject(error);
                };

                this.ws.onclose = () => {
                    console.log('WebSocket connection closed');
                    this.isConnected = false;
                };

            } catch (error) {
                console.error('Error creating WebSocket connection:', error);
                reject(error);
            }
        });
    }


    // Monitor for user_choice changes in the decision object
    startMonitoringDecision(decisionIndex) {
        const checkInterval = setInterval(() => {
            const storySegments = wwLib.wwVariable.getValue('storySegments') || [];
            const decision = storySegments[decisionIndex];
            
            if (decision && decision.user_choice && decision.user_choice !== "") {
                // User made a choice - send it to backend
                this.sendChoice(decision.user_choice);
                clearInterval(checkInterval);
                this.currentDecisionIndex = -1;
            }
            
            // Stop monitoring if connection is closed
            if (!this.isConnected) {
                clearInterval(checkInterval);
            }
        }, 100); // Check every 100ms
    }

    // Send user choice back to backend
    sendChoice(choice) {
        if (!this.ws || !this.isConnected) {
            console.error('WebSocket is not connected');
            return false;
        }

        try {
            const message = { choice: choice };
            this.ws.send(JSON.stringify(message));
            console.log('Sent choice to backend:', choice);
            return true;
        } catch (error) {
            console.error('Error sending choice:', error);
            return false;
        }
    }

    // Disconnect WebSocket
    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
            this.isConnected = false;
            this.currentDecisionIndex = -1;
            console.log('WebSocket disconnected');
        }
    }
};


// 2. WeWeb Workflow for "Generate Next Chapter" Button Click
const generateNextChapter = async () => {
    // Get or create the WebSocket manager instance
    if (!window.storyWSManager) {
        window.storyWSManager = new StoryWebSocketManager();
    }

    // Get WeWeb variables
    const userId = wwLib.wwVariable.getValue('userId');
    const storyId = wwLib.wwVariable.getValue('storyId');
    
    // Show loading state
    wwLib.wwVariable.updateValue('isGenerating', true);

    try {
        // Connect to WebSocket (URL is built inside connect now)
        await window.storyWSManager.connect(userId, storyId);
        
    } catch (error) {
        console.error('Failed to connect WebSocket:', error);
        wwLib.wwVariable.updateValue('isGenerating', false);
        
        wwLib.wwNotification.open({
            text: 'Failed to connect to story generator',
            duration: 5000,
            type: 'error'
        });
    }
};


// 3. Alternative: Manual function to update user choice (if not using automatic monitoring)
// Call this when user selects a choice in your decision box
const updateUserChoice = (decisionIndex, choice) => {
    const storySegments = wwLib.wwVariable.getValue('storySegments') || [];
    if (storySegments[decisionIndex] && storySegments[decisionIndex].type === 'decision') {
        storySegments[decisionIndex].user_choice = choice;
        wwLib.wwVariable.updateValue('storySegments', storySegments);
    }
};

// 4. Cleanup function (call this when leaving the page)
const cleanupWebSocket = () => {
    if (window.storyWSManager) {
        window.storyWSManager.disconnect();
        window.storyWSManager = null;
    }
};

// 5. WeWeb Variables Required:
/*
- userId (String): The current user's ID
- storyId (String): The current story's ID
- storySegments (Array): Array to store all story chunks exactly as received
- isGenerating (Boolean): Loading state for chapter generation
*/

// 6. Usage Notes:
/*
The WebSocket will:
1. Connect when "Generate Next Chapter" is clicked
2. Append chunks exactly as received to storySegments:
   - {"type": "text", "scene_text": "..."}
   - {"type": "decision", "question": "...", "options": N, "user_choice": ""}
3. Monitor decision objects for user_choice changes (from "" to a value)
4. Automatically send the choice back when detected
5. Close connection when {"type": "Chapter Complete!"} is received

Your decision box should modify the user_choice field directly in the storySegments array.
*/