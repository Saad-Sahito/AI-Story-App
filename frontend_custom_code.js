window.StoryWebSocketManager = class {
    constructor() {
        this.ws = null;
        this.isConnected = false;
        this.currentDecisionIndex = -1;
    }

    async connect(userId, storyId) {
        return new Promise((resolve, reject) => {
            try {
                if (this.ws) this.disconnect();

                // Always use wss:// if your site is HTTPS
                const wsUrl = `wss://api.whimsera.com/ws/next_chapter/${userId}/${storyId}`;
                this.ws = new WebSocket(wsUrl);

                this.ws.onopen = () => {
                    console.log("✅ WebSocket connected:", wsUrl);
                    this.isConnected = true;
                    resolve(true);
                };

                this.ws.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        console.log("📩 Received:", data);

                        const storySegments = wwLib.wwVariable.getValue("storySegments") || [];

                        if (data.error) {
                            wwLib.wwNotification.open({
                                text: `Error: ${data.error}`,
                                type: "error",
                                duration: 5000
                            });
                        } else if (data.chapter_complete) {
                            this.disconnect();
                            wwLib.wwVariable.updateValue("isGenerating", false);
                            wwLib.wwNotification.open({
                                text: "Chapter Complete!",
                                type: "success",
                                duration: 3000
                            });
                        } else {
                            storySegments.push(data);
                            wwLib.wwVariable.updateValue("storySegments", storySegments);

                            if (data.type === "decision") {
                                this.currentDecisionIndex = storySegments.length - 1;
                                this.startMonitoringDecision(this.currentDecisionIndex);
                            }
                        }
                    } catch (err) {
                        console.error("❌ Error parsing message:", err);
                    }
                };

                this.ws.onerror = (err) => {
                    console.error("❌ WebSocket error:", err);
                    this.isConnected = false;
                    reject(err);
                };

                this.ws.onclose = () => {
                    console.log("🔌 WebSocket closed");
                    this.isConnected = false;
                };
            } catch (err) {
                reject(err);
            }
        });
    }

    sendChoice(choice) {
        if (!this.ws || !this.isConnected) {
            console.error("⚠️ WebSocket not connected");
            return;
        }
        const msg = { choice };
        this.ws.send(JSON.stringify(msg));
        console.log("📤 Sent choice:", choice);
    }

    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
            this.isConnected = false;
        }
    }
};


const generateNextChapter = async () => {
    if (!window.storyWSManager) {
        window.storyWSManager = new StoryWebSocketManager();
    }

    const userId = wwLib.wwVariable.getValue("userId");
    const storyId = wwLib.wwVariable.getValue("storyId");

    wwLib.wwVariable.updateValue("isGenerating", true);

    try {
        await window.storyWSManager.connect(userId, storyId);
    } catch (err) {
        wwLib.wwVariable.updateValue("isGenerating", false);
        wwLib.wwNotification.open({
            text: "Failed to connect to story generator",
            type: "error",
            duration: 5000
        });
    }
};
