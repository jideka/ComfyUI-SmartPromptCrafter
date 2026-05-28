import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "smart.promptcrafter",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "SmartPromptCrafter") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            this.color       = "#1a1a2e";
            this.bgcolor     = "#16213e";
            this.title_color = "#a78bfa";
            return result;
        };
    },

    async setup() {
        app.api.addEventListener("smart.promptcrafter.status", (event) => {
            const msg = event?.detail?.message;
            if (msg) {
                app.ui?.dialog?.show?.(msg) ?? console.info("[SmartPromptCrafter]", msg);
            }
        });
    },
});
