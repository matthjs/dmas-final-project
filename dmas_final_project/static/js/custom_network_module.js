console.log("Custom Network Module JS is loaded");

class NetworkModule {
    constructor(width, height) {
        this.width = width;
        this.height = height;
        this.canvas = new fabric.Canvas("network_canvas", {
            width: this.width,
            height: this.height,
        });

        // Create a div for the agent information box
        const infoBox = document.createElement("div");
        infoBox.id = "agent-info-box";
        infoBox.style.border = "1px solid #000";
        infoBox.style.padding = "10px";
        infoBox.style.position = "absolute";
        infoBox.style.right = "20px";
        infoBox.style.top = "20px";
        infoBox.style.width = "300px";
        infoBox.innerHTML = "<b>Agent Information</b><br>Click an agent to view details.";
        document.body.appendChild(infoBox);
    }

    render(data) {
        this.canvas.clear();
        const nodes = data.nodes;
        const edges = data.edges;

        nodes.forEach((node) => {
            const circle = new fabric.Circle({
                radius: node.size,
                fill: node.color,
                left: Math.random() * this.width,
                top: Math.random() * this.height,
            });

            circle.on("mousedown", () => {
                this.updateInfoBox(node.agent_data);  // Update info box with agent data
            });

            this.canvas.add(circle);
        });

        // Draw edges as lines between nodes
        edges.forEach((edge) => {
            const sourceNode = nodes[edge.source];
            const targetNode = nodes[edge.target];
            const line = new fabric.Line(
                [sourceNode.left, sourceNode.top, targetNode.left, targetNode.top],
                {
                    stroke: edge.color,
                }
            );
            this.canvas.add(line);
        });
    }

    updateInfoBox(agentData) {
        const infoBox = document.getElementById("agent-info-box");
        infoBox.innerHTML = `<b>Agent Information:</b><br>ID: ${agentData.ID}<br>
                             Type: ${agentData.Type}<br>
                             Opinion: ${agentData.Opinion}<br>
                             Rationality: ${agentData.Rationality}<br>
                             Bias: ${agentData.Bias}<br>
                             Adjustability: ${agentData.Adjustability}`;
    }
}
