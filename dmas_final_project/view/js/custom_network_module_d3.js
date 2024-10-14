var CustomNetworkModule = function(canvas_width, canvas_height) {
    var self = this;

    // Create main div
    self.main_div = document.createElement("div");
    self.main_div.style.display = "flex";

    // Create network div
    self.network_div = document.createElement("div");
    self.network_div.style.width = canvas_width + "px";
    self.network_div.style.height = canvas_height + "px";
    self.network_div.style.border = "1px solid black";

    // Create agent info div
    self.agent_info_div = document.createElement("div");
    self.agent_info_div.id = "agent-info";
    self.agent_info_div.style.width = "300px";
    self.agent_info_div.style.border = "1px solid #ccc";
    self.agent_info_div.style.padding = "10px";
    self.agent_info_div.style.marginLeft = "10px";
    self.agent_info_div.innerHTML = "<h3>Agent Information</h3><p>Click on an agent to see details here.</p>";

    // Append divs
    self.main_div.appendChild(self.network_div);
    self.main_div.appendChild(self.agent_info_div);

    // Append main div to the page
    var containerElement = document.getElementById("elements");
    containerElement.appendChild(self.main_div);

    // Store data and SVG for later use
    self.data = null;
    self.svg = null;
    self.container = null;  // Store the container group
    self.zoom = null;       // Store the zoom behavior
    self.selected_agent = null;
    self.selected_agent_id = null;

    // Initialize the current zoom transform
    self.current_transform = d3.zoomIdentity; // Default identity transform

    // Create an SVG element
    self.svg = d3.select(self.network_div).append("svg")
        .attr("width", self.network_div.clientWidth)
        .attr("height", self.network_div.clientHeight);

    // Define zoom behavior
    self.zoom = d3.zoom()
        .scaleExtent([0.1, 10])  // Adjust the scale extent as needed
        .on("zoom", function(event) {
            self.container.attr("transform", event.transform);
            self.current_transform = event.transform;  // Save the current transform
        });

    // Apply zoom behavior to the SVG
    self.svg.call(self.zoom);

    // Create a container group for zooming
    self.container = self.svg.append("g");
};

CustomNetworkModule.prototype.render = function(data) {
    var self = this;
    self.data = data;  // Store the data for use in reset

    // Update selected_agent reference
    if (self.selected_agent_id) {
        var updated_agent = data.nodes.find(function(agent) {
            return agent.id === self.selected_agent_id;
        });
        if (updated_agent) {
            self.selected_agent = updated_agent;
        }
    }

    // No need to remove and recreate the SVG or container

    // Update the zoom transform
    self.container.attr("transform", self.current_transform);

    // Initialize simulation and set forces
    var simulation = d3.forceSimulation()
        .nodes(data.nodes)
        .force("link", d3.forceLink(data.edges).id(function(d) { return d.id; }))
        .force("charge", d3.forceManyBody())
        .force("center", d3.forceCenter(self.network_div.clientWidth / 2, self.network_div.clientHeight / 2));

    // Update links
    var link = self.container.selectAll(".links")
        .data([null]);

    link = link.enter()
        .append("g")
        .attr("class", "links")
        .merge(link);

    var linkLines = link.selectAll("line")
        .data(data.edges, function(d) { return d.source.id + "-" + d.target.id; });

    linkLines.exit().remove();

    linkLines = linkLines.enter()
        .append("line")
        .attr("stroke-width", function(d) { return d.width; })
        .attr("stroke", function(d) { return d.color; })
        .merge(linkLines);

    // Update nodes
    var node = self.container.selectAll(".nodes")
        .data([null]);

    node = node.enter()
        .append("g")
        .attr("class", "nodes")
        .merge(node);

    var nodeCircles = node.selectAll("circle")
        .data(data.nodes, function(d) { return d.id; });

    nodeCircles.exit().remove();

    nodeCircles = nodeCircles.enter()
        .append("circle")
        .attr("r", function(d) { return d.size; })
        .attr("fill", function(d) { return d.color; })
        .on("click", function(event, d) {
            self.selected_agent = d;  // Store the entire agent object
            self.selected_agent_id = d.id;  // Optionally keep the ID
            self.updateAgentInfo(d.agent_data);
        })
        .merge(nodeCircles);

    // Run the simulation for a fixed number of ticks and then stop it
    simulation.tick(300);  // Adjust the number of ticks as needed
    simulation.stop();     // Stop the simulation

    // Apply final positions
    linkLines
        .attr("x1", function(d) { return d.source.x; })
        .attr("y1", function(d) { return d.source.y; })
        .attr("x2", function(d) { return d.target.x; })
        .attr("y2", function(d) { return d.target.y; });

    nodeCircles
        .attr("cx", function(d) { return d.x; })
        .attr("cy", function(d) { return d.y; });

    // Continuously update the selected agent's data every half second
    if (!self.updateInterval) {
        self.updateInterval = d3.interval(function() {
            if (self.selected_agent && self.selected_agent.agent_data) {
                self.updateAgentInfo(self.selected_agent.agent_data);
            }
        }, 500);  // Update every 500ms
    }
};

CustomNetworkModule.prototype.updateAgentInfo = function(agentData) {
    var self = this;
    self.agent_info_div.innerHTML = `<h3>Agent ID: ${agentData.id}</h3>
                                     <p>Type: ${agentData.type}</p>`;
    if (agentData.bias !== undefined) {
        self.agent_info_div.innerHTML  += `<p>Bias: ${agentData.bias}</p>`;
    }

    if (agentData.adjustability !== undefined) {
        self.agent_info_div.innerHTML  += `<p>Adjustability: ${agentData.adjustability}</p>`;
    }

    if (agentData.opinion !== undefined) {
        self.agent_info_div.innerHTML  += `<p>Opinion: ${agentData.opinion}</p>`;
    }

    if (agentData.rationality !== undefined) {
        self.agent_info_div.innerHTML  += `<p>Rationality: ${agentData.rationality}</p>`;
    }
};

CustomNetworkModule.prototype.reset = function() {
    var self = this;
    // Clear agent info
    self.agent_info_div.innerHTML = "<h3>Agent Information</h3><p>Click on an agent to see details here.</p>";
    // Clear selected agent
    self.selected_agent = null;
    self.selected_agent_id = null;
    // Reset current transform to identity
    self.current_transform = d3.zoomIdentity;
    // Optionally, re-render with the initial data
    if (self.data) {
        self.render(self.data);
    }
};
