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
    var container = document.getElementById("elements");
    container.appendChild(self.main_div);

    // Store data and SVG for later use
    self.data = null;
    self.svg = null;
    self.selected_agent_id = null;
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

    // Remove existing SVG if any
    d3.select(self.network_div).select("svg").remove();

    // Create an SVG element
    self.svg = d3.select(self.network_div).append("svg")
        .attr("width", self.network_div.clientWidth)
        .attr("height", self.network_div.clientHeight);

    // Initialize simulation and set forces
    var simulation = d3.forceSimulation(data.nodes)
        .force("link", d3.forceLink(data.edges).id(function(d) { return d.id; }))
        .force("charge", d3.forceManyBody())
        .force("center", d3.forceCenter(self.network_div.clientWidth / 2, self.network_div.clientHeight / 2));

    // Create links
    var link = self.svg.append("g")
        .attr("class", "links")
        .selectAll("line")
        .data(data.edges)
        .enter().append("line")
        .attr("stroke-width", function(d) { return d.width; })
        .attr("stroke", function(d) { return d.color; });

    // Create nodes
    var node = self.svg.append("g")
        .attr("class", "nodes")
        .selectAll("circle")
        .data(data.nodes)
        .enter().append("circle")
        .attr("r", function(d) { return d.size; })
        .attr("fill", function(d) { return d.color; })
        .on("click", function(event, d) {
            self.selected_agent = d;  // Store the entire agent object
            self.selected_agent_id = d.id;  // Optionally keep the ID
            self.updateAgentInfo(d.agent_data);
        });
    
    // Run the simulation for a fixed number of ticks and then stop it
    simulation.tick(300);  // Adjust the number of ticks as needed
    simulation.stop();     // Stop the simulation

    // Apply final positions
    link
        .attr("x1", function(d) { return d.source.x; })
        .attr("y1", function(d) { return d.source.y; })
        .attr("x2", function(d) { return d.target.x; })
        .attr("y2", function(d) { return d.target.y; });

    node
        .attr("cx", function(d) { return d.x; })
        .attr("cy", function(d) { return d.y; });


    // Continuously log the selected agent's data every second
    d3.interval(function() {
        if (self.selected_agent && self.selected_agent.agent_data) {
            self.updateAgentInfo(self.selected_agent.agent_data);
        }
    }, 500);
    
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
    // Clear the SVG content
    if (self.svg) {
        self.svg.remove();
    }
    // Clear agent info
    self.agent_info_div.innerHTML = "<h3>Agent Information</h3><p>Click on an agent to see details here.</p>";
    // Optionally, re-render with the initial data
    if (self.data) {
        self.render(self.data);
    }
};
