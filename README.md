<br />
<p align="center">
  <h1 align="center">Agent-Based Modelling of News Media Feedback Dynamics and Opinion Polarization in Social Networks</h1>

  <p align="center">
  </p>
</p>

## About The Project

Explanation of Idea:
A bidirectional interaction model between users and news self-media, considering how user evaluations or complaints affect news self-media and other users opinions;

Interactions between users and media are not just one-directional (from media to users) but bidirectional. This means that users reactions to media content (like evaluations or complaints) would also influence how media agents behave and adjust their content in future interactions. 

User Feedback Influences Media: When users react to news content, these actions would be factored back into how the media agents decide on what content to publish next. For instance, if many users complain about a particular bias in news reporting, the media might adjust to offer more balanced content.

Impact on Other Users: Additionally, the model explores how the changes in media behavior, prompted by user feedback, affect the opinions of other users in the network. This reflects the real-world scenario were media evolution based on audience feedback can alter the broader public perception and discourse.


## Getting started

### Prerequisites
- [Docker v4.25](https://www.docker.com/get-started) or higher (if running docker container).
- [Poetry](https://python-poetry.org/).
## Running
Using docker: Run the docker-compose files to run all relevant services (`docker compose up` or `docker compose up --build`).

You can also set up a virtual environment using Poetry. Poetry can  be installed using `pip`:
```
pip install poetry
```
Then initiate the virtual environment with the required dependencies (see `poetry.lock`, `pyproject.toml`):
```
poetry config virtualenvs.in-project true    # ensures virtual environment is in project
poetry install
```
The virtual environment can be accessed from the shell using:
```
poetry shell
```
IDEs like Pycharm will be able to detect the interpreter of this virtual environment.

## Usage
Agent based model can be run in interactive mode (GUI) or data collection mode. A config file is used to define model parameters.
```
python main.py interactive --config_file <config.json> --network_file <network_file>

python main.py simulation --config_file config.json --steps 2000 --plot_path <path>
```

# License
This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](./LICENSE) file for details.
