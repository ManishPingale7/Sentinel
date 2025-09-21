# Social Media Disaster Simulator

This project contains a Python-based simulator for generating synthetic social media data related to various natural disasters. It is designed to create realistic-looking posts for platforms like Twitter, Facebook, Instagram, YouTube, and news outlets.

## Features

- **Multi-platform Simulation**: Generates data for Twitter, Facebook, Instagram, YouTube, and News platforms.
- **Multi-lingual Support**: Creates content in multiple languages including English, Hindi, Marathi, Tamil, Telugu, and Konkani.
- **Diverse Content Types**: Simulates different types of posts:
    - **Hazard Posts**: Direct reports and warnings about disasters.
    - **False Alarms**: Misinformation and unverified rumors.
    - **Noise**: Irrelevant or joke posts.
- **Realistic User Simulation**: Generates user profiles with varying follower counts, verification status, and bios.
- **Code-Switching**: Simulates posts that mix languages (e.g., Hinglish).
- **Rich Media Simulation**: Associates posts with relevant images.

## File Structure

- `Simulate data/`: Contains the core simulator logic and data.
  - `simulator.py`: The main script that runs the simulation.
  - `simulator_data.py`: Contains all the static data, templates, and translations used by the simulator.
- `Media/`: Contains images used in the generated social media posts.
- `myenv/`: Python virtual environment for the project.
- `Scraping scrapped/`: Contains scripts for scraping data from social media platforms.
- `requirements.txt`: A list of Python dependencies required to run the project.
- `README.md`: This file.

## How to Run the Simulator

1.  **Set up the environment**:
    Make sure you have Python installed. It is recommended to use a virtual environment.

    ```bash
    python -m venv myenv
    source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
    ```

2.  **Install dependencies**:
    Install all the required packages using the `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the simulator**:
    Navigate to the `Simulate data` directory and run the `simulator.py` script.

    ```bash
    cd "Simulate data"
    python simulator.py
    ```

    You can use command-line arguments to customize the simulation:
    - `--count` or `-c`: Number of posts to generate (default: 10).
    - `--delay` or `-d`: Delay in seconds between posts (default: 2.0).
    - `--output` or `-o`: File to save the generated posts in JSONL format.
    - `--batch` or `-b`: Generate all posts at once without delay.
    - `--media` or `-m`: Path to the media folder.

    **Example**:
    ```bash
    python simulator.py --count 20 --delay 0.5 --batch --output generated_posts.jsonl
    ```
    This command will generate 20 posts in a batch and save them to `generated_posts.jsonl`.
