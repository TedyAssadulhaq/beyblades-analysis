# Beyblade Battle Analyzer

This project provides a Python-based solution for detecting, tracking, and analyzing Beyblade battles from video footage. It utilizes the YOLO (You Only Look Once) model for object detection and custom logic for tracking Beyblades, determining their spin status, identifying the winner, and logging battle statistics to a CSV file, along with generating an annotated output video.

## Prerequisites

Before you begin, ensure you have the following installed:

* Python 3.8+
* pip (Python package installer)
* Git (for cloning the repository)

## Setup Instructions

Follow these steps to get the project up and running on your local machine:

1.  **Clone the Repository:**
    Open your terminal or command prompt and run:
    ```bash
    git clone [https://github.com/TedyAssadulhaq/beyblades-analysis.git](https://github.com/TedyAssadulhaq/beyblades-analysis.git)
    cd beyblades-analysis
    ```

2.  **Create a Virtual Environment (Recommended):**
    It's good practice to use a virtual environment to manage project dependencies.
    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment:**
    * **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install Dependencies:**
    First, create a `requirements.txt` file in the root of your project with the following content:
    ```
    opencv-python
    ultralytics
    numpy
    pandas
    torch
    torchvision
    pillow
    ```
    Then, install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Prepare Input Directories:**
    The script expects specific directory structures for input files. Create these folders in the root of your project:
    ```bash
    mkdir model
    mkdir input
    mkdir output
    ```

6.  **Place Your YOLO Model File:**
    Download your pre-trained YOLO model file (e.g., `best.pt`) and place it inside the `model/` directory.
    * Example: `model/best.pt`

7.  **Place Your Input Video File:**
    Place the video file you want to analyze (e.g., `battle.mp4`) inside the `input/` directory.
    * Example: `input/battle.mp4`

## Running the Application

Once everything is set up, you can run the Beyblade Battle Analyzer:

1.  **Ensure your virtual environment is activated.**
2.  **Run the main script:**
    ```bash
    python your_script_name.py
    ```
    (Replace `your_script_name.py` with the actual name of your Python script, likely `main.py` or similar, if you saved the provided code in a file.)

    A window titled "Beyblade Detection" will appear, showing the live processing of the video.

## Output

After the script finishes processing the video (or if you press 'q' to quit), the following files will be generated in the `output/` directory:

* `beyblade_output.mp4`: The processed video with detections, tracking IDs, spin status, and battle information overlaid.
* `beyblade_battles.csv`: A CSV file containing a summary of the battle, including:
    * `Battle_Number`: Sequential battle ID.
    * `Battle_Duration`: Total time of the battle.
    * `Winner`: The color name of the winning Beyblade.
    * `Winner_Spin_Duration`: How long the winner spun after the other Beyblades stopped.
    * `Fastest_Spinner_First_Half`: Indicates which Beyblade had a faster average spin score in the first half of the battle.
