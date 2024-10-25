# PRC Data Challenge - Actual TakeOff Weight (ATOW) Prediction

## Overview

The **Performance Review Commission (PRC) Data Challenge** is designed to engage data scientists, even without an aviation background, to create teams and compete in building an open Machine Learning (ML) model. The challenge is to accurately infer the **Actual TakeOff Weight (ATOW)** of flights across Europe in 2022.

We provide detailed flight information for **369,013** flights, including origin/destination airports, aircraft types, off-block and arrival times, and the estimated TakeOff Weight (ETOW). Thanks to collaboration with the **OpenSky Network (OSN)**, we also provide the corresponding flight trajectories, sampled at a maximum 1-second granularity, accounting for **158 GiB** of parquet files.

The challenge will be scored using two datasets:
- The **submission_set.csv**, containing **105,959 flights**, will be used for ranking intermediate submissions.
- An additional **52,190 flights** will be used for the final ranking and prize evaluation.

For more information, visit the [Data page on the challenge website](https://ansperformance.eu/study/data-challenge/).

## Table of Contents
- [Acronyms](#acronyms)
- [Flight List](#flight-list)
- [Trajectory Data](#trajectory-data)
- [Getting Started](#getting-started)
- [Model Submission](#model-submission)
- [License](#license)

## Acronyms

- **ADS-B**: Automatic Dependent Surveillance–Broadcast
- **ATOW**: Actual TakeOff Weight
- **ETOW**: Estimated TakeOff Weight
- **ML**: Machine Learning
- **MTOW**: Maximum TakeOff Weight
- **OSN**: OpenSky Network
- **PRC**: Performance Review Commission
- **TOW**: TakeOff Weight

## Flight List

The dataset contains **369,013 flights** that departed or arrived in Europe in 2022. It includes the following details:

- **Flight Identification**: Unique ID (`flight_id`), obfuscated callsign (`callsign`)
- **Origin/Destination**:
  - Aerodrome of Departure (`adep`) [ICAO code]
  - Aerodrome of Destination (`ades`) [ICAO code]
  - Airport name (`name_adep`, `name_ades`)
  - Country codes (`country_code_adep`, `country_code_ades`) [ISO2C]
- **Timing**:
  - Date of flight (`date`) [ISO 8601 UTC]
  - Actual Off-Block Time (`actual_offblock_time`) [ISO 8601 UTC]
  - Arrival Time (`arrival_time`) [ISO 8601 UTC]
- **Aircraft**:
  - Aircraft type code (`aircraft_type`) [ICAO aircraft type]
  - Wake Turbulence Category (`wtc`)
- **Airline**:
  - Obfuscated Aircraft Operator (AO) code (`airline`)
- **Operational Values**:
  - Flight duration (`flight_duration`) [min]
  - Taxi-out time (`taxiout_time`) [min]
  - Route length (`flown_distance`) [nmi]
  - Estimated TakeOff Weight (`tow`) [kg]

## Trajectory Data

Flight trajectories, provided as daily `.parquet` files, amount to approximately **158 GiB** and include a **1-second granularity** ADS-B position report for each flight. These trajectories cover most flights, though some might be incomplete due to limited ADS-B coverage.

Each trajectory file contains:
- **Flight Identification**: Unique ID (`flight_id`), ICAO 24-bit address (`icao24`)
- **4D Position**: Longitude, latitude, altitude, and timestamp
- **Speed**: Ground speed (`groundspeed`), track angle (`track`, `track_unwrapped`), vertical rate of climb/descent (`vertical_rate`)
- **Meteorological Info (optional)**:
  - Wind (`u_component_of_wind`, `v_component_of_wind`) [m/s]
  - Temperature [Kelvin]

Files are named in the format `<yyyy-mm-dd>.parquet` and contain all position reports for that date in UTC..

## Getting Started

This project uses **Poetry** to manage dependencies and virtual environments. Poetry ensures that your project environment is consistent across different machines and provides an easy way to manage dependencies and package your application.

### Prerequisites

1. **Install Poetry**: If you don’t have Poetry installed, you can install it by following the instructions [here](https://python-poetry.org/docs/#installation).
    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```

2. **Verify Poetry Installation**: Run the following command to ensure Poetry is correctly installed:
    ```bash
    poetry --version
    ```

### Installation and Setup

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/prc-data-challenge.git
    cd prc-data-challenge
    ```

2. **Install Dependencies**: Poetry will automatically create a virtual environment and install the required dependencies specified in the `pyproject.toml` file.
    ```bash
    poetry install
    ```

3. **Activate the Virtual Environment**:
   Poetry manages virtual environments automatically. You can activate it using the following command:
    ```bash
    poetry shell
    ```

### Adding New Dependencies

To add new dependencies, use:
```bash
poetry add <package-name>
```
This will automatically update your `pyproject.toml` and lock the package version in `poetry.lock`.

### Running the Project

Once your virtual environment is activated, you can run your project with any of your custom commands or scripts:
```bash
poetry shell
python <your_script.py>
```
or directly:
```bash
poetry python <your_script.py>
```

### Managing Dependencies

Poetry manages dependency versions and ensures your project remains consistent. To update dependencies:
```bash
poetry update
```
---


### Dataset Access
An access was granted to the participants of the challenge trought MinIO Client .
The dataset files are hosted on OSN infrastructure.
Upon registration of your team you should have received the relevant

* team name and ID
* BUCKET_ACCESS_KEY and BUCKET_ACCESS_SECRET.

### Additional datasets
Two additional datasets were used in this challenge:
- The Global Airport Database ([here](https://www.partow.net/miscellaneous/airportdatabase/))
- CADO airplane database (Link [here](https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/LLRJO0))

#### The Global Airport Database

**Description:** The Global Airport Database (GADB) is a FREE downloadable database of **9300 airports** big and small from all around the world. The database is presented in a simple token delimited format. The database provides detailed information about the airports listed including:
- ICAO code
- IATA code
- Name
- Country
- City
- Latitude-Longitude position
- Altitude

**License:** Mit License

#### CADO airplane database
**Description:** This database contains data of nearly **230 airplanes**. Each airplane is described by **31 parameters** such as: name, IATA code and category (general, commuter, regional, short-medium, long range), geometry, mass, max speed, typical cruise mach number, typical range, typical approach speed, take-off field length, landing field length, number of engine, type of engine, typical engine model, bypass ratio, max thrust or max power.

**Contribution:** Kambiri, Y.A. et al. (2024) ‘Energy consumption of Aircraft with new propulsion systems and storage media’, in. AIAA SCITECH 2024 Forum, American Institute of Aeronautics and Astronautics. Available at: https://doi.org/10.2514/6.2024-1707.

**License:** ODbL 1.0 license

## Model
The model used in this challenge is an [XG Boost](https://xgboost.readthedocs.io/en/stable/).

XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. XGBoost is ideal for ATOW prediction due to its ability to handle complex, non-linear relationships across these diverse features. It efficiently manages both categorical and continuous variables, which is beneficial for combining static factors like aircraft type with dynamic ones such as weather and flight parameters.

## Run the experiements 
We provide script that performs data preparation, feature engineering, and XGBoost-based regression modeling to predict the takeoff weight (TOW) of flights using Optuna for hyperparameter optimization. The final model's predictions are saved in a CSV file, and feature importance is visualized for analysis.
Configuration File

The script reads a configuration file located at `configs/credentials.json`, which must contain:

{
    "team_name": "your_team_name",
    "code": "your_code",
    "output_folder": "your_output_folder"
}

For feature engineering , we tried many different approaches in general we distinguishe between two :
* General feature extract:
  This is a simpler and more straightforward method for extracting features from trajectory data. In this approach, we analyze each signal in the dataset and extract basic statistics, including the mean, maximum, and standard deviation
  ### Usage
  Run the script using:
  ```bash
  python feature_extractor/general_feature_extractor.py
  ```

* Climb & takeoff segmentation:
  This method focuses on extracting statistics from the takeoff and climb phases. In the literature, many papers confirm that the Take-Off Weight (TOW) is strongly related to the vertical rate and speed of the aircraft during the early stages of flight. Therefore, we focused on segmenting this particular phase using a handcrafted method that considers various types of noise that may occur in the data, as well as occasional missing chunks in some trajectory data.
  ### Usage
  Run the script using:
  ```bash
  python feature_extractor/general_feature_extractor.py
  ```
  Note: There are some parameters in this script that were setup intuitively and they can be different depending on the dataset (vertical_rate_threshold : Threshold for vertical rate min_duration_threshold_minutes : Minimum takeoff duration in minutes). We suggested these value after an extensive analysis of the Trajectory data.

### Train description 
Main function for model training and tuning:
* Defines an Optuna objective function for optimizing model hyperparameters.
* Trains an XGBoost model using the best-found parameters.
* Evaluates the model and calculates RMSE.
* Generates feature importance plots.

Run the script using:
```bash
python module/xgboost_model.py
```
This methods can work very well on balanced dataset. But the challenge_set showed a unblaced represtation of each aircraft_type therefore we suggest a new method where instead of prediction the TOW diretly we try to predict the (mean(TOW@ChallengeSet)-TOW). This methods boosted considerable our performance in the final submission_set.

Before running the train script we need some inputs that are going to be given using this script : 
```bash
python module/pre_processing.py
```
Then run the final script using:
```bash
python module/xgboost_mean_diff.py
```
On the other hand due to the class imbalance in the dataset we tought about another approach that can boost the perforamnce of the model. This approach is based on creating multiple model for multiple aircraft_types.
Try this approach using using:
```bash
python module/xgboost_model_categories.py
```
Notes: 
* You can re-define the sub-categories that you want to use depending on the objectives.
* This methodes uses the xgboost_mean_diff's approach to compute TOW

## Model Submission

Submit your models for evaluation through the challenge submission platform. Models will be evaluated based on their ability to accurately predict the **Actual TakeOff Weight (ATOW)** for the flights in the provided dataset. Intermediate rankings will be done using **submission_set.csv**.

## License

This project is licensed under the **GNU General Public License v3.0**. You may obtain a copy of the license at [GPL-3.0 License](https://www.gnu.org/licenses/gpl-3.0.en.html).

---
