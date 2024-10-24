Here's an updated version of your `README.md` with the correct GPL-3.0 license:

---

# PRC Data Challenge - Actual TakeOff Weight (ATOW) Prediction

## Overview

The **Performance Review Commission (PRC) Data Challenge** is designed to engage data scientists, even without an aviation background, to create teams and compete in building an open Machine Learning (ML) model. The challenge is to accurately infer the **Actual TakeOff Weight (ATOW)** of flights across Europe in 2022.

We provide detailed flight information for **369,013** flights, including origin/destination airports, aircraft types, off-block and arrival times, and the estimated TakeOff Weight (ETOW). Thanks to collaboration with the **OpenSky Network (OSN)**, we also provide the corresponding flight trajectories, sampled at a maximum 1-second granularity, accounting for **158 GiB** of parquet files.

The challenge will be scored using two datasets:
- The **submission_set.csv**, containing **105,959 flights**, will be used for ranking intermediate submissions.
- An additional **52,190 flights** will be used for the final ranking and prize evaluation.

For more information, visit the Data page on the challenge website.

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

Files are named in the format `<yyyy-mm-dd>.parquet` and contain all position reports for that date in UTC.

### Important Notes:
- Some flights span multiple `.parquet` files due to crossing UTC midnight.
- Trajectories may not perfectly match the flight list times due to incomplete ADS-B coverage. The interval `[actual_offblock_time + taxiout_time, arrival_time]` approximates the flight's in-air portion.

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


### Data
Download the provided flight list and trajectory parquet files and store them in the `data/` directory. 

## Model Submission

Submit your models for evaluation through the challenge submission platform. Models will be evaluated based on their ability to accurately predict the **Actual TakeOff Weight (ATOW)** for the flights in the provided dataset. Intermediate rankings will be done using **submission_set.csv**.

## License

This project is licensed under the **GNU General Public License v3.0**. You may obtain a copy of the license at [GPL-3.0 License](https://www.gnu.org/licenses/gpl-3.0.en.html).

---

This version includes the correct license section referring to GPL-3.0. You can adjust the repository link and any additional details as needed for your specific project setup.