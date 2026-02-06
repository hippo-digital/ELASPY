#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the main interface of the simulator.

You can set all simulator parameters with this script and then run the
simulator. Below, all parameters are discussed. The input data itself is
discussed in the input data section of the website.

Parameters
----------
START_SEED_VALUE : int
    The initial seed value. The seed of the ith run is equal to
    ``START_SEED_VALUE  + (i-1)``.
DATA_DIRECTORY : str
    The folder, relative to the ``ROOT_DIRECTORY`` (automatically determined),
    where the input data is located.
SIMULATION_INPUT_DIRECTORY : str | None
    The folder, relative to the ``ROOT_DIRECTORY`` (automatically determined),
    where the simulation input data is located if ``LOAD_INPUT_DATA=True``.
    Otherwise it should be ``None``.
SIMULATION_OUTPUT_DIRECTORY : str
    The folder, relative to the ``ROOT_DIRECTORY`` (automatically determined),
    where the simulation output data should be saved.
TRAVEL_TIMES_FILE : str
    The name of the file that contains the data with the siren travel times
    between nodes.
DISTANCE_FILE : str
    The name of the file that contains the data with the distance between nodes.
NODES_FILE : str
    The name of the file that contains the data with the nodes of the region.
HOSPITAL_FILE : str
    The name of the file that contains the nodes where hospitals are located.
BASE_LOCATIONS_FILE : str
    The name of the file that contains the nodes where bases are located.
AMBULANCE_BASE_LOCATIONS_FILE : str:
    The name of the file that contains the assignment of ambulances to bases.
SCENARIO : str
    The scenario. The following are valid: RB1, RB2, FB1, RB1_FB1, RB1_RH1,
    RB1_FH1, FB1_RH1, FB1_FH1, RB50_RH50, Diesel.
CHARGING_SCENARIO_FILE : str
    The name of the file that contains the charging scenario data.
SIMULATION_PATIENT_OUTPUT_FILE_NAME : str
    The name of the file where the patient dataframe will be saved.
SIMULATION_AMBULANCE_OUTPUT_FILE_NAME : str
    The name of the file where the ambulance dataframe will be saved.
RUN_PARAMETERS_FILE_NAME : str
    The name of the file where the run parameters will be saved.
RUNNING_TIME_FILE_NAME : str
    The name of the file where the running times will be saved.
SIMULATION_PRINTS_FILE_NAME : str
    The name of the file where the simulation prints will be saved.
MEAN_RESPONSE_TIMES_FILE_NAME : str
    The name of the file where the mean response time of each run will be saved.
EMP_QUANTILE_RESPONSE_TIMES_FILE_NAME : str
    The name of the file where the 95% empirical quantile of the response time
    of each run will be saved.
BUSY_FRACTIONS_FILE_NAME : str
    The name of the file where the empirical busy fraction of each run will be
    saved.
INTERARRIVAL_TIMES_FILE : str
    The name of the file with the interarrival times of the patients if
    ``LOAD_INPUT_DATA=True``. Otherwise it should be ``None``.
ON_SITE_AID_TIMES_FILE : str
    The name of the file with the on-site aid times of the patients if
    ``LOAD_INPUT_DATA=True``. Otherwise it should be ``None``.
DROP_OFF_TIMES_FILE : str
    The name of the file with the handover times at the hospital of the
    patients if ``LOAD_INPUT_DATA=True``. Otherwise it should be ``None``.
LOCATION_IDS_FILE : str
    The name of the file with the patient arrival locations if
    ``LOAD_INPUT_DATA=True``. Otherwise it should be ``None``.
TO_HOSPITAL_FILE : str
    The name of the file that specifies for each patient whether transportation
    to the hospital is required or not if ``LOAD_INPUT_DATA=True``. Otherwise
    it should be ``None``.
NUM_RUNS : int
    The number of simulation runs.
PROCESS_TYPE : str
    The type of arrival process. Use "Time" to simulate an arrival process
    where patients arrive within ``PROCESS_TIME`` time. Use "Number" to
    simulate an arrival process where ``PROCESS_NUM_CALLS`` patients arrive.
PROCESS_NUM_CALLS : int | None
    The number of patients that arrive if ``PROCESS_TYPE="Number"``. Should be
    ``None`` if ``PROCESS_TYPE="Time"``.
PROCESS_TIME : float | None
    The time in minutes during which patients can arrive if
    ``PROCESS_TYPE="Time"``. Should be ``None`` if ``PROCESS_TYPE="Number"``.
NUM_AMBULANCES : int
    The number of ambulances.
PROB_GO_TO_HOSPITAL : float | None
    The probability that a patient has to be transported to a hospital if
    ``LOAD_INPUT_DATA=False``. Otherwise it should be ``None``.
CALL_LAMBDA : float | None
    The arrival rate parameter of the arrival Poisson process if
    ``LOAD_INPUT_DATA=False``. Otherwise it should be ``None``.
AID_PARAMETERS : list[float | int]
    The parameters of the lognormal distribution for providing treatment on
    site. The first parameter is the sigma parameter, the second the
    location parameter , the third the scale parameter and the last the
    cut-off/maximum value.
DROP_OFF_PARAMETERS : list[float | int]
    The parameters of the lognormal distribution for the handover time at the
    hospital. The first parameter is the sigma parameter, the second the
    location parameter , the third the scale parameter and the last the
    cut-off/maximum value.
ENGINE_TYPE : str
    The engine type. Either "electric" or "diesel".
IDLE_USAGE :  float | None
    The energy consumption when idle/stationary in kW. If
    ``ENGINE_TYPE="diesel"`` it should be ``None``.
DRIVING_USAGE :  float | None
    The energy consumption when driving in kWh/km. If ``ENGINE_TYPE="diesel"``
    it should be ``None``.
BATTERY_CAPACITY : float
    The battery capacity of an electric ambulance. If ``ENGINE_TYPE="diesel"``
    it is equal to infinity (i.e., ``numpy.inf``)).
NO_SIREN_PENALTY : float
    The penalty for driving without sirens. The driving times with siren are
    scaled according to this value. Should be between 0 and 1.
LOAD_INPUT_DATA : bool
    Whether the input data should be read from data (``True``) or generated
    before the simulation starts (``False``).
CRN_GENERATOR : str | None
    The pseudo-random number generator that should be used if
    `LOAD_INPUT_DATA=False``. Either "Generator" for using NumPy's default or
    "RandomState" for Numpy's legacy generator. It should be ``None`` if
    ``LOAD_INPUT_DATA=True``.
INTERVAL_CHECK_WP : float | None
    The interval (in minutes) at which the simulator checks for waiting
    patients. If ``ENGINE_TYPE="diesel"`` it should be ``None``.
TIME_AFTER_LAST_ARRIVAL : float | None
    The time after the last arriving patient the simulator needs to check for
    waiting patients. If ``ENGINE_TYPE="diesel"`` it should be ``None``.
AT_BOUNDARY : float
    The warm-up period (in minutes) for the busy fraction calculation.
FT_BOUNDARY : float
    The cool-down period (in minutes) for the busy fraction calculation.
PRINT : bool
    If ``True``, debug prints are provided that clarify the simulation process.
PRINT_STATISTICS : bool
    If ``True``, useful simulation statistics such as the mean response time
    are provided for each run.
PLOT_FIGURES : bool
    If ``True``, multiple plots are provided for each simulation run.
SAVE_PRINTS_TXT : bool
    If ``True``, the prints are saved to ``SIMULATION_PRINTS_FILE_NAME``.
SAVE_OUTPUT : bool
    If ``True``, saves the run parameters in ``RUN_PARAMETERS_FILE_NAME``, the
    running times in ``RUNNING_TIME_FILE_NAME``, the mean response time per run
    in ``MEAN_RESPONSE_TIMES_FILE_NAME``, the 95% empirical quantile of the
    response time per run in ``EMP_QUANTILE_RESPONSE_TIMES_FILE_NAME`` and the
    empirical busy fraction in ``BUSY_FRACTIONS_FILE_NAME``.
SAVE_PLOTS : bool
    If ``True``, saves the plots. Can only be ``True`` if ``PLOT_FIGURES=True``.
SAVE_DFS : bool
    If ``True``, saves the ambulance dataframe of each run in
    ``SIMULATION_AMBULANCE_OUTPUT_FILE_NAME`` (adding a run_i suffix) and the
    patient dataframe of each run in ``SIMULATION_PATIENT_OUTPUT_FILE_NAME``
    (adding a run_i suffix).
DATA_COLUMNS_PATIENT : list[str]
    The columns for the patient DataFrame.
DATA_COLUMNS_AMBULANCE : list[str]
    The columns for the ambulance DataFrame.
"""

import copy
import datetime
import os
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import scipy
import yaml
from ambulance_simulation import run_simulation
from input_output_functions import (
    calculate_busy_fraction,
    calculate_response_time_ecdf,
    check_input_parameters,
    print_parameters,
    save_input_parameters,
    save_simulation_output,
    simulation_statistics,
)
from plot_functions import (
    hist_battery_increase_decrease,
    plot_battery_levels,
    plot_response_times,
)
from tqdm import tqdm

###################################Seed########################################

################################Directories####################################
ROOT_DIRECTORY: str = os.path.dirname(os.path.dirname(__file__))
SIMULATION_INPUT_DIRECTORY: str | None = None
SIMULATION_OUTPUT_DIRECTORY: str = os.path.join(ROOT_DIRECTORY, "results/")
#################################File names####################################
TRAVEL_TIMES_FILE: str = "TRAVEL_TIMES_FILE.csv"
DISTANCE_FILE: str = "DISTANCE_FILE.csv"
NODES_FILE: str = "NODES_FILE.csv"
HOSPITAL_FILE: str = "HOSPITAL_FILE.csv"
BASE_LOCATIONS_FILE: str = "BASE_LOCATIONS_FILE.csv"
AMBULANCE_BASE_LOCATIONS_FILE: str = "AMBULANCE_BASE_LOCATIONS_FILE.csv"
SCENARIO: str = "Diesel"
CHARGING_SCENARIO_FILE: str = f"../../../charging_scenario_21_22_{SCENARIO}.csv"
SIMULATION_PATIENT_OUTPUT_FILE_NAME: str = f"Patient_df_{SCENARIO}"
SIMULATION_AMBULANCE_OUTPUT_FILE_NAME: str = f"Ambulance_df_{SCENARIO}"

RUN_PARAMETERS_FILE_NAME: str = f"run_parameters_{SCENARIO}"
RUNNING_TIME_FILE_NAME: str = f"running_times_{SCENARIO}"
SIMULATION_PRINTS_FILE_NAME: str = f"run_prints_{SCENARIO}"
MEAN_RESPONSE_TIMES_FILE_NAME: str = f"mean_response_times_all_runs_{SCENARIO}"
EMP_QUANTILE_RESPONSE_TIMES_FILE_NAME: str = (
    f"emp_quantile_response_times_all_runs_{SCENARIO}"
)
BUSY_FRACTIONS_FILE_NAME: str = f"busy_fractions_all_runs_{SCENARIO}"

INTERARRIVAL_TIMES_FILE: str | None = None
ON_SITE_AID_TIMES_FILE: str | None = None
DROP_OFF_TIMES_FILE: str | None = None
LOCATION_IDS_FILE: str | None = None
TO_HOSPITAL_FILE: str | None = None

############################Simulation parameters##############################
ENGINE_TYPE: str = "diesel"
IDLE_USAGE: float | None = None  # kW
DRIVING_USAGE: float | None = None  # kWh/km
BATTERY_CAPACITY: float
if ENGINE_TYPE == "electric":
    BATTERY_CAPACITY = 150.0
else:
    BATTERY_CAPACITY = np.inf
NO_SIREN_PENALTY: float = 0.95
LOAD_INPUT_DATA: bool = False
CRN_GENERATOR: str | None = "Generator"

# Check to see if we can serve queued patients every 2 minutes
# This used to be Electric specific
INTERVAL_CHECK_WP: float | None = 2
TIME_AFTER_LAST_ARRIVAL: float | None = 2

AT_BOUNDARY: float = 60.0
FT_BOUNDARY: float = 720.0
##############################Output Parameters################################
PRINT: bool = False
PRINT_STATISTICS: bool = True
PLOT_FIGURES: bool = False

SAVE_PRINTS_TXT: bool = False
SAVE_OUTPUT: bool = True
SAVE_PLOTS: bool = False
SAVE_DFS: bool = True

DATA_COLUMNS_PATIENT: list["str"] = [
    "patient_ID",
    "response_time",
    "arrival_time",
    "location_ID",
    "nr_ambulances_available",
    "nr_ambulances_not_assignable",
    "assigned_to_ambulance_nr",
    "waiting_time_before_assigned",
    "driving_time_to_patient",
    "ambulance_arrival_time",
    "on_site_aid_time",
    "to_hospital",
    "hospital_ID",
    "driving_time_to_hospital",
    "drop_off_time_hospital",
    "finish_time",
    "category",
]
DATA_COLUMNS_AMBULANCE: list["str"] = [
    "ambulance_ID",
    "time",
    "battery_level_before",
    "battery_level_after",
    "use_or_charge",
    "idle_or_driving_decrease",
    "idle_time",
    "source_location_ID",
    "target_location_ID",
    "driven_km",
    "battery_decrease",
    "charging_type",
    "charging_location_ID",
    "speed_charger",
    "charging_success",
    "waiting_time",
    "charging_interrupted",
    "charging_time",
    "battery_increase",
]
###################################Initialization##############################
BASE_SIMULATION_PARAMETERS: dict[str, Any] = {
    "ENGINE_TYPE": ENGINE_TYPE,
    "IDLE_USAGE": IDLE_USAGE,
    "DRIVING_USAGE": DRIVING_USAGE,
    "BATTERY_CAPACITY": BATTERY_CAPACITY,
    "NO_SIREN_PENALTY": NO_SIREN_PENALTY,
    "PRINT": PRINT,
    "DATA_COLUMNS_PATIENT": DATA_COLUMNS_PATIENT,
    "DATA_COLUMNS_AMBULANCE": DATA_COLUMNS_AMBULANCE,
    "TRAVEL_TIMES_FILE": TRAVEL_TIMES_FILE,
    "DISTANCE_FILE": DISTANCE_FILE,
    "NODES_FILE": NODES_FILE,
    "HOSPITAL_FILE": HOSPITAL_FILE,
    "BASE_LOCATIONS_FILE": BASE_LOCATIONS_FILE,
    "AMBULANCE_BASE_LOCATIONS_FILE": AMBULANCE_BASE_LOCATIONS_FILE,
    "SCENARIO": SCENARIO,
    "CHARGING_SCENARIO_FILE": CHARGING_SCENARIO_FILE,
    "SIMULATION_PATIENT_OUTPUT_FILE_NAME": SIMULATION_PATIENT_OUTPUT_FILE_NAME,
    "SIMULATION_AMBULANCE_OUTPUT_FILE_NAME": SIMULATION_AMBULANCE_OUTPUT_FILE_NAME,
    "SIMULATION_OUTPUT_DIRECTORY": SIMULATION_OUTPUT_DIRECTORY,
    "SIMULATION_INPUT_DIRECTORY": SIMULATION_INPUT_DIRECTORY,
    "SAVE_OUTPUT": SAVE_OUTPUT,
    "LOAD_INPUT_DATA": LOAD_INPUT_DATA,
    "INTERARRIVAL_TIMES_FILE": INTERARRIVAL_TIMES_FILE,
    "ON_SITE_AID_TIMES_FILE": ON_SITE_AID_TIMES_FILE,
    "DROP_OFF_TIMES_FILE": DROP_OFF_TIMES_FILE,
    "LOCATION_IDS_FILE": LOCATION_IDS_FILE,
    "TO_HOSPITAL_FILE": TO_HOSPITAL_FILE,
    "CRN_GENERATOR": CRN_GENERATOR,
    "INTERVAL_CHECK_WP": INTERVAL_CHECK_WP,
    "TIME_AFTER_LAST_ARRIVAL": TIME_AFTER_LAST_ARRIVAL,
    "RUN_PARAMETERS_FILE_NAME": RUN_PARAMETERS_FILE_NAME,
    "RUNNING_TIME_FILE_NAME": RUNNING_TIME_FILE_NAME,
    "PLOT_FIGURES": PLOT_FIGURES,
    "SAVE_PLOTS": SAVE_PLOTS,
    "SAVE_DFS": SAVE_DFS,
    "PRINT_STATISTICS": PRINT_STATISTICS,
    "SIMULATION_PRINTS_FILE_NAME": SIMULATION_PRINTS_FILE_NAME,
    "SAVE_PRINTS_TXT": SAVE_PRINTS_TXT,
    "MEAN_RESPONSE_TIMES_FILE_NAME": MEAN_RESPONSE_TIMES_FILE_NAME,
    "EMP_QUANTILE_RESPONSE_TIMES_FILE_NAME": EMP_QUANTILE_RESPONSE_TIMES_FILE_NAME,
    "AT_BOUNDARY": AT_BOUNDARY,
    "FT_BOUNDARY": FT_BOUNDARY,
    "BUSY_FRACTIONS_FILE_NAME": BUSY_FRACTIONS_FILE_NAME,
}
SIMULATION_DATA: dict[str, Any] = {
    "DATA_COLUMNS_PATIENT": DATA_COLUMNS_PATIENT,
    "DATA_COLUMNS_AMBULANCE": DATA_COLUMNS_AMBULANCE,
}


####################################Run########################################
def run(config_file: Path):

    # read yaml
    with open(config_file, "rt") as f:
        yaml_config = yaml.safe_load(f)

    # Set Simulation Parameters from the yaml (with can also override the base if necessary)
    SIMULATION_PARAMETERS = BASE_SIMULATION_PARAMETERS | yaml_config

    # Set the number of ambulances based on the file so that parameter doesn't have to be
    # kept in sync manually.
    SIMULATION_PARAMETERS["NUM_AMBULANCES"] = pd.read_csv(
        f"{SIMULATION_PARAMETERS['DATA_DIRECTORY']}"
        f"{SIMULATION_PARAMETERS['AMBULANCE_BASE_LOCATIONS_FILE']}",
        index_col=0,
    ).shape[0]

    if SIMULATION_PARAMETERS["PROCESS_TYPE"] == "Time":
        SIMULATION_PARAMETERS["PROCESS_NUM_CALLS"] = None
    else:
        raise Exception("Parameter file loading does not support PROCESS_TYPE != Time")

    # Extract NUM_RUNS for use here
    NUM_RUNS = SIMULATION_PARAMETERS["NUM_RUNS"]

    start_time_script = datetime.datetime.now()

    copy_simulation_parameters = copy.deepcopy(SIMULATION_PARAMETERS)
    check_input_parameters(SIMULATION_PARAMETERS)

    if SIMULATION_PARAMETERS["SAVE_OUTPUT"]:
        save_input_parameters(SIMULATION_PARAMETERS)
    else:
        print_parameters(SIMULATION_PARAMETERS)

    if SIMULATION_PARAMETERS["SAVE_PRINTS_TXT"]:
        sys.stdout = open(
            f"{SIMULATION_PARAMETERS['SIMULATION_OUTPUT_DIRECTORY']}"
            f"{SIMULATION_PARAMETERS['SIMULATION_PRINTS_FILE_NAME']}.txt",
            "wt",
        )

    mean_response_times: np.ndarray = np.zeros((NUM_RUNS))
    emp_quantile_response_times: np.ndarray = np.zeros((NUM_RUNS))
    busy_fractions: np.ndarray = np.zeros(NUM_RUNS)
    running_times: np.ndarray = np.zeros(NUM_RUNS)

    for run_nr in tqdm(range(NUM_RUNS)):
        start_time_simulation_run = datetime.datetime.now()
        try:
            if not SIMULATION_PARAMETERS["LOAD_INPUT_DATA"]:
                SIMULATION_PARAMETERS["SEED_VALUE"] = (
                    SIMULATION_PARAMETERS["START_SEED_VALUE"] + run_nr
                )

            run_simulation(SIMULATION_PARAMETERS, SIMULATION_DATA)
        except Exception as e:
            if NUM_RUNS > 1:
                print(f"Run {run_nr} failed to serve all patients!")
            else:
                raise e

        end_time_simulation_run = datetime.datetime.now()
        running_times[run_nr] = (
            end_time_simulation_run - start_time_simulation_run
        ).total_seconds()

        # Create DataFrames of simulation output
        start_time_df = datetime.datetime.now()
        df_patient = pd.DataFrame(
            SIMULATION_DATA["output_patient"],
            columns=SIMULATION_PARAMETERS["DATA_COLUMNS_PATIENT"],
        )
        df_patient = calculate_response_time_ecdf(df_patient)
        df_ambulance = pd.DataFrame(
            SIMULATION_DATA["output_ambulance"],
            columns=SIMULATION_PARAMETERS["DATA_COLUMNS_AMBULANCE"],
        )
        print(
            "The running time for creating the dfs is: "
            f"{datetime.datetime.now() - start_time_df}."
        )

        # Plot simulation output
        start_time_plots_stats = datetime.datetime.now()
        if SIMULATION_PARAMETERS["PLOT_FIGURES"]:
            plot_response_times(df_patient, run_nr, SIMULATION_PARAMETERS)
            if SIMULATION_PARAMETERS["ENGINE_TYPE"] == "electric":
                plot_battery_levels(df_ambulance, run_nr, SIMULATION_PARAMETERS)
                hist_battery_increase_decrease(
                    df_ambulance, run_nr, SIMULATION_PARAMETERS
                )
        if SIMULATION_PARAMETERS["PRINT_STATISTICS"]:
            simulation_statistics(
                df_patient,
                df_ambulance,
                start_time_simulation_run,
                end_time_simulation_run,
                SIMULATION_DATA["nr_times_no_fast_no_regular_available"],
                SIMULATION_PARAMETERS,
            )
        print(
            "The running time for creating the plots and printing the "
            "simulation stats is: "
            f"{datetime.datetime.now() - start_time_plots_stats}."
        )

        # Save simulation output
        if SIMULATION_PARAMETERS["SAVE_DFS"]:
            start_time_saving = datetime.datetime.now()
            save_simulation_output(
                SIMULATION_PARAMETERS["SIMULATION_OUTPUT_DIRECTORY"],
                SIMULATION_PARAMETERS["SIMULATION_PATIENT_OUTPUT_FILE_NAME"],
                df_patient,
                run_nr,
            )
            save_simulation_output(
                SIMULATION_PARAMETERS["SIMULATION_OUTPUT_DIRECTORY"],
                SIMULATION_PARAMETERS["SIMULATION_AMBULANCE_OUTPUT_FILE_NAME"],
                df_ambulance,
                run_nr,
            )
            save_simulation_output(
                SIMULATION_PARAMETERS["SIMULATION_OUTPUT_DIRECTORY"],
                "meal_breaks",
                pd.DataFrame(
                    SIMULATION_DATA["output_breaks"],
                    columns=[
                        "Ambulance_ID",
                        "meal_break_drive_start",
                        "meal_break_drive_end",
                        "meal_break_end",
                    ],
                ),
                run_nr,
            )
            save_simulation_output(
                SIMULATION_PARAMETERS["SIMULATION_OUTPUT_DIRECTORY"],
                "state",
                pd.DataFrame.from_records(SIMULATION_DATA["output_state"]),
                run_nr,
            )
            print(
                "The running time for saving the data is: "
                f"{datetime.datetime.now() - start_time_saving}."
            )

        mean_response_times[run_nr] = np.mean(df_patient["response_time"])
        emp_quantile_response_times[run_nr] = np.min(
            df_patient.loc[df_patient["ecdf_rt"] >= 0.95]["response_time"]
        )
        busy_fractions[run_nr] = calculate_busy_fraction(
            df_patient, SIMULATION_PARAMETERS
        )

    m_mean_response_times = np.mean(mean_response_times)
    m_emp_quantile_response_times = np.mean(emp_quantile_response_times)
    m_busy_fractions = np.mean(busy_fractions)

    print("\nAll runs finished")
    print(f"The mean mean response time over all runs is: {m_mean_response_times}.")
    if NUM_RUNS > 1:
        CI_error_m_mean_response_times = scipy.stats.t.ppf(0.975, NUM_RUNS - 1) * (
            np.std(mean_response_times, ddof=1) / np.sqrt(NUM_RUNS)
        )
        print(
            "The 95% CI of the mean mean response time is:"
            f"({m_mean_response_times - CI_error_m_mean_response_times},"
            f"{m_mean_response_times + CI_error_m_mean_response_times})."
        )

    print(
        "The mean 95% empirical quantile of the response time over "
        f"all runs is: {m_emp_quantile_response_times}."
    )
    if NUM_RUNS > 1:
        CI_error_m_emp_quantile_response_times = scipy.stats.t.ppf(
            0.975, NUM_RUNS - 1
        ) * (np.std(emp_quantile_response_times, ddof=1) / np.sqrt(NUM_RUNS))
        print(
            "The 95% CI of the mean 95% empirical quantile of the response time is:"
            f"({m_emp_quantile_response_times - CI_error_m_emp_quantile_response_times},"
            f"{m_emp_quantile_response_times + CI_error_m_emp_quantile_response_times})."
        )

    print(f"The mean busy fraction over all runs is: {m_busy_fractions}.")
    if NUM_RUNS > 1:
        CI_error_m_busy_fractions = scipy.stats.t.ppf(0.975, NUM_RUNS - 1) * (
            np.std(busy_fractions, ddof=1) / np.sqrt(NUM_RUNS)
        )
        print(
            "The 95% CI of the mean busy fraction is:"
            f"({m_busy_fractions - CI_error_m_busy_fractions},"
            f"{m_busy_fractions + CI_error_m_busy_fractions})."
        )

    if SIMULATION_PARAMETERS["SAVE_OUTPUT"]:
        pd.DataFrame(mean_response_times).to_csv(
            f"{SIMULATION_PARAMETERS['SIMULATION_OUTPUT_DIRECTORY']}"
            f"{SIMULATION_PARAMETERS['MEAN_RESPONSE_TIMES_FILE_NAME']}.csv"
        )
        pd.DataFrame(emp_quantile_response_times).to_csv(
            f"{SIMULATION_PARAMETERS['SIMULATION_OUTPUT_DIRECTORY']}"
            f"{SIMULATION_PARAMETERS['EMP_QUANTILE_RESPONSE_TIMES_FILE_NAME']}.csv"
        )
        pd.DataFrame(busy_fractions).to_csv(
            f"{SIMULATION_PARAMETERS['SIMULATION_OUTPUT_DIRECTORY']}"
            f"{SIMULATION_PARAMETERS['BUSY_FRACTIONS_FILE_NAME']}.csv"
        )
        pd.DataFrame(
            {
                "run_nr": np.arange(NUM_RUNS),
                "Running_time (sec)": running_times,
            }
        ).to_csv(
            f"{SIMULATION_PARAMETERS['SIMULATION_OUTPUT_DIRECTORY']}"
            f"{SIMULATION_PARAMETERS['RUNNING_TIME_FILE_NAME']}.csv"
        )

    for key in copy_simulation_parameters.keys():
        if copy_simulation_parameters[key] != SIMULATION_PARAMETERS[key]:
            raise Exception(
                "The SIMULATION_PARAMETERS were altered during "
                "the simulation. This should not happen. Error."
            )

    print(
        "\nRunning the complete main.py script takes: "
        f"{datetime.datetime.now() - start_time_script}."
    )
    print("\007")


if __name__ == "__main__":
    import shutil
    import sys

    config_path = Path(sys.argv[-1])
    assert config_path.is_file, "config file not found"
    run(config_path)
    shutil.copy(config_path, "./results/")
