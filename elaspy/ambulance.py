#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Optional

import numpy as np
import pandas as pd
import simpy as sp
from coordinate_methods import (
    calculate_new_coordinate,
    select_closest_location_ID,
)
from patient import Patient


class Ambulance:
    """
    A class to represent an ambulance.

    Attributes
    ----------
    env : sp.core.Environment
        The SimPy environment.
    resource : sp.resources.resource.PreemptiveResource
        A preemptive resource used for events that occupy the ambulance.
    assigned_to_patient : bool
        Whether the ambulance is currently assigned to a patient or not.
    helps_patient : bool
        Whether the ambulance is currently helping a patient or not.
    drives_to_base : bool
        Whether the ambulance is currently driving to its base or not.
    charges : bool
        Whether the ambulance is currently charging or not.
    current_location_ID : int
        The current location ID of the ambulance.
    base_location_ID : int
        The base location ID of the ambulance.
    battery : float
        The current battery level of the ambulance (kWh).
    MAX_BATTERY_LEVEL : float
        The maximum battery level of the ambulance.
    ENGINE_TYPE : str
        The engine type of the ambulance. "diesel" or "electric".
    ambulance_ID : int
        The ambulance ID.
    charging_since : float
        The start time of the current charging session. Equal to ``np.nan`` if
        the ambulance is not charging.
    speed_charger : float
        The speed of the charger of the current charging session. Equal to
        ``np.nan`` if the ambulance is not charging.

    """

    def __init__(
        self,
        env: sp.core.Environment,
        base_location_ID: int,
        ENGINE_TYPE: str,
        ID: int,
        BATTERY_CAPACITY: float,
        extra_data: pd.Series,
        mb_policy_func,
    ) -> None:
        """
        Initializes an ambulance.

        Parameters
        ----------
        env : sp.core.Environment
            The SimPy environment.
        base_location_ID : int
            The base location of the ambulance.
        ENGINE_TYPE : str
            The engine type of the ambulance. "diesel" or "electric".
        ID : int
            The ambulance ID.
        BATTERY_CAPACITY : float
            The battery capacity of the ambulance.

        """

        self.env: sp.core.Environment = env
        self.resource: sp.resources.resource.PreemptiveResource = sp.PreemptiveResource(
            env, capacity=1
        )
        self.patient: Patient | None = None
        self.assigned_to_patient: bool = False
        self.helps_patient: bool = False
        self.drives_to_base: bool = False
        self.charges: bool = False
        self.current_location_ID: int = base_location_ID
        self.base_location_ID: int = base_location_ID
        self.battery: float = BATTERY_CAPACITY
        self.MAX_BATTERY_LEVEL: float = BATTERY_CAPACITY
        self.ENGINE_TYPE: str = ENGINE_TYPE
        self.ambulance_ID: int = ID
        self.charging_since: float = np.nan
        self.speed_charger: float = np.nan

        extra_data = pd.Series() if extra_data is None else extra_data

        self.had_meal_break = extra_data.get("had_meal_break", 0) == 1
        self.in_meal_break = False
        self.meal_break_start = extra_data.get("meal_break_start", None)
        self.meal_break_end = extra_data.get("meal_break_end", None)

        self.shift_start_time = extra_data.get("shift_start", None)
        self.shift_end_time = extra_data.get("shift_end", None)
        self.shift_duration = extra_data.get("shift_duration", None)

        self.mb_allowed_by_policy = mb_policy_func

    def check_patient_reachable(
        self,
        ambulance_location_ID: int,
        patient_location_ID: int,
        hospital_location_ID: int,
        charging_stations_hospitals: dict[
            str, list[sp.resources.resource.Resource | float]
        ],
        SIMULATION_PARAMETERS: dict[str, Any],
        SIMULATION_DATA: dict[str, Any],
    ) -> bool:
        """
        Checks whether the patient is reachable for the ambulance.

        The patient is always reachable for a diesel vehicle, but for an
        electric vehicle, the current battery level is taken into account.
        Whether the ambulance is currently driving or charging is also taken
        into account. See the paper by Dieleman and Jagtenberg for the
        procedure that is used to determine whether a patient is reachable.

        Parameters
        ----------
        ambulance_location_ID : int
            The current location ID of the ambulance.
        patient_location_ID : int
            The arrival location of the patient.
        hospital_location_ID : int
            The assigned hospital (in case the patient needs to be transported).
        charging_stations_hospitals : dict[str, list[sp.resources.resource.Resource | float]
            The charging stations resources at all hospitals together with
            their charging speeds.
        SIMULATION_PARAMETERS : dict[str, Any]
            The simulation parameters. The parameter ``PRINT`` is at least
            necessary. Note that methods that are called within this method may
            require more parameters. See ``main.py`` for parameter explanations.
        SIMULATION_DATA : dict[str, Any]
            The simulation data. Methods that are called within this method
            require data. See these methods for explanations.

        Raises
        ------
        Exception
            1. If the ambulance is driving and charging at the same time.
            2. If an invalid ``ENGINE_TYPE`` is specified.

        Returns
        -------
        bool
            Whether the patient is reachable or not.

        """
        # XXX: Potential improvemnet: add a max distance/time to allow assignement over

        if self.ENGINE_TYPE == "diesel":
            # Disabled this print as it was flooding the log
            # if SIMULATION_PARAMETERS["PRINT"]:
            #     print("The ENGINE_TYPE is diesel, so the patient is reachable.")
            return True
        elif self.ENGINE_TYPE == "electric":
            (
                required_battery_to_patient,
                _,
            ) = Ambulance.calculate_battery_reduction_and_distance_driving(
                ambulance_location_ID,
                patient_location_ID,
                siren_off=False,
                SIMULATION_PARAMETERS=SIMULATION_PARAMETERS,
                SIMULATION_DATA=SIMULATION_DATA,
            )

            required_battery_idling = Ambulance.calculate_battery_reduction_idling(
                SIMULATION_PARAMETERS["AID_PARAMETERS"][-1],
                SIMULATION_PARAMETERS,
            )

            (
                required_battery_to_hospital,
                _,
            ) = Ambulance.calculate_battery_reduction_and_distance_driving(
                patient_location_ID,
                hospital_location_ID,
                siren_off=False,
                SIMULATION_PARAMETERS=SIMULATION_PARAMETERS,
                SIMULATION_DATA=SIMULATION_DATA,
            )

            (
                required_battery_to_base,
                _,
            ) = Ambulance.calculate_battery_reduction_and_distance_driving(
                patient_location_ID,
                self.base_location_ID,
                siren_off=True,
                SIMULATION_PARAMETERS=SIMULATION_PARAMETERS,
                SIMULATION_DATA=SIMULATION_DATA,
            )

            (
                required_battery_hospital_to_base,
                _,
            ) = Ambulance.calculate_battery_reduction_and_distance_driving(
                hospital_location_ID,
                self.base_location_ID,
                siren_off=True,
                SIMULATION_PARAMETERS=SIMULATION_PARAMETERS,
                SIMULATION_DATA=SIMULATION_DATA,
            )

            route_APH = (
                required_battery_to_patient
                + required_battery_idling
                + required_battery_to_hospital
            )
            route_APB = (
                required_battery_to_patient
                + required_battery_idling
                + required_battery_to_base
            )

            if str(hospital_location_ID) not in charging_stations_hospitals.keys():
                route_APHB = (
                    required_battery_to_patient
                    + required_battery_idling
                    + required_battery_to_hospital
                    + required_battery_hospital_to_base
                )
                if SIMULATION_PARAMETERS["PRINT"]:
                    print(
                        f"Hospital {hospital_location_ID} is "
                        "not part of the charging_stations_hospitals keys."
                    )
                    print(
                        f"For ambulance {self.ambulance_ID}, the trip "
                        f"A->P->H->B requires {required_battery_to_patient},"
                        f"{required_battery_idling},"
                        f"{required_battery_to_hospital},"
                        f"{required_battery_hospital_to_base}, "
                        f"(Total+safety: {route_APHB}) of battery."
                    )

            if SIMULATION_PARAMETERS["PRINT"]:
                print(
                    f"For ambulance {self.ambulance_ID}, the trip A->P->H "
                    f"requires {required_battery_to_patient},"
                    f"{required_battery_idling},"
                    f"{required_battery_to_hospital}, "
                    f"(Total+safety: {route_APH}) of battery."
                )
                print(
                    f"For ambulance {self.ambulance_ID}, the trip A->P->B "
                    f"requires {required_battery_to_patient},"
                    f"{required_battery_idling},"
                    f"{required_battery_to_base}, "
                    f"(Total+safety: {route_APB}) of battery."
                )

            current_battery = self.battery

            if self.drives_to_base:
                (
                    reduction_battery_to_base,
                    _,
                ) = Ambulance.calculate_battery_reduction_and_distance_driving(
                    self.current_location_ID,
                    ambulance_location_ID,
                    siren_off=True,
                    SIMULATION_PARAMETERS=SIMULATION_PARAMETERS,
                    SIMULATION_DATA=SIMULATION_DATA,
                )
                if SIMULATION_PARAMETERS["PRINT"]:
                    print(
                        "The battery decrease since driving to the base is "
                        f"equal to: {reduction_battery_to_base} kWh."
                    )
                current_battery = current_battery - reduction_battery_to_base
            if self.charges:
                battery_increase_since_charging = (
                    self.calculate_battery_increase_until_now()
                )
                if SIMULATION_PARAMETERS["PRINT"]:
                    print(
                        "The battery increase since charging is equal "
                        f"to: {battery_increase_since_charging} kWh."
                    )
                current_battery = current_battery + battery_increase_since_charging

            if self.drives_to_base and self.charges:
                raise Exception("Driving to base and charges is true. Error.")

            if (
                str(hospital_location_ID) not in charging_stations_hospitals.keys()
            ) and (current_battery >= route_APB and current_battery >= route_APHB):
                if SIMULATION_PARAMETERS["PRINT"]:
                    print(
                        "The hospital has no charger. The current battery "
                        f"level of the ambulance is {current_battery}. "
                        "The routes A->P->B and A->P->H->B are reachable."
                    )
                return True
            elif (str(hospital_location_ID) in charging_stations_hospitals.keys()) and (
                current_battery >= route_APH and current_battery >= route_APB
            ):
                if SIMULATION_PARAMETERS["PRINT"]:
                    print(
                        "The hospital has at least one charger. The current "
                        f"battery level of the ambulance is {current_battery}."
                        " The routes A->P->H and A->P->B are reachable."
                    )
                return True
            else:
                if SIMULATION_PARAMETERS["PRINT"]:
                    print(
                        "The current battery level of the ambulance is "
                        f"{current_battery}. The routes A->P->H and A->P->B "
                        "(if a charger is available) or A->P->H->B and A->P->B"
                        " are not reachable."
                    )
                return False
        else:
            raise Exception("Wrong ENGINE_TYPE specified.")

    def set_assigned_to_patient(self) -> None:
        """
        Sets the ``assigned_to_patient`` variable to ``True``.

        """
        self.assigned_to_patient = True

    def is_available_for_assignment(self) -> bool:
        if self.env.now < self.shift_start_time:
            return False

        if self.env.now > self.shift_end_time:
            return False

        if self.in_meal_break:
            return False

        if self.assigned_to_patient:
            return False

        return True

    def process_patient(
        self,
        patient: Patient,
        simulation_times: dict[str, np.ndarray],
        charging_stations_hospitals: dict[
            str, list[sp.resources.resource.Resource | float]
        ],
        SIMULATION_PARAMETERS: dict[str, Any],
        SIMULATION_DATA: dict[str, Any],
    ):
        """
        Processes a patient.

        The ambulance first drives to the patient, then performs on-site
        treatment. If the patient needs to be transported to the hospital,
        the ambulance drives to the hospital and then drops the patient off.

        Parameters
        ----------
        patient_ID : int
            The patient ID.
        patient_location_ID : int
            The arrival location of the patient.
        hospital_location_ID : int
            The assigned hospital (in case the patient needs to be
            transported).
        simulation_times : dict[str, np.ndarray]
            Contains the interarrival times, the on-site aid times and the
            drop-off times.
        charging_stations_hospitals : dict[str, list[sp.resources.resource.Resource | float]]
            The charging stations resources at all hospitals together with
            their charging speeds.
        to_hospital_bool : np.ndarray
            Specifies for each patient whether transportation to the hospital
            is required or not.
        SIMULATION_PARAMETERS : dict[str, Any]
            The simulation parameters. The parameter ``PRINT`` is at least
            necessary. Note that methods that are called within this method
            may require more parameters. See ``main.py`` for parameter
            explanations.
        SIMULATION_DATA : dict[str, Any]
            The simulation data. ``output_patient`` is at least necessary.
            See ``main.py`` and the input data section on the ELASPY website
            for explanations. Note that methods that are called within this
            method may require more data.

        Raises
        ------
        Exception
            1. If an ambulance is already helping a patient.
            2. If more than one patient was assigned to the ambulance.
            3. If the ambulance shift is not yet started.

        """

        if self.env.now < self.shift_start_time:
            raise Exception(
                "Ambulance cannot be assigned to a patient before the start of its shift"
            )

        if self.env.now > self.shift_end_time:
            raise Exception(
                "Ambulance cannot be assigned to a patient after the end of its shift"
            )

        if SIMULATION_PARAMETERS["PRINT"]:
            print(
                f"{self.env.now}: Patient {patient.patient_ID} gets "
                "in the process patient method "
                f"with ambulance {self.ambulance_ID}."
            )
        if self.helps_patient:
            raise Exception(
                f"Ambulance {self.ambulance_ID} is already helping patient {self.patient.patient_ID},"
                f"so it cannot help another patient ({patient.patient_ID}). Error."
            )

        # Patient uses ambulance resource; always possible because of earlier check.
        # So that ambulance is used according to Simpy.
        with self.resource.request(priority=1) as req:
            if len(self.resource.queue) > 0:
                raise Exception(
                    f"More than one patients were assigned to ambulance {self.ambulance_ID} at the same time. Error."
                )
            yield req

            self.patient = patient
            patient_ID = patient.patient_ID

            waiting_time_assigned = (
                self.env.now - SIMULATION_DATA["output_patient"][patient_ID, 2]
            )  # minus arrival time patient.

            SIMULATION_DATA["output_patient"][patient_ID, 6] = self.ambulance_ID
            SIMULATION_DATA["output_patient"][patient_ID, 7] = waiting_time_assigned

            if SIMULATION_PARAMETERS["PRINT"]:
                print(
                    f"{self.env.now}: Patient {patient_ID} is assigned "
                    f"to Ambulance {self.ambulance_ID}."
                )
                print(
                    f"Patient {patient_ID} had a waiting time of "
                    f"{waiting_time_assigned} before an ambulance was assigned."
                )
                print(
                    f"Ambulance {self.ambulance_ID} is at "
                    f"location {self.current_location_ID}."
                )
            self.helps_patient = True

            start_driving_time = self.env.now
            try:
                self.drive_process = self.env.process(
                    self.go_to_patient(
                        SIMULATION_PARAMETERS,
                        SIMULATION_DATA,
                    )
                )
                yield self.drive_process

                response_time = (
                    self.env.now - SIMULATION_DATA["output_patient"][patient_ID, 2]
                )  # minus arrival time patient.
                SIMULATION_DATA["output_patient"][patient_ID, 9] = self.env.now
                yield self.env.process(
                    self.on_site_aid_patient(
                        simulation_times["on_site"],
                        SIMULATION_PARAMETERS,
                        SIMULATION_DATA,
                    )
                )

                if self.patient.to_hospital:
                    SIMULATION_DATA["output_patient"][patient_ID, 11] = 1
                    SIMULATION_DATA["output_patient"][patient_ID, 12] = (
                        self.patient.hospital_location_ID
                    )
                    if SIMULATION_PARAMETERS["PRINT"]:
                        print(f"Patient {patient_ID} has to be brought to hospital.")

                    yield self.env.process(
                        self.go_to_hospital(
                            SIMULATION_PARAMETERS,
                            SIMULATION_DATA,
                        )
                    )
                    yield self.env.process(
                        self.drop_off_time(
                            simulation_times["drop_off"],
                            charging_stations_hospitals,
                            SIMULATION_PARAMETERS,
                            SIMULATION_DATA,
                        )
                    )
                else:
                    if SIMULATION_PARAMETERS["PRINT"]:
                        print(
                            f"Patient {patient_ID} does not have to be brought to hospital."
                        )
                    SIMULATION_DATA["output_patient"][patient_ID, 11] = 0

                if SIMULATION_PARAMETERS["PRINT"]:
                    print(
                        f"{self.env.now}: Ambulance {self.ambulance_ID} "
                        f"has finished treating patient {patient_ID}."
                    )

                SIMULATION_DATA["output_patient"][patient_ID, 15] = self.env.now

                SIMULATION_DATA["output_patient"][patient_ID, 1] = response_time
                if SIMULATION_PARAMETERS["PRINT"]:
                    print(
                        f"Patient {patient_ID} had a total response time of {response_time}."
                    )

            except sp.Interrupt as interrupt:
                if SIMULATION_PARAMETERS["PRINT"]:
                    print("Preemption. While driving to a patient.")

                driven_time = self.env.now - start_driving_time  # type: ignore

                if SIMULATION_PARAMETERS["PRINT"]:
                    print(f"Driving since {start_driving_time}.")  # type: ignore
                    print(
                        f"Ambulance {self.ambulance_ID} got preempted by "
                        f"{interrupt.cause} at {self.env.now}"  # type: ignore
                        f" after {driven_time}."
                    )

                (new_x, new_y) = calculate_new_coordinate(
                    driven_time,
                    self.current_location_ID,
                    self.patient.patient_location_ID,
                    siren_off=False,
                    SIMULATION_PARAMETERS=SIMULATION_PARAMETERS,
                    SIMULATION_DATA=SIMULATION_DATA,
                )
                new_location_ID = select_closest_location_ID(
                    (new_x, new_y), SIMULATION_PARAMETERS, SIMULATION_DATA
                )

                if SIMULATION_PARAMETERS["ENGINE_TYPE"] == "electric":
                    (
                        battery_reduction,
                        distance_km,
                    ) = Ambulance.calculate_battery_reduction_and_distance_driving(
                        self.current_location_ID,
                        new_location_ID,
                        siren_off=False,
                        SIMULATION_PARAMETERS=SIMULATION_PARAMETERS,
                        SIMULATION_DATA=SIMULATION_DATA,
                    )
                    self.add_ambulance_data_battery_decrease(
                        battery_reduction,
                        idle=False,
                        idle_time=None,
                        driven_km=distance_km,
                        source_location_ID=self.current_location_ID,
                        target_location_ID=new_location_ID,
                        SIMULATION_PARAMETERS=SIMULATION_PARAMETERS,
                        SIMULATION_DATA=SIMULATION_DATA,
                    )
                    self.decrease_battery(battery_reduction)
                    if SIMULATION_PARAMETERS["PRINT"]:
                        print(
                            f"The battery of ambulance {self.ambulance_ID} "
                            f"is reduced by: {battery_reduction}. "
                            "The current battery level of ambulance "
                            f"{self.ambulance_ID} is {self.battery}."
                        )
                elif SIMULATION_PARAMETERS["ENGINE_TYPE"] == "diesel":
                    self.add_ambulance_data_diesel(
                        idle=False,
                        idle_time=None,
                        source_location_ID=self.current_location_ID,
                        target_location_ID=new_location_ID,
                        SIMULATION_PARAMETERS=SIMULATION_PARAMETERS,
                        SIMULATION_DATA=SIMULATION_DATA,
                    )
                else:
                    raise Exception(
                        "Wrong 'ENGINE_TYPE' specified. "
                        "It should be either 'diesel' or 'electric'."
                    )

                self.current_location_ID = new_location_ID

            finally:
                del self.drive_process

            # Release the resource and the state flags
            self.helps_patient = False
            self.assigned_to_patient = False
            self.patient = None

        assert len(self.resource.queue) < 1, "Queue not freed"

        # Now the ambulance is available again check if this ambulance should take a meal break.
        if SIMULATION_PARAMETERS["WITH_MEAL_BREAKS"]:
            yield self.env.process(
                self.check_meal_break(SIMULATION_DATA, SIMULATION_PARAMETERS)
            )

    def go_to_patient(
        self,
        SIMULATION_PARAMETERS: dict[str, Any],
        SIMULATION_DATA: dict[str, Any],
    ):
        """
        Drive to the patient.

        Parameters
        ----------
        patient_ID : int
            The patient ID.
        patient_location_ID : int
            The arrival location of the patient.
        SIMULATION_PARAMETERS : dict[str, Any]
            The simulation parameters. The parameters ``PRINT`` and
            ``ENGINE_TYPE`` are at least necessary. Note that methods that are
            called within this method may require more parameters. See
            ``main.py`` for parameter explanations.
        SIMULATION_DATA : dict[str, Any]
            The simulation data. ``output_patient`` is at least necessary. See
            ``main.py`` and the input data section on the ELASPY website for
            explanations. Note that methods that are called within this
            method may require more data.

        Raises
        ------
        Exception
            If an invalid ``ENGINE_TYPE`` is specified.

        """

        to_site_travel_time = SIMULATION_DATA["SIREN_DRIVING_MATRIX"].loc[
            self.current_location_ID, self.patient.patient_location_ID
        ]
        SIMULATION_DATA["output_patient"][self.patient.patient_ID, 8] = (
            to_site_travel_time
        )
        if SIMULATION_PARAMETERS["PRINT"]:
            print(
                f"{self.env.now}: Ambulance {self.ambulance_ID} drives "
                f"from location {self.current_location_ID} to patient "
                f"{self.patient.patient_ID} at {self.patient.patient_location_ID} in "
                f"{to_site_travel_time}."
            )

        try:
            self.scene_arrival_time = self.env.now + to_site_travel_time
            self.scene_drive_time_start = self.env.now
            yield self.env.timeout(to_site_travel_time)

            if SIMULATION_PARAMETERS["ENGINE_TYPE"] == "electric":
                (
                    battery_reduction,
                    distance_km,
                ) = Ambulance.calculate_battery_reduction_and_distance_driving(
                    self.current_location_ID,
                    self.patient.patient_location_ID,
                    siren_off=False,
                    SIMULATION_PARAMETERS=SIMULATION_PARAMETERS,
                    SIMULATION_DATA=SIMULATION_DATA,
                )
                self.add_ambulance_data_battery_decrease(
                    battery_reduction,
                    idle=False,
                    idle_time=None,
                    driven_km=distance_km,
                    source_location_ID=self.current_location_ID,
                    target_location_ID=self.patient.patient_location_ID,
                    SIMULATION_PARAMETERS=SIMULATION_PARAMETERS,
                    SIMULATION_DATA=SIMULATION_DATA,
                )
                self.decrease_battery(battery_reduction)
                if SIMULATION_PARAMETERS["PRINT"]:
                    print(
                        f"{self.env.now}: The battery of ambulance "
                        f"{self.ambulance_ID} is reduced by: {battery_reduction}. "
                        "The current battery level of ambulance "
                        f"{self.ambulance_ID} is {self.battery}."
                    )
            elif SIMULATION_PARAMETERS["ENGINE_TYPE"] == "diesel":
                self.add_ambulance_data_diesel(
                    idle=False,
                    idle_time=None,
                    source_location_ID=self.current_location_ID,
                    target_location_ID=self.patient.patient_location_ID,
                    SIMULATION_PARAMETERS=SIMULATION_PARAMETERS,
                    SIMULATION_DATA=SIMULATION_DATA,
                )
            else:
                raise Exception(
                    "Wrong 'ENGINE_TYPE' specified. "
                    "It should be either 'diesel' or 'electric'."
                )

            self.current_location_ID = self.patient.patient_location_ID
            if SIMULATION_PARAMETERS["PRINT"]:
                print(
                    f"{self.env.now}: Ambulance {self.ambulance_ID} "
                    f"arrived at patient {self.patient.patient_ID}."
                )

        finally:
            # Remove the arrival time variable
            # Driving data for the partial journey will be updated in the process_patient
            # exception handler
            del self.scene_arrival_time
            del self.scene_drive_time_start

    def on_site_aid_patient(
        self,
        on_site_aid_times: np.ndarray,
        SIMULATION_PARAMETERS: dict[str, Any],
        SIMULATION_DATA: dict[str, Any],
    ):
        """
        Help the patient on site.

        Parameters
        ----------
        patient_ID : int
            The patient ID.
        on_site_aid_times : np.ndarray
            Contains the on-site aid times.
        SIMULATION_PARAMETERS : dict[str, Any]
            The simulation parameters. The parameters ``PRINT`` and
            ``ENGINE_TYPE`` are at least necessary. Note that methods that are
            called within this method may require more parameters. See
            ``main.py`` for parameter explanations.
        SIMULATION_DATA : dict[str, Any]
            The simulation data. ``output_patient`` is at least necessary. See
            ``main.py`` and the input data section on the ELASPY website for
            explanations. Note that methods that are called within this method
            may require more data.

        Raises
        ------
        Exception
            If an invalid ``ENGINE_TYPE`` is specified.

        """
        if SIMULATION_PARAMETERS["PRINT"]:
            print(
                f"{self.env.now}: Ambulance {self.ambulance_ID} "
                f"treats patient {self.patient.patient_ID} on site "
                f"({self.current_location_ID}) in "
                f"{on_site_aid_times[self.patient.patient_ID]}."
            )
        yield self.env.timeout(on_site_aid_times[self.patient.patient_ID])
        SIMULATION_DATA["output_patient"][self.patient.patient_ID, 10] = (
            on_site_aid_times[self.patient.patient_ID]
        )

        if SIMULATION_PARAMETERS["ENGINE_TYPE"] == "electric":
            battery_reduction = Ambulance.calculate_battery_reduction_idling(
                on_site_aid_times[self.patient.patient_ID], SIMULATION_PARAMETERS
            )
            self.add_ambulance_data_battery_decrease(
                battery_reduction,
                idle=True,
                idle_time=on_site_aid_times[self.patient.patient_ID],
                driven_km=None,
                source_location_ID=None,
                target_location_ID=None,
                SIMULATION_PARAMETERS=SIMULATION_PARAMETERS,
                SIMULATION_DATA=SIMULATION_DATA,
            )
            self.decrease_battery(battery_reduction)
            if SIMULATION_PARAMETERS["PRINT"]:
                print(
                    f"{self.env.now}: The battery of ambulance "
                    f"{self.ambulance_ID} is reduced by: {battery_reduction}. "
                    "The current battery level of ambulance "
                    f"{self.ambulance_ID} is {self.battery}."
                )
        elif SIMULATION_PARAMETERS["ENGINE_TYPE"] == "diesel":
            self.add_ambulance_data_diesel(
                idle=True,
                idle_time=on_site_aid_times[self.patient.patient_ID],
                source_location_ID=None,
                target_location_ID=None,
                SIMULATION_PARAMETERS=SIMULATION_PARAMETERS,
                SIMULATION_DATA=SIMULATION_DATA,
            )
        else:
            raise Exception(
                "Wrong 'ENGINE_TYPE' specified. "
                "It should be either 'diesel' or 'electric'."
            )

        if SIMULATION_PARAMETERS["PRINT"]:
            print(
                f"{self.env.now}: Ambulance {self.ambulance_ID} "
                f"treated patient {self.patient.patient_ID} on site."
            )

    def go_to_hospital(
        self,
        SIMULATION_PARAMETERS: dict[str, Any],
        SIMULATION_DATA: dict[str, Any],
    ):
        """
        Drive to the hospital.

        Parameters
        ----------
        patient_ID : int
            The patient ID.
        hospital_location_ID : int
            The assigned hospital.
        SIMULATION_PARAMETERS : dict[str, Any]
            The simulation parameters. The parameters ``PRINT`` and
            ``ENGINE_TYPE`` are at least necessary. Note that methods that are
            called within this method may require more parameters. See
            ``main.py`` for parameter explanations.
        SIMULATION_DATA : dict[str, Any]
            The simulation data. ``output_patient`` and
            ``SIREN_DRIVING_MATRIX`` are at least necessary.
            ``SIREN_DRIVING_MATRIX`` is based on ``TRAVEL_TIMES_FILE``.
            See ``main.py`` and the input data section on the
            ELASPY website for explanations. Note that methods that are called
            within this method may require more data.

        Raises
        ------
        Exception
            If an invalid ``ENGINE_TYPE`` is specified.

        """
        # hospital_ID = select_hospital(self.current_location_ID)
        to_hospital_travel_time = SIMULATION_DATA["SIREN_DRIVING_MATRIX"].loc[
            self.current_location_ID, self.patient.hospital_location_ID
        ]
        SIMULATION_DATA["output_patient"][self.patient.patient_ID, 13] = (
            to_hospital_travel_time
        )
        if SIMULATION_PARAMETERS["PRINT"]:
            print(
                f"{self.env.now}: Ambulance {self.ambulance_ID} drives "
                f"patient {self.patient.patient_ID} to hospital {self.patient.hospital_location_ID} "
                f"in {to_hospital_travel_time}."
            )
        yield self.env.timeout(to_hospital_travel_time)

        if SIMULATION_PARAMETERS["ENGINE_TYPE"] == "electric":
            (
                battery_reduction,
                distance_km,
            ) = Ambulance.calculate_battery_reduction_and_distance_driving(
                self.current_location_ID,
                self.patient.hospital_location_ID,
                siren_off=False,
                SIMULATION_PARAMETERS=SIMULATION_PARAMETERS,
                SIMULATION_DATA=SIMULATION_DATA,
            )
            self.add_ambulance_data_battery_decrease(
                battery_reduction,
                idle=False,
                idle_time=None,
                driven_km=distance_km,
                source_location_ID=self.current_location_ID,
                target_location_ID=self.patient.hospital_location_ID,
                SIMULATION_PARAMETERS=SIMULATION_PARAMETERS,
                SIMULATION_DATA=SIMULATION_DATA,
            )
            self.decrease_battery(battery_reduction)
            if SIMULATION_PARAMETERS["PRINT"]:
                print(
                    f"{self.env.now}: The battery of ambulance "
                    f"{self.ambulance_ID} is reduced by: {battery_reduction}. "
                    "The current battery level of ambulance "
                    f"{self.ambulance_ID} is {self.battery}."
                )
        elif SIMULATION_PARAMETERS["ENGINE_TYPE"] == "diesel":
            self.add_ambulance_data_diesel(
                idle=False,
                idle_time=None,
                source_location_ID=self.current_location_ID,
                target_location_ID=self.patient.hospital_location_ID,
                SIMULATION_PARAMETERS=SIMULATION_PARAMETERS,
                SIMULATION_DATA=SIMULATION_DATA,
            )
        else:
            raise Exception(
                "Wrong 'ENGINE_TYPE' specified. "
                "It should be either 'diesel' or 'electric'."
            )

        self.current_location_ID = self.patient.hospital_location_ID
        if SIMULATION_PARAMETERS["PRINT"]:
            print(
                f"{self.env.now}: Ambulance {self.ambulance_ID} "
                f"arrived at {self.current_location_ID} "
                f"(hospital {self.patient.hospital_location_ID})."
            )

    def drop_off_time(
        self,
        drop_off_times: np.ndarray,
        charging_stations_hospitals: dict[
            str, list[sp.resources.resource.Resource | float]
        ],
        SIMULATION_PARAMETERS: dict[str, Any],
        SIMULATION_DATA: dict[str, Any],
    ):
        """
        Drop the patient off at the hospital.

        Parameters
        ----------
        patient_ID : int
            The patient ID.
        hospital_location_ID : int
            The assigned hospital.
        drop_off_times : np.ndarray
            Contains the drop-off times.
        charging_stations_hospitals : dict[str, list[sp.resources.resource.Resource | float]]
            The charging stations resources at all hospitals together with
            their charging speeds.
        SIMULATION_PARAMETERS : dict[str, Any]
            The simulation parameters. The parameters ``PRINT`` and
            ``ENGINE_TYPE`` are at least necessary. Note that methods that are
            called within this method may require more parameters. See
            ``main.py`` for parameter explanations.
        SIMULATION_DATA : dict[str, Any]
            The simulation data. ``output_patient`` is at least necessary. See
            ``main.py`` and the input data section on the ELASPY website for
            explanations. Note that methods that are called within this method
            may require more data.

        """
        if SIMULATION_PARAMETERS["PRINT"]:
            print(
                f"{self.env.now}: Ambulance {self.ambulance_ID} drops "
                f"patient {self.patient.patient_ID} off at hospital "
                f"in {drop_off_times[self.patient.patient_ID]}."
            )

        SIMULATION_DATA["output_patient"][self.patient.patient_ID, 14] = drop_off_times[
            self.patient.patient_ID
        ]

        dropping_off_patient = self.env.timeout(drop_off_times[self.patient.patient_ID])

        if SIMULATION_PARAMETERS["ENGINE_TYPE"] == "diesel":
            yield dropping_off_patient
        else:  # Ambulance is electric.
            if (
                str(self.patient.hospital_location_ID)
                in charging_stations_hospitals.keys()
            ):
                if SIMULATION_PARAMETERS["PRINT"]:
                    print(
                        f"Hospital {self.patient.hospital_location_ID} is part of the "
                        "charging_stations_hospitals keys. "
                        "Charging is in principle possible. "
                        f"Ambulance {self.ambulance_ID} will try to "
                        "charge while it is dropping off the patient."
                    )
                charging = self.env.process(
                    self.charge_at_drop_off(
                        self.patient.hospital_location_ID,
                        charging_stations_hospitals,
                        SIMULATION_PARAMETERS,
                        SIMULATION_DATA,
                    )
                )
            else:
                if SIMULATION_PARAMETERS["PRINT"]:
                    print(
                        f"Hospital {self.patient.hospital_location_ID} is not part of the "
                        "charging_stations_hospitals keys. "
                        "The ambulance cannot charge during drop-off."
                    )

            yield dropping_off_patient
            if (
                str(self.patient.hospital_location_ID)
                in charging_stations_hospitals.keys()
            ):
                if not charging.triggered:
                    charging.interrupt("Dropped patient off. Stop charging.")

        if SIMULATION_PARAMETERS["PRINT"]:
            print(
                f"{self.env.now}: Ambulance {self.ambulance_ID} finished handover of "
                f"patient {self.patient.patient_ID} off at hospital "
                f"{self.patient.hospital_location_ID}."
            )
            if SIMULATION_PARAMETERS["ENGINE_TYPE"] == "electric":
                print(
                    "The current battery level of "
                    f"ambulance {self.ambulance_ID} is {self.battery}."
                )

    def go_to_base_station(
        self,
        SIMULATION_PARAMETERS: dict[str, Any],
        SIMULATION_DATA: dict[str, Any],
    ):
        """
        Drive to its base station.

        Parameters
        ----------
        SIMULATION_PARAMETERS : dict[str, Any]
            The simulation parameters. The parameters ``NO_SIREN_PENALTY``,
            ``PRINT`` and ``ENGINE_TYPE`` are at least necessary. Note that
            methods that are called within this method may require more
            parameters. See ``main.py`` for parameter explanations.
        SIMULATION_DATA : dict[str, Any]
            The simulation data. ``SIREN_DRIVING_MATRIX ``is at least necessary.
            It is based on ``TRAVEL_TIMES_FILE``. See ``main.py`` and the
            input data section on the ELASPY website for explanations.
            Note that methods that are called within this method may require
            more data.

        Raises
        ------
        Exception
            1. If there is a queue at the ambulance resource.
            2. If the variable ``drives_to_base`` conflicts with the variables
            ``charges``, ``assigned_to_patient`` and/or ``helps_patient``.
            3. If an invalid ``ENGINE_TYPE`` is specified.

        Returns
        -------
        driving_interrupted : bool
            Whether the driving was interrupted or not.

        """

        # Check if the ambulance is already where it's supposed to be
        if self.current_location_ID == self.base_location_ID:
            if SIMULATION_PARAMETERS["PRINT"]:
                print(
                    f"{self.env.now}: Ambulance {self.ambulance_ID} is already at its base location"
                )
            return False

        # Driving to base uses ambulance resource; always possible because of earlier check (when the patient was assigned).
        # So that ambulance is used according to Simpy.
        driving_interrupted = False
        with self.resource.request(priority=2) as req:
            if len(self.resource.queue) > 0:
                raise Exception(
                    "There should not be a queue when an ambulance"
                    "starts driving to its base. Error.",
                    self.current_location_ID,
                    self.base_location_ID,
                    self.resource,
                    self.resource.queue,
                )
            yield req

            if self.charges or self.assigned_to_patient or self.helps_patient:
                raise Exception(
                    "Error: The variable 'drives_to_base' will be "
                    "set to True, but one or more of the variables "
                    "'charges', 'assigned_to_patient' or "
                    "'helps_patient' is also True. "
                    "This is not possible."
                )

            self.drives_to_base = True
            self.drives_to_base_start_time = self.env.now
            try:
                to_base_station_driving_time = (
                    SIMULATION_DATA["SIREN_DRIVING_MATRIX"].loc[
                        self.current_location_ID, self.base_location_ID
                    ]
                    / SIMULATION_PARAMETERS["NO_SIREN_PENALTY"]
                )
                if SIMULATION_PARAMETERS["PRINT"]:
                    print(
                        f"{self.env.now}: Ambulance {self.ambulance_ID} "
                        f"goes from {self.current_location_ID} to its base "
                        f"station at {self.base_location_ID} in "
                        f"{to_base_station_driving_time}."
                    )
                    print(f"Drives to base is: {self.drives_to_base}.")
                yield self.env.timeout(to_base_station_driving_time)

                if SIMULATION_PARAMETERS["ENGINE_TYPE"] == "electric":
                    (
                        battery_reduction,
                        distance_km,
                    ) = Ambulance.calculate_battery_reduction_and_distance_driving(
                        self.current_location_ID,
                        self.base_location_ID,
                        siren_off=True,
                        SIMULATION_PARAMETERS=SIMULATION_PARAMETERS,
                        SIMULATION_DATA=SIMULATION_DATA,
                    )
                    self.add_ambulance_data_battery_decrease(
                        battery_reduction,
                        idle=False,
                        idle_time=None,
                        driven_km=distance_km,
                        source_location_ID=self.current_location_ID,
                        target_location_ID=self.base_location_ID,
                        SIMULATION_PARAMETERS=SIMULATION_PARAMETERS,
                        SIMULATION_DATA=SIMULATION_DATA,
                    )
                    self.decrease_battery(battery_reduction)
                elif SIMULATION_PARAMETERS["ENGINE_TYPE"] == "diesel":
                    self.add_ambulance_data_diesel(
                        idle=False,
                        idle_time=None,
                        source_location_ID=self.current_location_ID,
                        target_location_ID=self.base_location_ID,
                        SIMULATION_PARAMETERS=SIMULATION_PARAMETERS,
                        SIMULATION_DATA=SIMULATION_DATA,
                    )
                else:
                    raise Exception(
                        "Wrong 'ENGINE_TYPE' specified. "
                        "It should be either 'diesel' or 'electric'."
                    )

                self.current_location_ID = self.base_location_ID
                if SIMULATION_PARAMETERS["PRINT"]:
                    print(
                        f"{self.env.now}: Ambulance {self.ambulance_ID} "
                        f"is at its base at {self.current_location_ID}."
                    )
                    if SIMULATION_PARAMETERS["ENGINE_TYPE"] == "electric":
                        print(
                            f"Its battery is reduced by: {battery_reduction}. "
                            "The current battery level of ambulance "
                            f"{self.ambulance_ID} is {self.battery}."
                        )
            except sp.Interrupt as interrupt:
                if SIMULATION_PARAMETERS["PRINT"]:
                    print("Preemption. While driving to base.")
                driven_time = self.env.now - interrupt.cause.usage_since  # type: ignore
                if SIMULATION_PARAMETERS["PRINT"]:
                    print(f"Driving since {interrupt.cause.usage_since}.")  # type: ignore
                    print(
                        f"Ambulance {self.ambulance_ID} got preempted by "
                        f"{interrupt.cause.by} at {self.env.now}"  # type: ignore
                        f" after {driven_time}."
                    )

                (new_x, new_y) = calculate_new_coordinate(
                    driven_time,
                    self.current_location_ID,
                    self.base_location_ID,
                    True,
                    SIMULATION_PARAMETERS,
                    SIMULATION_DATA,
                )
                new_location_ID = select_closest_location_ID(
                    (new_x, new_y), SIMULATION_PARAMETERS, SIMULATION_DATA
                )

                if SIMULATION_PARAMETERS["ENGINE_TYPE"] == "electric":
                    (
                        battery_reduction,
                        distance_km,
                    ) = Ambulance.calculate_battery_reduction_and_distance_driving(
                        self.current_location_ID,
                        new_location_ID,
                        siren_off=True,
                        SIMULATION_PARAMETERS=SIMULATION_PARAMETERS,
                        SIMULATION_DATA=SIMULATION_DATA,
                    )
                    self.add_ambulance_data_battery_decrease(
                        battery_reduction,
                        idle=False,
                        idle_time=None,
                        driven_km=distance_km,
                        source_location_ID=self.current_location_ID,
                        target_location_ID=new_location_ID,
                        SIMULATION_PARAMETERS=SIMULATION_PARAMETERS,
                        SIMULATION_DATA=SIMULATION_DATA,
                    )
                    self.decrease_battery(battery_reduction)
                    if SIMULATION_PARAMETERS["PRINT"]:
                        print(
                            f"The battery of ambulance {self.ambulance_ID} "
                            f"is reduced by: {battery_reduction}. "
                            "The current battery level of ambulance "
                            f"{self.ambulance_ID} is {self.battery}."
                        )
                elif SIMULATION_PARAMETERS["ENGINE_TYPE"] == "diesel":
                    self.add_ambulance_data_diesel(
                        idle=False,
                        idle_time=None,
                        source_location_ID=self.current_location_ID,
                        target_location_ID=new_location_ID,
                        SIMULATION_PARAMETERS=SIMULATION_PARAMETERS,
                        SIMULATION_DATA=SIMULATION_DATA,
                    )
                else:
                    raise Exception(
                        "Wrong 'ENGINE_TYPE' specified. "
                        "It should be either 'diesel' or 'electric'."
                    )

                self.current_location_ID = new_location_ID
                driving_interrupted = True

        self.drives_to_base = False
        self.drives_to_base_start_time = None
        if SIMULATION_PARAMETERS["PRINT"]:
            print(f"Drives to base is: {self.drives_to_base}.")

        return driving_interrupted

    def check_base_reachable(
        self,
        SIMULATION_PARAMETERS: dict[str, Any],
        SIMULATION_DATA: dict[str, Any],
    ) -> bool:
        """
        Checks whether the base is reachable with the current battery level.

        Parameters
        ----------
        SIMULATION_PARAMETERS : dict[str, Any]
            The simulation parameters. The parameter ``PRINT`` is at least
            necessary. Note that methods that are called within this method may
            require more parameters. See ``main.py`` for parameter explanations.
        SIMULATION_DATA : dict[str, Any]
            The simulation data. Methods that are called within this method
            require data. See these methods for explanations.

        Returns
        -------
        bool
            Whether the base is reachable or not.

        """

        (
            required_battery_reduction_base,
            _,
        ) = Ambulance.calculate_battery_reduction_and_distance_driving(
            self.current_location_ID,
            self.base_location_ID,
            siren_off=True,
            SIMULATION_PARAMETERS=SIMULATION_PARAMETERS,
            SIMULATION_DATA=SIMULATION_DATA,
        )

        if self.battery >= required_battery_reduction_base:
            if SIMULATION_PARAMETERS["PRINT"]:
                print(
                    f"The battery level is {self.battery}, which is "
                    "larger than the required battery level "
                    f"({required_battery_reduction_base}) so ambulance "
                    f"{self.ambulance_ID} drives to its base to charge."
                )
            return True
        else:
            if SIMULATION_PARAMETERS["PRINT"]:
                print(
                    f"The battery level is {self.battery}, which is smaller "
                    "than the required battery level "
                    f"({required_battery_reduction_base}) so ambulance "
                    f"{self.ambulance_ID} will charge at the hospital "
                    "until it can reach its base."
                )
            return False

    def add_ambulance_data_diesel(
        self,
        idle: bool,
        idle_time: Optional[float],
        source_location_ID: Optional[int],
        target_location_ID: Optional[int],
        SIMULATION_PARAMETERS: dict[str, Any],
        SIMULATION_DATA: dict[str, Any],
    ) -> None:
        """
        Adds data for a diesel vehicle to output_ambulance.

        Parameters
        ----------
        idle : bool
            Whether the ambulance was idle/stationary or not.
        idle_time : Optional[float]
            The idle/stationary time of the ambulance.
        source_location_ID : Optional[int]
            The initial location (i.e., source) of the ambulance.
        target_location_ID : Optional[int]
            The target location the ambulance drove to.
        SIMULATION_PARAMETERS : dict[str, Any]
            The simulation parameters. The parameter ``DATA_COLUMNS_AMBULANCE``
            is at least necessary. See ``main.py`` for parameter explanations.
        SIMULATION_DATA : dict[str, Any]
            The simulation data. ``output_ambulance`` is at least necessary.
            See ``main.py`` and the input data section on the ELASPY website
            for explanations.

        """

        # Add new row with nans
        SIMULATION_DATA["output_ambulance"] = np.append(
            SIMULATION_DATA["output_ambulance"],
            np.full(
                (1, len(SIMULATION_PARAMETERS["DATA_COLUMNS_AMBULANCE"])),
                np.nan,
            ),
            axis=0,
        )

        # add general data
        SIMULATION_DATA["output_ambulance"][-1, 0] = self.ambulance_ID
        SIMULATION_DATA["output_ambulance"][-1, 1] = self.env.now

        if idle:
            # ambulance was idle
            SIMULATION_DATA["output_ambulance"][-1, 5] = 0
            SIMULATION_DATA["output_ambulance"][-1, 6] = idle_time
        else:
            # ambulance was driving
            SIMULATION_DATA["output_ambulance"][-1, 5] = 1
            SIMULATION_DATA["output_ambulance"][-1, 7] = source_location_ID
            SIMULATION_DATA["output_ambulance"][-1, 8] = target_location_ID

    def add_ambulance_data_battery_decrease(
        self,
        decrease_quantity: float,
        idle: bool,
        idle_time: Optional[float],
        driven_km: Optional[float],
        source_location_ID: Optional[int],
        target_location_ID: Optional[int],
        SIMULATION_PARAMETERS: dict[str, Any],
        SIMULATION_DATA: dict[str, Any],
    ) -> None:
        """
        Adds battery decrease data to output_ambulance for an electric vehicle.

        Parameters
        ----------
        decrease_quantity : float
            The battery decrease in kWh.
        idle : bool
            Whether the ambulance was idle/stationary or not.
        idle_time : Optional[float]
            The idle/stationary time of the ambulance.
        driven_km : Optional[float]
            The number of kilometers driven.
        source_location_ID : Optional[int]
            The initial location (i.e., source) of the ambulance.
        target_location_ID : Optional[int]
            The target location the ambulance drove to.
        SIMULATION_PARAMETERS : dict[str, Any]
            The simulation parameters. The parameter ``DATA_COLUMNS_AMBULANCE``
            is at least necessary. See ``main.py`` for parameter explanations.
        SIMULATION_DATA : dict[str, Any]
            The simulation data. ``output_ambulance`` is at least necessary.
            See ``main.py`` and the input data section on the ELASPY website
            for explanations.

        """

        # Add new row with nans
        SIMULATION_DATA["output_ambulance"] = np.append(
            SIMULATION_DATA["output_ambulance"],
            np.full(
                (1, len(SIMULATION_PARAMETERS["DATA_COLUMNS_AMBULANCE"])),
                np.nan,
            ),
            axis=0,
        )

        # add general data
        SIMULATION_DATA["output_ambulance"][-1, 0] = self.ambulance_ID
        SIMULATION_DATA["output_ambulance"][-1, 1] = self.env.now
        SIMULATION_DATA["output_ambulance"][-1, 2] = self.battery
        SIMULATION_DATA["output_ambulance"][-1, 3] = self.battery - decrease_quantity
        # ambulance used battery
        SIMULATION_DATA["output_ambulance"][-1, 4] = 0

        if idle:
            # ambulance was idle
            SIMULATION_DATA["output_ambulance"][-1, 5] = 0
            SIMULATION_DATA["output_ambulance"][-1, 6] = idle_time

        else:
            # ambulance was driving
            SIMULATION_DATA["output_ambulance"][-1, 5] = 1
            SIMULATION_DATA["output_ambulance"][-1, 7] = source_location_ID
            SIMULATION_DATA["output_ambulance"][-1, 8] = target_location_ID
            SIMULATION_DATA["output_ambulance"][-1, 9] = driven_km

        SIMULATION_DATA["output_ambulance"][-1, 10] = decrease_quantity

    def add_ambulance_data_charging(
        self,
        charging_location_ID: int,
        speed_charger: float,
        waiting_time_at_charger: float,
        charging_type: int,
        charging_time: float,
        increase_quantity: float,
        charging_success: bool,
        charging_interrupted: int,
        SIMULATION_PARAMETERS: dict[str, Any],
        SIMULATION_DATA: dict[str, Any],
    ) -> None:
        """
        Adds battery increase data to output_ambulance for an electric vehicle.

        Parameters
        ----------
        charging_location_ID : int
            The charging location ID.
        speed_charger : float
            The speed of the charger (in kW).
        waiting_time_at_charger : float
            The waiting time in minutes before an ambulance could charge or
            was assigned to another patient while waiting to charge.
        charging_type : int
            The charging type. "2" for charging at the base, "1" for charging
            at the hospital after treating a patient and "0" for charging
            during patient handover.
        charging_time : float
            The time the ambulance charged. It is 0 if the ambulance could not
            charge before it was assigned to a new patient.
        increase_quantity : float
            The battery increase in kWh.
        charging_success : bool
            Whether the ambulance could or could not charge during the charging
            session.
        charging_interrupted : int
            "1" if the charging session was interrupted,
            "0" if it was not interrupted.
        SIMULATION_PARAMETERS : dict[str, Any]
            The simulation parameters. The parameter ``DATA_COLUMNS_AMBULANCE``
            is at least necessary. See ``main.py`` for parameter explanations.
        SIMULATION_DATA : dict[str, Any]
            The simulation data. ``output_ambulance`` is at least necessary.
            See ``main.py`` and the input data section on the ELASPY website
            for explanations.

        """
        # charging_type 0: drop-off
        # charging_type 1: hospital
        # charging_type 2: base

        # Add new row with nans
        SIMULATION_DATA["output_ambulance"] = np.append(
            SIMULATION_DATA["output_ambulance"],
            np.full(
                (1, len(SIMULATION_PARAMETERS["DATA_COLUMNS_AMBULANCE"])),
                np.nan,
            ),
            axis=0,
        )

        # add general data
        SIMULATION_DATA["output_ambulance"][-1, 0] = self.ambulance_ID
        SIMULATION_DATA["output_ambulance"][-1, 1] = self.env.now
        SIMULATION_DATA["output_ambulance"][-1, 2] = self.battery
        SIMULATION_DATA["output_ambulance"][-1, 3] = self.battery + increase_quantity
        # ambulance charged battery
        SIMULATION_DATA["output_ambulance"][-1, 4] = 1

        SIMULATION_DATA["output_ambulance"][-1, 11] = charging_type
        SIMULATION_DATA["output_ambulance"][-1, 12] = charging_location_ID
        SIMULATION_DATA["output_ambulance"][-1, 13] = speed_charger

        if charging_success:
            SIMULATION_DATA["output_ambulance"][-1, 14] = 1
            SIMULATION_DATA["output_ambulance"][-1, 15] = waiting_time_at_charger
            SIMULATION_DATA["output_ambulance"][-1, 16] = charging_interrupted
            SIMULATION_DATA["output_ambulance"][-1, 17] = charging_time
            SIMULATION_DATA["output_ambulance"][-1, 18] = increase_quantity
        else:
            SIMULATION_DATA["output_ambulance"][-1, 14] = 0
            SIMULATION_DATA["output_ambulance"][-1, 15] = waiting_time_at_charger
            SIMULATION_DATA["output_ambulance"][-1, 16] = charging_interrupted

    def decrease_battery(self, decrease_quantity: float) -> None:
        """
        Decreases the battery with a certain amount.

        Parameters
        ----------
        decrease_quantity : float
            The battery decrease in kWh.

        Raises
        ------
        Exception
            If the battery level of the ambulance falls below 0.

        """
        self.battery = self.battery - decrease_quantity

        # Due to Python's floating point arithmetic, it can be the case that
        # self.battery < 0 but only on many decimals after the comma. In these
        # cases, the self.battery value is set to 0 explicitly. Abs used as
        # comparison is with 0 (see Pytest documentation).
        if self.battery == approx(0, abs=1e-14):
            self.battery = 0

        if self.battery < 0:
            raise Exception(
                f"The battery of ambulance {self.ambulance_ID} "
                "is lower than 0. Exit code."
            )

    def increase_battery(self, increase_quantity: float) -> None:
        """
        Increases the battery with a certain amount.

        Parameters
        ----------
        increase_quantity : float
            The battery increase in kWh.

        Raises
        ------
        Exception
            If the battery level of the ambulance becomes higher than the
            ``MAX_BATTERY_LEVEL``.
        """

        self.battery = self.battery + increase_quantity

        # Due to Python's floating point arithmetic, it can be the case that
        # self.battery > MAX_BATTERY_LEVEL but only on many decimals
        # after the comma. In these cases, the self.battery value is set to
        # MAX_BATTERY_LEVEL explicitly.
        if self.battery == approx(self.MAX_BATTERY_LEVEL, abs=1e-14):
            self.battery = self.MAX_BATTERY_LEVEL

        if self.battery > self.MAX_BATTERY_LEVEL:
            raise Exception(
                f"The battery of ambulance {self.ambulance_ID} is "
                "charged more than its MAX_BATTERY_LEVEL. "
                "Exit code."
            )

    def calculate_battery_increase_until_now(self) -> float:
        """
        Calculates the battery increase due to charging until this moment.

        Returns
        -------
        float
            The battery increase in kWh.

        """
        return ((self.env.now - self.charging_since) / 60) * self.speed_charger

    def charge_at_base(
        self,
        charging_stations_bases: dict[
            str, list[sp.resources.resource.Resource | float]
        ],
        SIMULATION_PARAMETERS: dict[str, Any],
        SIMULATION_DATA: dict[str, Any],
    ):
        """
        Charge at its base station.

        Parameters
        ----------
        charging_stations_bases : dict[str, list[sp.resources.resource.Resource | float]]
            The charging stations resources at all bases together with their
            charging speeds.
        SIMULATION_PARAMETERS : dict[str, Any]
            The simulation parameters. The parameter ``PRINT`` is at least
            necessary. See ``main.py`` for parameter explanations.
        SIMULATION_DATA : dict[str, Any]
            The simulation data. Methods that are called within this method
            require data. See these methods for explanations.

        """
        if SIMULATION_PARAMETERS["PRINT"]:
            print(f"Ambulance {self.ambulance_ID} will charge at its base.")

        selected_charger, speed_charger = Ambulance.select_charging_station(
            charging_stations_bases,
            self.base_location_ID,
            SIMULATION_PARAMETERS,
            SIMULATION_DATA,
        )

        required_increase_battery_to_full = self.MAX_BATTERY_LEVEL - self.battery

        if SIMULATION_PARAMETERS["PRINT"]:
            print(
                f"{self.env.now}: ambulance {self.ambulance_ID} will charge "
                "at its base until its battery is full "
                f"(battery increase of {required_increase_battery_to_full} "
                "kWh is necessary)."
            )

        yield self.env.process(
            self.charge_battery(
                selected_charger,
                speed_charger,
                required_increase_battery_to_full,
                self.base_location_ID,
                charging_type=2,
                SIMULATION_PARAMETERS=SIMULATION_PARAMETERS,
                SIMULATION_DATA=SIMULATION_DATA,
            )
        )

    def charge_at_hospital(
        self,
        hospital_location_ID: int,
        charging_stations_hospitals: dict[
            str, list[sp.resources.resource.Resource | float]
        ],
        SIMULATION_PARAMETERS: dict[str, Any],
        SIMULATION_DATA: dict[str, Any],
    ):
        """
        Charge at a hospital.

        Parameters
        ----------
        hospital_location_ID : int
            The hospital location ID.
        charging_stations_hospitals : dict[str, list[sp.resources.resource.Resource | float]]
            The charging stations resources at all hospitals together with
            their charging speeds.
        SIMULATION_PARAMETERS : dict[str, Any]
            The simulation parameters. The parameter ``PRINT`` is at least
            necessary. Note that methods that are called within this method
            may require more parameters. See ``main.py`` for parameter
            explanations.
        SIMULATION_DATA : dict[str, Any]
            The simulation data. Methods that are called within this method
            require data. See these methods for explanations.

        Raises
        ------
        Exception
            If the ambulance is not at a hospital, but it can also not reach
            its base.

        Returns
        ------
        charging_interrupted : bool
            Whether the charging was interrupted or not.

        """

        if self.current_location_ID != hospital_location_ID:
            raise Exception(
                f"Ambulance {self.ambulance_ID} is not at "
                f"hospital {hospital_location_ID}, but it also "
                "cannot reach its base. Exit code."
            )

        else:
            if SIMULATION_PARAMETERS["PRINT"]:
                print(f"Ambulance {self.ambulance_ID} will charge at the hospital.")

            (
                selected_charger,
                speed_charger,
            ) = Ambulance.select_charging_station(
                charging_stations_hospitals,
                hospital_location_ID,
                SIMULATION_PARAMETERS,
                SIMULATION_DATA,
            )

            required_increase_battery_to_base = (
                Ambulance.calculate_battery_reduction_and_distance_driving(
                    hospital_location_ID,
                    self.base_location_ID,
                    siren_off=True,
                    SIMULATION_PARAMETERS=SIMULATION_PARAMETERS,
                    SIMULATION_DATA=SIMULATION_DATA,
                )[0]
                - self.battery
            )

            if SIMULATION_PARAMETERS["PRINT"]:
                print(
                    f"{self.env.now}: ambulance {self.ambulance_ID} will "
                    "charge at the hospital until it can reach its base "
                    "(a battery increase of "
                    f"{required_increase_battery_to_base} kWh is required)."
                )
            charging_interrupted = yield self.env.process(
                self.charge_battery(
                    selected_charger,
                    speed_charger,
                    required_increase_battery_to_base,
                    hospital_location_ID,
                    charging_type=1,
                    SIMULATION_PARAMETERS=SIMULATION_PARAMETERS,
                    SIMULATION_DATA=SIMULATION_DATA,
                )
            )

        return charging_interrupted

    def charge_battery(
        self,
        selected_charger: sp.resources.resource.Resource,
        speed_charger: float,
        required_increase_battery: float,
        charging_location_ID: int,
        charging_type: int,
        SIMULATION_PARAMETERS: dict[str, Any],
        SIMULATION_DATA: dict[str, Any],
    ):
        """
        Charges the ambulance battery and interrupts charging if necessary.

        Parameters
        ----------
        selected_charger : sp.resources.resource.Resource
            The charger that was selected to use during charging.
        speed_charger : float
            The speed of the charger in kW.
        required_increase_battery : float
            The required battery increase in kWh.
        charging_location_ID : int
            The charging location ID.
        charging_type : int
            The charging type. "2" for charging at the base, "1" for charging
            at the hospital after treating a patient and "0" for charging
            during patient handover..
        SIMULATION_PARAMETERS : dict[str, Any]
            The simulation parameters. The parameter ``PRINT`` is at least
            necessary. Note that methods that are called within this method
            may require more parameters. See ``main.py`` for parameter
            explanations.
        SIMULATION_DATA : dict[str, Any]
            The simulation data. Methods that are called within this method
            require data. See these methods for explanations.

        Raises
        ------
        Exception
            1. If there is a queue at the ambulance resource.
            2. If the variable ``charges`` conflicts with ``drives_to_base``.
            3. If the variable ``charges`` conflicts with ``assigned_to_patient``
            and/or ``helps_patient``.
            4. If the request was triggered, but the ambulance was not charged.

        Return
        ------
        charging_interrupted : bool
            Whether the charging was interrupted or not.

        """

        charging_interrupted = False
        charging_time = Ambulance.calculate_charging_time(
            required_increase_battery, speed_charger
        )
        if SIMULATION_PARAMETERS["PRINT"]:
            print(
                f"Ambulance {self.ambulance_ID} will charge "
                f"{required_increase_battery} kWh in {charging_time}."
            )
        with self.resource.request(priority=2) as req:
            if len(self.resource.queue) > 0:
                raise Exception(
                    "There should not be a queue when an ambulance "
                    "starts charging. Error."
                )
            yield req
            try:
                request = selected_charger.request()
                if SIMULATION_PARAMETERS["PRINT"]:
                    print(f"Before charging the queue is: {selected_charger.queue}.")
                start_waiting = self.env.now
                yield request
                waiting_time_at_charger = self.env.now - start_waiting
                if SIMULATION_PARAMETERS["PRINT"]:
                    print(
                        f"Ambulance {self.ambulance_ID} has waited for "
                        f"{waiting_time_at_charger} before it could charge."
                    )
                    print(
                        "When assigned to the charger, the queue is: "
                        f"{selected_charger.queue}."
                    )
                if self.drives_to_base:
                    raise Exception(
                        "Error: the variable 'charges' will be set "
                        "to True, but the variable 'drives_to_base' "
                        "is also True. This is not possible."
                    )
                if self.helps_patient or self.assigned_to_patient:
                    raise Exception(
                        "Error: The variable 'charges' will be "
                        "set to True in charge_battery(), but one "
                        "or more of the variables "
                        "'assigned_to_patient' or 'helps_patient' "
                        "is also True. This is not possible in the "
                        "charge_battery() function."
                    )
                self.charges = True
                self.charging_since = self.env.now
                self.speed_charger = speed_charger
                if SIMULATION_PARAMETERS["PRINT"]:
                    print(
                        f"Ambulance {self.ambulance_ID} has started "
                        f"charging at: {self.charging_since}."
                    )
                    print(f"The required charging time is: {charging_time}.")
                yield self.env.timeout(charging_time)
                self.add_ambulance_data_charging(
                    charging_location_ID,
                    speed_charger,
                    waiting_time_at_charger,
                    charging_type,
                    charging_time,
                    required_increase_battery,
                    charging_success=True,
                    charging_interrupted=0,
                    SIMULATION_PARAMETERS=SIMULATION_PARAMETERS,
                    SIMULATION_DATA=SIMULATION_DATA,
                )
                self.increase_battery(required_increase_battery)
                selected_charger.release(request)
                if SIMULATION_PARAMETERS["PRINT"]:
                    print(
                        f"{self.env.now}: ambulance {self.ambulance_ID} "
                        f"has finished charging. "
                        f"Its battery level is {self.battery}."
                    )
            except sp.Interrupt as interrupt:
                if SIMULATION_PARAMETERS["PRINT"]:
                    print("PREEMPTION. While charging.")
                    print(
                        f"Ambulance {self.ambulance_ID} got preempted by "
                        f"{interrupt.cause.by} at {self.env.now}. "  # type: ignore
                        f"Charging since: {self.charging_since}."
                    )

                if self.charging_since is not np.nan:
                    increase_battery_after_interrupt = (
                        self.calculate_battery_increase_until_now()
                    )
                    if SIMULATION_PARAMETERS["PRINT"]:
                        print(
                            f"The battery of ambulance {self.ambulance_ID} "
                            f"was {self.battery} and is increased by "
                            f"{increase_battery_after_interrupt}."
                        )
                    achieved_charging_time = self.env.now - self.charging_since
                    self.add_ambulance_data_charging(
                        charging_location_ID,
                        speed_charger,
                        waiting_time_at_charger,
                        charging_type,
                        achieved_charging_time,
                        increase_battery_after_interrupt,
                        charging_success=True,
                        charging_interrupted=1,
                        SIMULATION_PARAMETERS=SIMULATION_PARAMETERS,
                        SIMULATION_DATA=SIMULATION_DATA,
                    )
                    self.increase_battery(increase_battery_after_interrupt)
                    if SIMULATION_PARAMETERS["PRINT"]:
                        print(f"The queue is: {selected_charger.queue}.")
                elif not request.triggered:
                    if SIMULATION_PARAMETERS["PRINT"]:
                        print(
                            "The request was not triggered. "
                            "The ambulance has not been charging until now."
                        )
                    selected_charger.put_queue.remove(request)
                    waiting_time_at_charger = self.env.now - start_waiting
                    self.add_ambulance_data_charging(
                        charging_location_ID,
                        speed_charger,
                        waiting_time_at_charger,
                        charging_type,
                        charging_time=0,
                        increase_quantity=0,
                        charging_success=False,
                        charging_interrupted=1,
                        SIMULATION_PARAMETERS=SIMULATION_PARAMETERS,
                        SIMULATION_DATA=SIMULATION_DATA,
                    )
                    if SIMULATION_PARAMETERS["PRINT"]:
                        print(
                            f"Ambulance {self.ambulance_ID} has been "
                            f"waiting for {waiting_time_at_charger}."
                        )
                        print(
                            "After removal from the queue, the queue is: "
                            f"{selected_charger.queue}."
                        )
                else:
                    raise Exception(
                        "Request triggered, but ambulance not charged. Error."
                    )
                if SIMULATION_PARAMETERS["PRINT"]:
                    print(f"Before release users: {selected_charger.users}.")
                selected_charger.release(request)
                if SIMULATION_PARAMETERS["PRINT"]:
                    print(f"After release users: {selected_charger.users}.")
                    print(f"The queue is: {selected_charger.queue}.")

                charging_interrupted = True

        if SIMULATION_PARAMETERS["PRINT"]:
            print(
                f"The battery of ambulance {self.ambulance_ID} "
                f"is equal to: {self.battery} kWh."
            )
        self.charges = False
        self.charging_since = np.nan
        self.speed_charger = np.nan
        return charging_interrupted

    def charge_at_drop_off(
        self,
        hospital_location_ID: int,
        charging_stations_hospitals: dict[
            str, list[sp.resources.resource.Resource | float]
        ],
        SIMULATION_PARAMETERS: dict[str, Any],
        SIMULATION_DATA: dict[str, Any],
    ):
        """
        Charges the battery during patient handover. Interrupts if necessary.

        Parameters
        ----------
        hospital_location_ID : int
            The hospital location ID.
        charging_stations_hospitals : dict[str, list[sp.resources.resource.Resource | float]]
            The charging stations resources at all hospitals together with
            their charging speeds.
        SIMULATION_PARAMETERS : dict[str, Any]
            The simulation parameters. The parameter ``PRINT`` is at least
            necessary. Note that methods that are called within this method
            may require more parameters. See ``main.py`` for parameter
            explanations.
        SIMULATION_DATA : dict[str, Any]
            The simulation data. Methods that are called within this method
            require data. See these methods for explanations.

        Raises
        ------
        Exception
            1. If the variable ``charges`` conflicts with ``drives_to_base``.
            2. If ``helps_patient`` and/or ``assigned_to_patient`` are not True.
            3. If the request was triggered, but the ambulance was not charged.

        """

        selected_charger, speed_charger = Ambulance.select_charging_station(
            charging_stations_hospitals,
            hospital_location_ID,
            SIMULATION_PARAMETERS,
            SIMULATION_DATA,
        )

        required_increase_battery_to_full = self.MAX_BATTERY_LEVEL - self.battery

        charging_time = Ambulance.calculate_charging_time(
            required_increase_battery_to_full, speed_charger
        )

        charging_type = 0  # drop-off charging
        if SIMULATION_PARAMETERS["PRINT"]:
            print(
                f"{self.env.now}: ambulance {self.ambulance_ID} will charge "
                "during the drop-off for as long as it can "
                f"(maximum of {required_increase_battery_to_full} kWh "
                "for a full battery)."
            )

        try:
            request = selected_charger.request()
            if SIMULATION_PARAMETERS["PRINT"]:
                print(f"Before charging the queue is: {selected_charger.queue}.")
            start_waiting = self.env.now
            yield request
            waiting_time_at_charger = self.env.now - start_waiting
            if SIMULATION_PARAMETERS["PRINT"]:
                print(
                    f"Ambulance {self.ambulance_ID} has waited for "
                    f"{waiting_time_at_charger} before it could charge."
                )
                print(
                    "When assigned to the charger, the queue is: "
                    f"{selected_charger.queue}."
                )
            if self.drives_to_base:
                raise Exception(
                    "Error: the variable 'charges' will be set "
                    "to True, but the variable 'drives_to_base' "
                    "is also True. This is not possible."
                )
            if not (self.helps_patient and self.assigned_to_patient):
                raise Exception(
                    "Error: in charge_at_drop_off the variables "
                    "'helps_patient' and 'assigned_to_patient' "
                    "should be True, but at least one is False."
                )

            self.charges = True
            self.charging_since = self.env.now
            self.speed_charger = speed_charger
            if SIMULATION_PARAMETERS["PRINT"]:
                print(
                    f"Ambulance {self.ambulance_ID} has started "
                    f"charging at: {self.charging_since}."
                )
                print(
                    "The required time to charge to a full battery is: "
                    f"{charging_time}."
                )
            yield self.env.timeout(charging_time)

            self.add_ambulance_data_charging(
                hospital_location_ID,
                speed_charger,
                waiting_time_at_charger,
                charging_type,
                charging_time,
                required_increase_battery_to_full,
                charging_success=True,
                charging_interrupted=0,
                SIMULATION_PARAMETERS=SIMULATION_PARAMETERS,
                SIMULATION_DATA=SIMULATION_DATA,
            )
            self.increase_battery(required_increase_battery_to_full)
            selected_charger.release(request)
            if SIMULATION_PARAMETERS["PRINT"]:
                print(
                    f"{self.env.now}: ambulance {self.ambulance_ID} "
                    "has finished charging. "
                    f"Its battery level is {self.battery}."
                )

        except sp.Interrupt as interrupt:
            if SIMULATION_PARAMETERS["PRINT"]:
                print("Preemption. While charging during drop-off.")
                print(
                    f"Ambulance {self.ambulance_ID} got preempted by "
                    f"{interrupt.cause} at {self.env.now}. "  # type: ignore
                    f"Charging since: {self.charging_since}."
                )

            if self.charging_since is not np.nan:
                increase_battery_after_interrupt = (
                    self.calculate_battery_increase_until_now()
                )
                if SIMULATION_PARAMETERS["PRINT"]:
                    print(
                        f"The battery of ambulance {self.ambulance_ID} "
                        f"was {self.battery} and is increased by "
                        f"{increase_battery_after_interrupt}."
                    )
                achieved_charging_time = self.env.now - self.charging_since
                self.add_ambulance_data_charging(
                    hospital_location_ID,
                    speed_charger,
                    waiting_time_at_charger,
                    charging_type,
                    achieved_charging_time,
                    increase_battery_after_interrupt,
                    charging_success=True,
                    charging_interrupted=1,
                    SIMULATION_PARAMETERS=SIMULATION_PARAMETERS,
                    SIMULATION_DATA=SIMULATION_DATA,
                )
                self.increase_battery(increase_battery_after_interrupt)
                if SIMULATION_PARAMETERS["PRINT"]:
                    print(f"The queue is: {selected_charger.queue}.")
            elif not request.triggered:
                if SIMULATION_PARAMETERS["PRINT"]:
                    print(
                        "The request was not triggered. "
                        "The ambulance has not been charging until now."
                    )
                selected_charger.put_queue.remove(request)
                waiting_time_at_charger = self.env.now - start_waiting
                self.add_ambulance_data_charging(
                    hospital_location_ID,
                    speed_charger,
                    waiting_time_at_charger,
                    charging_type,
                    charging_time=0,
                    increase_quantity=0,
                    charging_success=False,
                    charging_interrupted=1,
                    SIMULATION_PARAMETERS=SIMULATION_PARAMETERS,
                    SIMULATION_DATA=SIMULATION_DATA,
                )
                if SIMULATION_PARAMETERS["PRINT"]:
                    print(
                        "After removal from the queue, the queue is: "
                        f"{selected_charger.queue}."
                    )
                    print(
                        f"Ambulance {self.ambulance_ID} has been waiting "
                        f"for {waiting_time_at_charger}."
                    )
            else:
                raise Exception("Request triggered, but ambulance not charged. Error.")

            if SIMULATION_PARAMETERS["PRINT"]:
                print(f"Before release users: {selected_charger.users}.")
            selected_charger.release(request)
            if SIMULATION_PARAMETERS["PRINT"]:
                print(f"After release users: {selected_charger.users}.")
                print(f"The queue is: {selected_charger.queue}.")

        self.charges = False
        self.charging_since = np.nan
        self.speed_charger = np.nan
        if SIMULATION_PARAMETERS["PRINT"]:
            print(
                f"The battery of ambulance {self.ambulance_ID} is equal "
                f"to: {self.battery} kWh."
            )

    def check_meal_break(self, SIMULATION_DATA, SIMULATION_PARAMETERS):
        "Allow the vehicle crew to take a break at their home station if appropriate"

        assert SIMULATION_PARAMETERS["ENGINE_TYPE"] == "diesel", (
            "Electric ambulances are not supported for meal breaks"
        )

        assert SIMULATION_PARAMETERS["WITH_MEAL_BREAKS"], (
            "Cannot check_meal_break if meal breaks are disabled"
        )

        if not self.had_meal_break and self.env.now >= self.meal_break_start:
            if self.shift_end_time:
                assert self.env.now < self.shift_end_time, (
                    f"Trying to schedule a meal break after the end of shift for Ambulance {self.ambulance_ID}"
                )

            if self.assigned_to_patient:
                # We'll check again automatically when we're done with this patient so return immediately
                return

            # Meal Break Policies
            if not self.mb_allowed_by_policy(self):
                print(
                    f"{self.env.now}: Tried to trigger a meal break for ambulance {self.ambulance_ID} but was disallowed by policy"
                )
                return

            if SIMULATION_PARAMETERS["PRINT"]:
                print(
                    f"{self.env.now}: Ambulance {self.ambulance_ID} is taking a",
                    f"late meal break ({self.env.now - self.meal_break_end} mins late)"
                    if self.env.now > self.meal_break_end
                    else "meal break",
                )

            self.in_meal_break = True
            meal_break_drive_start = self.env.now

            if self.drives_to_base:
                if SIMULATION_PARAMETERS["PRINT"]:
                    print(
                        f"{self.env.now}: Ambulance {self.ambulance_ID} is interrupted for a meal break while driving to base."
                    )

            # Determine the location the ambulance will now drive to for its break
            # This can be either "home" or "closest"
            current_loc_id = self.get_current_location_id(
                SIMULATION_PARAMETERS, SIMULATION_DATA
            )

            if SIMULATION_PARAMETERS["MEAL_BREAK_LOCATION"] == "home":
                destination_id = self.base_location_ID
                est_saving = None
            elif SIMULATION_PARAMETERS["MEAL_BREAK_LOCATION"] == "closest":
                station_loc_ids = SIMULATION_DATA["NODES_BASE_LOCATIONS"].iloc[:, 0]

                closest_station_id = (
                    SIMULATION_DATA["SIREN_DRIVING_MATRIX"]
                    .loc[
                        current_loc_id,
                        station_loc_ids,
                    ]
                    .idxmin()
                )

                time_to_base = (
                    SIMULATION_DATA["SIREN_DRIVING_MATRIX"].loc[
                        current_loc_id, self.base_location_ID
                    ]
                    / SIMULATION_PARAMETERS["NO_SIREN_PENALTY"]
                )

                time_to_closest = (
                    SIMULATION_DATA["SIREN_DRIVING_MATRIX"].loc[
                        current_loc_id, closest_station_id
                    ]
                    / SIMULATION_PARAMETERS["NO_SIREN_PENALTY"]
                )

                est_saving = time_to_base - time_to_closest
                if est_saving > SIMULATION_PARAMETERS["MEAL_BREAK_CLOSEST_MIN_SAVING"]:
                    destination_id = closest_station_id
                else:
                    destination_id = self.base_location_ID
                    est_saving = None

            else:
                raise NotImplementedError(
                    f"MEAL_BREAK_LOCATION must be set to either home or closest and not to {SIMULATION_PARAMETERS['MEAL_BREAK_LOCATION']}"
                )

            if SIMULATION_PARAMETERS["PRINT"]:
                if destination_id != self.base_location_ID:
                    print(
                        f"{self.env.now}: Ambulance {self.ambulance_ID} uses the closer location {destination_id} for its break (est saving {est_saving:g} minutes)."
                    )
                else:
                    print(
                        f"{self.env.now}: Ambulance {self.ambulance_ID} uses its base location {destination_id} for its break."
                    )

            with self.resource.request(priority=1) as req:
                assert not len(self.resource.queue), (
                    "Ambulance cannot take a break as it's already busy"
                )
                yield req

                driving_time = (
                    SIMULATION_DATA["SIREN_DRIVING_MATRIX"].loc[
                        current_loc_id, destination_id
                    ]
                    / SIMULATION_PARAMETERS["NO_SIREN_PENALTY"]
                )
                if SIMULATION_PARAMETERS["PRINT"]:
                    print(
                        f"{self.env.now}: Ambulance {self.ambulance_ID} "
                        f"goes from {self.current_location_ID} to its base "
                        f"station at {destination_id} in "
                        f"{driving_time} for a meal break."
                    )

                # Driving Time
                yield self.env.timeout(driving_time)
                self.current_location_ID = destination_id
                meal_break_drive_end = self.env.now

                self.add_ambulance_data_diesel(
                    idle=False,
                    idle_time=None,
                    source_location_ID=self.current_location_ID,
                    target_location_ID=destination_id,
                    SIMULATION_PARAMETERS=SIMULATION_PARAMETERS,
                    SIMULATION_DATA=SIMULATION_DATA,
                )

                if SIMULATION_PARAMETERS["PRINT"]:
                    print(
                        f"{self.env.now}: Ambulance {self.ambulance_ID} "
                        f"arrived at the station at {destination_id} "
                        f" for a meal break"
                    )

                # Actual Break Time
                if self.shift_start_time and self.shift_end_time:
                    mb_duration_mins = SIMULATION_PARAMETERS["MEAL_BREAK_DURATION"][
                        self.shift_duration
                    ]
                else:
                    mb_duration_mins = 45

                assert mb_duration_mins

                if SIMULATION_PARAMETERS["PRINT"]:
                    print(
                        f"{self.env.now}: Ambulance {self.ambulance_ID} is resting for {mb_duration_mins} minutes."
                    )

                yield self.env.timeout(mb_duration_mins)
                meal_break_end = self.env.now

                if SIMULATION_PARAMETERS["PRINT"]:
                    print(
                        f"{self.env.now}: Ambulance {self.ambulance_ID} "
                        f"completed its meal break at "
                        f"station {self.current_location_ID}."
                    )

                SIMULATION_DATA["output_breaks"].append(
                    (
                        self.ambulance_ID,
                        meal_break_drive_start,
                        meal_break_drive_end,
                        meal_break_end,
                    )
                )

                self.in_meal_break = False
                self.had_meal_break = True

    def get_current_location_id(self, SIMULATION_PARAMETERS, SIMULATION_DATA):
        # Ambulance is currently driving to a patient
        if hasattr(self, "scene_arrival_time"):
            driven_time = self.env.now - self.scene_drive_time_start
            (new_x, new_y) = calculate_new_coordinate(
                driven_time,
                self.current_location_ID,
                self.patient.patient_location_ID,
                siren_off=False,
                SIMULATION_PARAMETERS=SIMULATION_PARAMETERS,
                SIMULATION_DATA=SIMULATION_DATA,
            )
            return select_closest_location_ID(
                (new_x, new_y), SIMULATION_PARAMETERS, SIMULATION_DATA
            )

        assert not self.assigned_to_patient, (
            "Should only care about the current location of an assigned ambulance when driving to the patient"
        )

        # Ambulance is currently driving to its base
        if self.drives_to_base:
            driven_time = self.env.now - self.drives_to_base_start_time

            (new_x, new_y) = calculate_new_coordinate(
                driven_time,
                self.current_location_ID,
                self.base_location_ID,
                siren_off=True,
                SIMULATION_PARAMETERS=SIMULATION_PARAMETERS,
                SIMULATION_DATA=SIMULATION_DATA,
            )
            return select_closest_location_ID(
                (new_x, new_y), SIMULATION_PARAMETERS, SIMULATION_DATA
            )

        return self.current_location_ID

    @staticmethod
    def calculate_battery_reduction_and_distance_driving(
        source_location_ID: int,
        target_location_ID: int,
        siren_off: bool,
        SIMULATION_PARAMETERS: dict[str, Any],
        SIMULATION_DATA: dict[str, Any],
    ) -> tuple[float, float]:
        """
        Calculates the battery reduction and distance during driving.

        Parameters
        ----------
        source_location_ID : int
            The initial location (i.e., source) of the ambulance.
        target_location_ID : int
            The target location the ambulance drove to.
        siren_off : bool
            Whether the siren is on or off.
        SIMULATION_PARAMETERS : dict[str, Any]
            The simulation parameters. The parameters ``PRINT`` and
            ``DRIVING_USAGE`` are at least necessary. See ``main.py`` for
            parameter explanations.
        SIMULATION_DATA : dict[str, Any]
            The simulation data. ``DISTANCE_MATRIX`` is at least necessary. It
            is based on ``DISTANCE_FILE``. See ``main.py`` and the input data
            section on the ELASPY website for explanations.

        Returns
        -------
        battery_reduction : float
            The battery reduction during driving in kWh.
        distance_travelled : float
            The distance travelled in kilometers.

        """
        # note: siren_off is not used (as it should not be).
        distance_travelled = SIMULATION_DATA["DISTANCE_MATRIX"].loc[
            source_location_ID, target_location_ID
        ]
        if SIMULATION_PARAMETERS["PRINT"]:
            print(
                f"Traveling between {source_location_ID} and "
                f"{target_location_ID} is {distance_travelled} km and costs "
                f"{distance_travelled * SIMULATION_PARAMETERS['DRIVING_USAGE']}"
                " kWh of battery."
            )
        return (
            distance_travelled * SIMULATION_PARAMETERS["DRIVING_USAGE"],
            distance_travelled,
        )

    @staticmethod
    def calculate_battery_reduction_idling(
        idle_time: float, SIMULATION_PARAMETERS: dict[str, Any]
    ) -> float:
        """
        Calculates the battery reduction during idling.

        Parameters
        ----------
        idle_time : float
            The idle/stationary time of the ambulance.
        SIMULATION_PARAMETERS : dict[str, Any]
            The simulation parameters. The parameters ``PRINT`` and
            ``IDLE_USAGE`` are at least necessary. See ``main.py`` for
            parameter explanations.

        Returns
        -------
        float
            The battery reduction during idling/being stationary in kWh.

        """
        if SIMULATION_PARAMETERS["PRINT"]:
            print(
                f"An idle/stationary time of: {idle_time} takes "
                f"{(idle_time / 60) * SIMULATION_PARAMETERS['IDLE_USAGE']} "
                "kWh of battery."
            )
        return (idle_time / 60) * SIMULATION_PARAMETERS["IDLE_USAGE"]

    @staticmethod
    def calculate_charging_time(
        required_battery_increase: float, speed_charger: float
    ) -> float:
        """
        Calculates the charging time to reach a certain battery increase.

        Parameters
        ----------
        required_battery_increase : float
            The required battery increase in kWh.
        speed_charger : float
            The speed of the charger in kW.

        Returns
        -------
        float
            The required charging time.

        """
        return (
            required_battery_increase / speed_charger
        ) * 60  # Required charging time in minutes.

    @staticmethod
    def select_charging_station(
        charging_stations_location: dict[
            str, list[sp.resources.resource.Resource | float]
        ],
        location_ID: int,
        SIMULATION_PARAMETERS: dict[str, Any],
        SIMULATION_DATA: dict[str, Any],
    ):
        """
        Selects a charging station based on current availability.

        Parameters
        ----------
        charging_stations_location : dict[str, list[sp.resources.resource.Resource | float]]
            The charging stations resources at all locations of the same type
            (bases or hospitals) together with their charging speeds.
        location_ID : int
            The location ID.
        SIMULATION_PARAMETERS : dict[str, Any]
            The simulation parameters. The parameter ``PRINT`` is at least
            necessary. See ``main.py`` for parameter explanations.
        SIMULATION_DATA : dict[str, Any]
            The simulation data. ``nr_times_no_fast_no_regular_available`` is
            at least necessary. It represents the number of times no fast and
            regular chargers were available.

        Raises
        ------
        Exception
            If invalid input is detected.

        Returns
        -------
        sp.resources.resource.Resource
            The selected charger.
        float
            The charging speed in kW of the selected charger.

        """
        # Only regular chargers available:
        if (
            charging_stations_location[str(location_ID)][0] is np.nan
            and charging_stations_location[str(location_ID)][2] is not np.nan
        ):
            if SIMULATION_PARAMETERS["PRINT"]:
                print(
                    f"At location {location_ID} only regular chargers are "
                    "available. One is selected."
                )
            return (
                charging_stations_location[str(location_ID)][2],
                charging_stations_location[str(location_ID)][3],
            )
        # Only fast chargers available:
        elif (
            charging_stations_location[str(location_ID)][0] is not np.nan
            and charging_stations_location[str(location_ID)][2] is np.nan
        ):
            if SIMULATION_PARAMETERS["PRINT"]:
                print(
                    f"At location {location_ID} only fast chargers are "
                    "available. One is selected."
                )
            return (
                charging_stations_location[str(location_ID)][0],
                charging_stations_location[str(location_ID)][1],
            )
        # Both fast and regular chargers available:
        elif (
            charging_stations_location[str(location_ID)][0] is not np.nan
            and charging_stations_location[str(location_ID)][2] is not np.nan
        ):
            # The type: ignores are necessary below due to the type hint of
            # Resource|float of charging_stations_location. If not ignored, mypy
            # provides an error that a 'float' does not have attributes
            #'capacity' or 'users'. Albeit true, in this case positions [0] and
            # [2] correspond to the Resource objects that do have attributes
            #'capacity' and 'users'.
            if SIMULATION_PARAMETERS["PRINT"]:
                print(
                    f"At location {location_ID} both regular and fast "
                    "chargers are available. Select based on availability."
                )
                print(
                    "There are "
                    f"{len(charging_stations_location[str(location_ID)][0].users)} "  # type: ignore
                    "ambulances using a fast charger "
                    "and the capacity is "
                    f"{charging_stations_location[str(location_ID)][0].capacity}."  # type: ignore
                )
                print(
                    "There are "
                    f"{len(charging_stations_location[str(location_ID)][2].users)} "  # type: ignore
                    "ambulances using a regular charger "
                    "and the capacity is "
                    f"{charging_stations_location[str(location_ID)][2].capacity}."  # type: ignore
                )
            if (
                len(charging_stations_location[str(location_ID)][0].users)
                < charging_stations_location[str(location_ID)][0].capacity
            ):  # type: ignore
                if SIMULATION_PARAMETERS["PRINT"]:
                    print("A fast charger is available and thus selected.")
                return (
                    charging_stations_location[str(location_ID)][0],
                    charging_stations_location[str(location_ID)][1],
                )
            elif (
                len(charging_stations_location[str(location_ID)][2].users)
                < charging_stations_location[str(location_ID)][2].capacity
            ):  # type: ignore
                if SIMULATION_PARAMETERS["PRINT"]:
                    print("A regular charger is available and thus selected.")
                return (
                    charging_stations_location[str(location_ID)][2],
                    charging_stations_location[str(location_ID)][3],
                )
            else:
                if SIMULATION_PARAMETERS["PRINT"]:
                    print(
                        "No regular nor fast charger is available. "
                        "Select a fast charger."
                    )
                SIMULATION_DATA["nr_times_no_fast_no_regular_available"] += 1
                return (
                    charging_stations_location[str(location_ID)][0],
                    charging_stations_location[str(location_ID)][1],
                )
        else:
            raise Exception("Cannot select a charger. Check the input. Error")
