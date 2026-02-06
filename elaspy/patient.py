#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class Patient:
    """
    A class to represent a patient.

    Attributes
    ----------
    patient_ID : int
        The patient ID.
    arrival_time : float
        The arrival time of the patient.
    patient_location_ID : int
        The arrival location of the patient.
    hospital_location_ID : int
        The assigned hospital (in case the patient needs to be transported).
    category: int
        The category assigned to the call
    """

    def __init__(
        self,
        ID: int,
        arrival_time: float,
        location_ID: int,
        hospital_ID: int,
        category: int,
        to_hospital: bool,
    ) -> None:
        """
        Initializes a patient.

        Parameters
        ----------
        ID : int
            The patient ID.
        arrival_time : float
            The arrival time of the patient.
        location_ID : int
            The arrival location of the patient.
        hospital_ID : int
            The assigned hospital (in case the patient needs to be transported).
        """
        self.patient_ID = ID
        self.arrival_time = arrival_time
        self.patient_location_ID = location_ID
        self.hospital_location_ID = hospital_ID
        self.category = category
        self.to_hospital = to_hospital

    def __repr__(self) -> str:
        return f"Patient {self.patient_ID} (Cat{self.category})"
