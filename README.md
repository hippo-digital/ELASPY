[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# ELASPY (Extended)

A forked and extended version of the [ELASPY](https://nanned.github.io/ELASPY) ([Github](https://github.com/NanneD/ELASPY)) simulator that extends the simulation with support for:

* Ambulances with variable shift patterns
* Meal Breaks for Ambulance crews during shift (including policy for starting the break)
* Multiple prioritised categories of patient
* Patient-Ambulance reassignment where a new closer ambulance becomes available during the drive-to-patient phase of the patient process

Additionally, the simulation now takes parameters in YAML format to enable easier repeatable experimentation.  An example configuration file can be found in [example-config.yaml](example-config.yaml)

## Documentation

Much of the upstream documentation on the functioning of the simulator and the parameters used is still relevant to this repo.  For additional parameters see the comments in the [example-config.yaml] file.

## License

The GNU General Public License v3 (GPL-3) license is used. For more information, please see the included LICENSE.md file.
