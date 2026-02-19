"""
This file contains the AquacropModel class that runs the simulation.

DATA ASSIMILATION EXTENSIONS:
=========================================================================
The following additions enable pause-update-resume workflows for data
assimilation (DA). They are designed so that:

  1. Normal (non-DA) runs are completely unaffected.
  2. All new functionality is accessed through a small number of clearly
     named public methods on the existing AquaCropModel class.
  3. No existing functions, modules, or file structure are modified.

New public methods for DA users:
  - run_model(..., target_date="YYYY/MM/DD")  # advance to a specific date
  - get_da_state()          -> dict   # extract key state variables
  - update_da_state(dict)   -> None   # inject updated state variables
                                      # (with consistency recalculation)

Typical DA workflow:
    model.run_model(target_date="2020/04/15")  # run to first observation date
    state = model.get_da_state()               # extract state
    # ... user applies their EO/ML update externally ...
    state['canopy_cover'] = updated_cc          # modify as needed
    state['biomass'] = updated_bio
    state['soil_moisture'] = updated_th
    state['root_depth'] = updated_z_root
    model.update_da_state(state)               # inject & reconcile
    model.run_model(target_date="2020/05/01", initialize_model = False)  # resume to next obs date
    # ... repeat as many times as needed ...
    model.run_model(till_termination=True, initialize_model = False)      # finish the season
"""
import time
import datetime
import copy
import os
import logging
import warnings
import numpy as np
from typing import Dict, Union, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    # Important: classes are only imported when types are checked, not in production.
    from pandas import DataFrame
    from aquacrop.entities.clockStruct import ClockStruct
    from aquacrop.entities.co2 import CO2
    from aquacrop.entities.crop import Crop
    from aquacrop.entities.initParamVariables import InitialCondition
    from aquacrop.entities.inititalWaterContent import InitialWaterContent
    from aquacrop.entities.paramStruct import ParamStruct
    from aquacrop.entities.soil import Soil


# pylint: disable=wrong-import-position
from .entities.co2 import CO2
from .entities.fieldManagement import FieldMngt
from .entities.groundWater import GroundWater
from .entities.irrigationManagement import IrrigationManagement
from .entities.output import Output
from .initialize.compute_variables import compute_variables
from .initialize.create_soil_profile import create_soil_profile
from .initialize.read_clocks_parameters import read_clock_parameters
from .initialize.read_field_managment import read_field_management
from .initialize.read_groundwater_table import read_groundwater_table
from .initialize.read_irrigation_management import read_irrigation_management
from .initialize.read_model_initial_conditions import read_model_initial_conditions
from .initialize.read_model_parameters import read_model_parameters
from .initialize.read_weather_inputs import read_weather_inputs
from .timestep.check_if_model_is_finished import check_model_is_finished
from .timestep.run_single_timestep import solution_single_time_step
from .timestep.update_time import update_time
from .timestep.outputs_when_model_is_finished import outputs_when_model_is_finished

class AquaCropModel:
    """
    This is the main class of the AquaCrop-OSPy model.
    It is in charge of executing all the operations.

    Parameters:

        sim_start_time (str): YYYY/MM/DD, Simulation start date

        sim_end_time (str): date YYYY/MM/DD, Simulation end date

        weather_df: daily weather data , created using prepare_weather

        soil: Soil object contains paramaters and variables of the soil
                used in the simulation

        crop: Crop object contains Paramaters and variables of the crop used
                in the simulation

        initial_water_content: Defines water content at start of simulation

        irrigation_management: Defines irrigation strategy

        field_management: Defines field management options

        fallow_field_management: Defines field management options during fallow period

        groundwater: Stores information on water table parameters

        co2_concentration: Defines CO2 concentrations

        off_season: (True) simulate off-season or (False) skip ahead to start of 
                    next growing season


    """

    # Model parameters
    __steps_are_finished: bool = False  # True if all steps of the simulation are done.
    __has_model_executed: bool = False  # Determines if the model has been run
    __has_model_finished: bool = False  # Determines if the model is finished
    __start_model_execution: float = 0.0  # Time when the execution start
    __end_model_execution: float = 0.0  # Time when the execution end
    # Attributes initialised later
    _clock_struct: "ClockStruct"
    _param_struct: "ParamStruct"
    _init_cond: "InitialCondition"
    _outputs: "Output"
    _weather: "DataFrame"
    
    # --- DA extension: previous-day state snapshot ---
    _da_state_previous_day: Optional[dict] = None

    def __init__(
        self,
        sim_start_time: str,
        sim_end_time: str,
        weather_df: "DataFrame",
        soil: "Soil",
        crop: "Crop",
        initial_water_content: "InitialWaterContent",
        irrigation_management: Optional["IrrigationManagement"] = None,
        field_management: Optional["FieldMngt"] = None,
        fallow_field_management: Optional["FieldMngt"] = None,
        groundwater: Optional["GroundWater"] = None,
        co2_concentration: Optional["CO2"] = None,
        off_season: bool=False,
    ) -> None:

        self.sim_start_time = sim_start_time
        self.sim_end_time = sim_end_time
        self.weather_df = weather_df
        self.soil = soil
        self.crop = crop
        self.initial_water_content = initial_water_content   
        self.co2_concentration = co2_concentration
        self.off_season = off_season
      
        self.irrigation_management = irrigation_management
        self.field_management = field_management
        self.fallow_field_management = fallow_field_management
        self.groundwater = groundwater

        if irrigation_management is None:
            self.irrigation_management = IrrigationManagement(irrigation_method=0)
        if field_management is None:
            self.field_management = FieldMngt()
        if fallow_field_management is None:
            self.fallow_field_management = FieldMngt()
        if groundwater is None:
            self.groundwater = GroundWater()
        if co2_concentration is None:
            self.co2_concentration = CO2()

    @property
    def sim_start_time(self) -> str:
        """
        Return sim start date
        """
        return self._sim_start_time

    @sim_start_time.setter
    def sim_start_time(self, value: str) -> None:
        """
        Check if sim start date is in a correct format.
        """

        if _sim_date_format_is_correct(value) is not False:
            self._sim_start_time = value
        else:
            raise ValueError("sim_start_time format must be 'YYYY/MM/DD'")

    @property
    def sim_end_time(self) -> str:
        """
        Return sim end date
        """
        return self._sim_end_time

    @sim_end_time.setter
    def sim_end_time(self, value: str) -> None:
        """
        Check if sim end date is in a correct format.
        """
        if _sim_date_format_is_correct(value) is not False:
            self._sim_end_time = value
        else:
            raise ValueError("sim_end_time format must be 'YYYY/MM/DD'")

    @property
    def weather_df(self) -> "DataFrame":
        """
        Return weather dataframe
        """
        return self._weather_df

    @weather_df.setter
    def weather_df(self, value: "DataFrame"):
        """
        Check if weather dataframe is in a correct format.
        """
        weather_df_columns = "Date MinTemp MaxTemp Precipitation ReferenceET".split(" ")
        if not all([column in value for column in weather_df_columns]):
            raise ValueError(
                "Error in weather_df format. Check if all the following columns exist "
                + "(Date MinTemp MaxTemp Precipitation ReferenceET)."
            )

        self._weather_df = value

    def _initialize(self) -> None:
        """
        Initialise all model variables
        """

        # Initialize ClockStruct object
        self._clock_struct = read_clock_parameters(
            self.sim_start_time, self.sim_end_time, self.off_season
        )

        # get _weather data
        self.weather_df = read_weather_inputs(self._clock_struct, self.weather_df)

        # read model params
        self._clock_struct, self._param_struct = read_model_parameters(
            self._clock_struct, self.soil, self.crop, self.weather_df
        )

        # read irrigation management
        self._param_struct = read_irrigation_management(
            self._param_struct, self.irrigation_management, self._clock_struct
        )

        # read field management
        self._param_struct = read_field_management(
            self._param_struct, self.field_management, self.fallow_field_management
        )

        # read groundwater table
        self._param_struct = read_groundwater_table(
            self._param_struct, self.groundwater, self._clock_struct
        )

        # Compute additional variables
        self._param_struct.CO2 = self.co2_concentration
        self._param_struct = compute_variables(
            self._param_struct, self.weather_df, self._clock_struct
        )

        # read, calculate inital conditions
        self._param_struct, self._init_cond = read_model_initial_conditions(
            self._param_struct, self._clock_struct, self.initial_water_content, self.crop
        )

        self._param_struct = create_soil_profile(self._param_struct)

        # Outputs results (water_flux, crop_growth, final_stats)
        self._outputs = Output(self._clock_struct.time_span, self._init_cond.th)

        # save model _weather to _init_cond
        self._weather = self.weather_df.values
        
        # DA extension: reset previous-day snapshot
        self._da_state_previous_day = None

    def run_model(
        self,
        num_steps: int = 1,
        till_termination: bool = False,
        initialize_model: bool = True,
        process_outputs: bool = False,
        target_date: Optional[str] = None,
    ) -> bool:
        """
        This function is responsible for executing the model.

        Arguments:

            num_steps: Number of steps (Days) to be executed.

            till_termination: Run the simulation to completion

            initialize_model: Whether to initialize the model \
            (i.e., go back to beginning of season)

            process_outputs: process outputs into dataframe before \
                simulation is finished
                
            target_date: (str, optional) Run the model up to (and including)
                this date in 'YYYY/MM/DD' format. When provided, num_steps
                and till_termination are ignored. The model will pause at
                end-of-day on target_date, ready for state extraction
                and/or update. Set initialize_model=False when resuming
                after a DA update.

        Returns:
            True if finished
        """

        if initialize_model:
            self._initialize()
            
        # ---------------------------------------------------------------
        # NEW: target_date mode — run until a specific calendar date
        # ---------------------------------------------------------------
        if target_date is not None:
            if _sim_date_format_is_correct(target_date) is False:
                raise ValueError("target_date format must be 'YYYY/MM/DD'")

            target_dt = datetime.datetime.strptime(target_date, "%Y/%m/%d")

            # Validate: target_date must be within the simulation window
            sim_end_dt = datetime.datetime.strptime(self.sim_end_time, "%Y/%m/%d")
            if target_dt > sim_end_dt:
                raise ValueError(
                    f"target_date ({target_date}) is after sim_end_time "
                    f"({self.sim_end_time})."
                )

            self.__start_model_execution = time.time()
            while self._clock_struct.model_is_finished is False:
                # Check if we have reached the target date
                # step_start_time is the date of the current timestep
                current_date = self._clock_struct.step_start_time
                if current_date > target_dt:
                    # We have passed the target — stop before this step
                    break

                # Snapshot state BEFORE this timestep (for "day before" record)
                self._da_state_previous_day = self._snapshot_da_state()

                (
                    self._clock_struct,
                    self._init_cond,
                    self._param_struct,
                    self._outputs,
                ) = self._perform_timestep()

                # Check if the step we just ran was the target date
                if current_date >= target_dt:
                    break

            self.__end_model_execution = time.time()
            self.__has_model_executed = True
            self.__has_model_finished = self._clock_struct.model_is_finished
            return True
        
        # ORIGINAL till termination and num_steps logic are unaffected

        if till_termination:
            self.__start_model_execution = time.time()
            while self._clock_struct.model_is_finished is False:
                (
                    self._clock_struct,
                    self._init_cond,
                    self._param_struct,
                    self._outputs,
                ) = self._perform_timestep()
            self.__end_model_execution = time.time()
            self.__has_model_executed = True
            self.__has_model_finished = True
            return True
        else:
            if num_steps < 1:
                raise ValueError("num_steps must be equal to or greater than 1.")
            self.__start_model_execution = time.time()
            for i in range(num_steps):

                if (i == range(num_steps)[-1]) and (process_outputs is True):
                    self.__steps_are_finished = True
                    
                # DA extension: keep rolling snapshot of previous day
                self._da_state_previous_day = self._snapshot_da_state()

                (
                    self._clock_struct,
                    self._init_cond,
                    self._param_struct,
                    self._outputs,
                ) = self._perform_timestep()

                if self._clock_struct.model_is_finished:
                    self.__end_model_execution = time.time()
                    self.__has_model_executed = True
                    self.__has_model_finished = True
                    return True

            self.__end_model_execution = time.time()
            self.__has_model_executed = True
            self.__has_model_finished = False
            return True

    def _perform_timestep(
        self,
    ) -> Tuple["ClockStruct", "InitialCondition", "ParamStruct", "Output"]:

        """
        Function to run a single time-step (day) calculation of AquaCrop-OS
        """

        # extract _weather data for current timestep
        weather_step = _weather_data_current_timestep(
            self._weather, self._clock_struct.time_step_counter
        )

        # Get model solution_single_time_step
        new_cond, param_struct, outputs = solution_single_time_step(
            self._init_cond,
            self._param_struct,
            self._clock_struct,
            weather_step,
            self._outputs,
        )

        # Check model termination
        clock_struct = self._clock_struct
        clock_struct.model_is_finished = check_model_is_finished(
            self._clock_struct.step_end_time,
            self._clock_struct.simulation_end_date,
            self._clock_struct.model_is_finished,
            self._clock_struct.season_counter,
            self._clock_struct.n_seasons,
            new_cond.harvest_flag,
        )

        # Update time step
        clock_struct, _init_cond, param_struct = update_time(
            clock_struct, new_cond, param_struct, self._weather, self.crop
        )

        # Create  _outputsdataframes when model is finished
        final_water_flux_growth_outputs = outputs_when_model_is_finished(
            clock_struct.model_is_finished,
            outputs.water_flux,
            outputs.water_storage,
            outputs.crop_growth,
            self.__steps_are_finished,
        )

        if final_water_flux_growth_outputs is not False:
            (
                outputs.water_flux,
                outputs.water_storage,
                outputs.crop_growth,
            ) = final_water_flux_growth_outputs

        return clock_struct, _init_cond, param_struct, outputs
    
    # ===================================================================
    #  DATA ASSIMILATION METHODS  (new — no changes to existing methods)
    # ===================================================================

    def _snapshot_da_state(self) -> dict:
        """
        Take a lightweight snapshot of DA-relevant state variables.
        Used internally to record the "day before" values.

        Returns:
            dict with copies of key state variables
        """
        ic = self._init_cond
        return {
            "canopy_cover": float(ic.canopy_cover),
            "canopy_cover_adj": float(ic.canopy_cover_adj),
            "canopy_cover_ns": float(ic.canopy_cover_ns),
            "biomass": float(ic.biomass),
            "biomass_ns": float(ic.biomass_ns),
            "root_depth": float(ic.z_root),
            "soil_moisture": np.copy(ic.th),  # per-compartment volumetric WC
            # Additional context variables (read-only, for user reference)
            "harvest_index": float(ic.harvest_index),
            "harvest_index_adj": float(ic.harvest_index_adj),
            "ccx_w": float(ic.ccx_w),
            "ccx_act": float(ic.ccx_act),
            "ccx_w_ns": float(ic.ccx_w_ns),
            "ccx_act_ns": float(ic.ccx_act_ns),
        }

    def get_da_state(self) -> dict:
        """
        Extract current internal state variables for data assimilation.

        This should be called AFTER run_model(..., target_date=...) has
        paused the simulation. It returns the values for both the current
        day (end of the last simulated timestep) and the previous day,
        enabling users to assess the trajectory of change.

        Returns:
            dict with structure:
            {
                "current_date": str,       # "YYYY/MM/DD" of last completed step
                "current": {
                    "canopy_cover": float,      # Green canopy cover (0-1 fraction)
                    "canopy_cover_adj": float,  # Adjusted CC accounting for stress
                    "canopy_cover_ns": float,   # CC under no-stress conditions
                    "biomass": float,           # Above-ground biomass (tonnes/ha)
                    "biomass_ns": float,        # Biomass under no-stress conditions
                    "root_depth": float,        # Effective rooting depth (m)
                    "soil_moisture": np.array,  # Volumetric WC per soil compartment
                    "harvest_index": float,     # Current harvest index
                    "harvest_index_adj": float, # Adjusted HI
                    "ccx_w": float,             # Max CC reached (with stress)
                    "ccx_act": float,           # Actual max CC for season
                    "ccx_w_ns": float,          # Max CC (no stress)
                    "ccx_act_ns": float,        # Actual max CC (no stress)
                },
                "previous_day": {             # Same structure, or None if N/A
                    ...
                }
            }

        Raises:
            ValueError: If model has not been run yet.
        """
        if not self.__has_model_executed:
            raise ValueError(
                "Cannot extract state before running the model. "
                "Call run_model() first."
            )

        # Determine the date of the last completed timestep.
        # After a timestep, the clock has already advanced, so the last
        # simulated day is step_start_time minus 1 day (or use the
        # time_step_counter to look it up).
        # The most reliable approach: step_start_time is the NEXT day to
        # simulate, so the last completed day is one day before.
        last_completed = (
            self._clock_struct.step_start_time - datetime.timedelta(days=1)
        )
        current_date_str = last_completed.strftime("%Y/%m/%d")

        current_state = self._snapshot_da_state()

        return {
            "current_date": current_date_str,
            "current": current_state,
            "previous_day": copy.deepcopy(self._da_state_previous_day),
        }

    def update_da_state(
        self,
        updated_state: dict,
    ) -> None:
        """
        Update internal state variables with data-assimilation corrections,
        then recalculate all dependent coefficients to ensure model
        consistency going forward.

        This should be called AFTER get_da_state() and BEFORE the next
        call to run_model(..., initialize_model=False).

        Arguments:
            updated_state: dict containing one or more of the following keys.
                Only keys present will be updated; others are left unchanged.

                "canopy_cover": float   # Green canopy cover (0-1 fraction)
                "biomass": float        # Above-ground biomass (tonnes/ha)
                "root_depth": float     # Effective rooting depth (m)
                "soil_moisture": array  # Volumetric WC per soil compartment

        Raises:
            ValueError: If model has not been run, or if values are outside
                physically plausible bounds.

        Notes on consistency recalculations performed:
        -----------------------------------------------
        When state variables are externally modified, several derived
        quantities in the model's InitialCondition (NewCond) object must
        be brought into agreement. The following adjustments are made:

        CANOPY COVER update triggers:
          - CCadj (adjusted CC) is set to the new CC value
          - CCxAct (actual max CC achieved) is updated if new CC > current CCxAct
          - CCxW (max CC with water stress) is updated similarly
          - If CC was reduced below CCxAct, the model interprets this as
            stress-induced decline and sets CC_NS proportionally
          - Crop transpiration coefficient (Kcb) will be recalculated on the
            next timestep automatically since it depends on CC

        BIOMASS update triggers:
          - biomass_ns (no-stress biomass) is scaled proportionally to maintain
            the ratio between stressed and non-stressed biomass

        ROOT DEPTH update triggers:
          - Clamped to [Zmin, Zmax] for the crop
          - Root zone water content (Wr) and depletion (Dr) are recalculated
            from the soil moisture profile over the new root zone extent
          - TAW (total available water) is recalculated
          - All water stress coefficients (Ks for expansion, stomatal closure,
            senescence, aeration) will be recalculated on the next timestep

        SOIL MOISTURE update triggers:
          - Each compartment is clamped to [wilting point, saturation]
          - Root zone depletion (Dr) and total water in root zone (Wr) are
            recalculated
          - All water stress coefficients will be recalculated automatically
            on the next timestep since they depend on Dr/TAW
        """
        if not self.__has_model_executed:
            raise ValueError(
                "Cannot update state before running the model. "
                "Call run_model() first."
            )

        ic = self._init_cond
        crop = self._param_struct.Seasonal_Crop_List[
            self._clock_struct.season_counter
        ]
        soil_profile = self._param_struct.Soil.Profile

        # --- 1. SOIL MOISTURE UPDATE ---
        if "soil_moisture" in updated_state:
            new_th = np.array(updated_state["soil_moisture"], dtype=np.float64)

            if len(new_th) != len(ic.th):
                raise ValueError(
                    f"soil_moisture must have {len(ic.th)} compartments, "
                    f"got {len(new_th)}."
                )

            # Clamp each compartment to [th_wp, th_s]
            for i in range(len(new_th)):
                th_wp = soil_profile.th_wp[i]   # wilting point
                th_s = soil_profile.th_s[i]     # saturation
                if new_th[i] < th_wp:
                    warnings.warn(
                        f"Soil moisture in compartment {i} ({new_th[i]:.4f}) "
                        f"is below wilting point ({th_wp:.4f}). "
                        f"Clamping to wilting point."
                    )
                    new_th[i] = th_wp
                if new_th[i] > th_s:
                    warnings.warn(
                        f"Soil moisture in compartment {i} ({new_th[i]:.4f}) "
                        f"exceeds saturation ({th_s:.4f}). "
                        f"Clamping to saturation."
                    )
                    new_th[i] = th_s

            ic.th = new_th

        # --- 2. ROOT DEPTH UPDATE ---
        if "root_depth" in updated_state:
            new_z_root = float(updated_state["root_depth"])

            # Clamp to crop limits
            z_min = crop.Zmin
            z_max = crop.Zmax
            if new_z_root < z_min:
                warnings.warn(
                    f"root_depth ({new_z_root:.4f}) is below minimum "
                    f"({z_min:.4f}). Clamping to minimum."
                )
                new_z_root = z_min
            if new_z_root > z_max:
                warnings.warn(
                    f"root_depth ({new_z_root:.4f}) exceeds maximum "
                    f"({z_max:.4f}). Clamping to maximum."
                )
                new_z_root = z_max

            ic.z_root = new_z_root

        # --- 3. CANOPY COVER UPDATE ---
        if "canopy_cover" in updated_state:
            new_cc = float(updated_state["canopy_cover"])

            # Clamp to [0, 1]
            if new_cc < 0:
                warnings.warn("canopy_cover < 0. Clamping to 0.")
                new_cc = 0.0
            if new_cc > 1:
                warnings.warn("canopy_cover > 1. Clamping to 1.0.")
                new_cc = 1.0

            old_cc = ic.canopy_cover

            # Update CC and adjusted CC
            ic.canopy_cover = new_cc
            ic.canopy_cover_adj = new_cc

            # Update max-CC trackers
            if new_cc > ic.ccx_act:
                ic.ccx_act = new_cc
            if new_cc > ic.ccx_w:
                ic.ccx_w = new_cc

            # Scale the no-stress canopy cover proportionally.
            # This preserves the relationship between stressed and
            # non-stressed trajectories. If the user is increasing CC
            # beyond what was simulated, we also scale CC_NS up
            # (but never above CCx from crop parameters).
            if old_cc > 0:
                scale_factor = new_cc / old_cc
                new_cc_ns = ic.canopy_cover_ns * scale_factor
                new_cc_ns = min(new_cc_ns, crop.CCx)
                new_cc_ns = max(new_cc_ns, new_cc)  # NS should be >= stressed
                ic.canopy_cover_ns = new_cc_ns
            else:
                # If old CC was 0 but new is >0, set NS = CC
                ic.canopy_cover_ns = new_cc

            # Update NS max-CC trackers
            if ic.canopy_cover_ns > ic.ccx_act_ns:
                ic.ccx_act_ns = ic.canopy_cover_ns
            if ic.canopy_cover_ns > ic.ccx_w_ns:
                ic.ccx_w_ns = ic.canopy_cover_ns

        # --- 4. BIOMASS UPDATE ---
        if "biomass" in updated_state:
            new_bio = float(updated_state["biomass"])

            if new_bio < 0:
                warnings.warn("biomass < 0. Clamping to 0.")
                new_bio = 0.0

            old_bio = ic.biomass

            # Scale non-stress biomass proportionally
            if old_bio > 0:
                scale_factor = new_bio / old_bio
                ic.biomass_ns = ic.biomass_ns * scale_factor
            else:
                ic.biomass_ns = new_bio

            ic.biomass = new_bio

        # --- 5. RECALCULATE ROOT ZONE WATER BALANCE ---
        # After any soil moisture or root depth change, recompute
        # root zone water content (Wr), depletion (Dr), and TAW.
        # These are the variables that drive all water stress
        # coefficients on the next timestep.
        if "soil_moisture" in updated_state or "root_depth" in updated_state:
            self._recalculate_root_zone_water()

        # Store snapshot for potential next DA cycle
        self._da_state_previous_day = self._snapshot_da_state()

    def _recalculate_root_zone_water(self) -> None:
        """
        Recalculate root zone water content, depletion, and TAW
        from the current soil moisture profile and root depth.

        This ensures that water stress coefficients (Ks_exp, Ks_sto,
        Ks_sen, Ks_aer) are correctly computed on the next timestep.

        The logic mirrors the root_zone_water() function in
        aquacrop/solution/solution_root_zone_water.py but is applied
        here as a consistency adjustment after DA state injection.
        """
        ic = self._init_cond
        soil_profile = self._param_struct.Soil.Profile
        crop = self._param_struct.Seasonal_Crop_List[
            self._clock_struct.season_counter
        ]

        z_root = ic.z_root
        n_comp = len(ic.th)
        dz = soil_profile.dz  # thickness of each compartment (array)

        # Compute water stored in root zone and at field capacity / WP
        wr = 0.0      # actual water in root zone (mm)
        wr_fc = 0.0   # water at field capacity in root zone (mm)
        wr_wp = 0.0   # water at wilting point in root zone (mm)
        wr_s = 0.0    # water at saturation in root zone (mm)

        depth_from_surface = 0.0
        for i in range(n_comp):
            comp_thick = dz[i]  # thickness of compartment i (m)
            depth_from_surface += comp_thick

            if depth_from_surface <= z_root:
                # Entire compartment is within root zone
                factor = 1.0
            elif (depth_from_surface - comp_thick) < z_root:
                # Compartment straddles root zone boundary
                factor = (z_root - (depth_from_surface - comp_thick)) / comp_thick
            else:
                # Below root zone
                break

            wr += ic.th[i] * 1000 * comp_thick * factor
            wr_fc += soil_profile.th_fc[i] * 1000 * comp_thick * factor
            wr_wp += soil_profile.th_wp[i] * 1000 * comp_thick * factor
            wr_s += soil_profile.th_s[i] * 1000 * comp_thick * factor

        # Total available water and current depletion
        taw = wr_fc - wr_wp  # Total Available Water (mm)
        dr = wr_fc - wr       # Root zone depletion (mm)

        # Clamp depletion: can't be negative (wetter than FC) or > TAW
        if dr < 0:
            dr = 0.0
        if taw > 0 and dr > taw:
            dr = taw

        # Update the InitialCondition object
        ic.depletion = dr
        ic.taw = taw

        # Update surface layer water content for soil evaporation
        # (the first compartment's water content drives evaporation calcs,
        # which is already updated via th, so no extra action needed)

    # ===================================================================
    #  END OF DATA ASSIMILATION METHODS
    # ===================================================================

    def get_simulation_results(self):
        """
        Return all the simulation results
        """
        if self.__has_model_executed:
            if self.__has_model_finished:
                return self._outputs.final_stats
            else:
                return False  # If the model is not finished, the results are not generated.
        else:
            raise ValueError(
                "You cannot get results without running the model. "
                + "Please execute the run_model() method."
            )

    def get_water_storage(self):
        """
        Return water storage in soil results
        """
        if self.__has_model_executed:
            return self._outputs.water_storage
        else:
            raise ValueError(
                "You cannot get results without running the model. "
                + "Please execute the run_model() method."
            )

    def get_water_flux(self):
        """
        Return water flux results
        """
        if self.__has_model_executed:
            return self._outputs.water_flux
        else:
            raise ValueError(
                "You cannot get results without running the model. "
                + "Please execute the run_model() method."
            )

    def get_crop_growth(self):
        """
        Return crop growth results
        """
        if self.__has_model_executed:
            return self._outputs.crop_growth
        else:
            raise ValueError(
                "You cannot get results without running the model. "
                + "Please execute the run_model() method."
            )

    def get_additional_information(self) -> Dict[str, Union[bool, float]]:
        """
        Additional model information.

        Returns:
            dict: {has_model_finished,execution_time}

        """
        if self.__has_model_executed:
            return {
                "has_model_finished": self.__has_model_finished,
                "execution_time": self.__end_model_execution
                - self.__start_model_execution,
            }
        else:
            raise ValueError(
                "You cannot get results without running the model. "
                + "Please execute the run_model() method."
            )


def check_iwc_soil_match(iwc_layers: int, soil_layers: int) -> bool:
    """
    This function checks if the number of soil layers is equivalent between the user-specified soil profile and initial water content.
    
    Arguments:
        iwc_layers
        soil_layers
        
    Return:
        boolean: True if number of layers match
    
    """
    if(iwc_layers == soil_layers):
        return True
    else:
        return False
        



def _sim_date_format_is_correct(date: str) -> bool:
    """
    This function checks if the start or end date of the simulation is in the correct format.

    Arguments:
        date

    Return:
        boolean: True if the date is correct.
    """
    format_dates_string = "%Y/%m/%d"
    try:
        datetime.datetime.strptime(date, format_dates_string)
        return True
    except ValueError:
        return False


def _weather_data_current_timestep(_weather, time_step_counter):
    """
    Extract _weather data for current timestep
    """
    return _weather[time_step_counter]
