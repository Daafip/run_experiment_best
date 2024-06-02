import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
from pathlib import Path
import pandas as pd
from datetime import datetime
from datetime import timedelta
import xarray as xr
from tqdm import tqdm
import gc
from typing import Any
from ewatercycle.forcing import sources
from ewatercycle_DA import DA
from pydantic import BaseModel

param_names = ["Imax", "Ce", "Sumax", "Beta", "Pmax", "Tlag", "Kf", "Ks", "FM"]
stor_names = ["Si", "Su", "Sf", "Ss", "Sp"]


class Experiment(BaseModel):
    """Runs DA experiment"""

    n_particles: int
    storage_parameter_bounds: tuple
    s_0: Any
    model_name: str
    HRU_id: str
    sigma_tuple: tuple
    assimilate_window: int
    experiment_start_date: str
    experiment_end_date: str
    alpha: float
    save: bool
    f_n_particles: float

    units: dict = {}
    lst_camels_forcing: list = []
    time: list = []
    lst_N_eff: list = []
    lst_n_resample_indexes: list = []

    ensemble: Any = None
    paths: tuple[Path, ...] | None = None
    ref_model: Any = None
    ds_obs_dir: Path | None = None
    state_vector_arr: Any = None

    @staticmethod
    def H(Z):
        """Operator function extracts observable state from the state vector"""
        len_Z = 15
        if len(Z) == len_Z:
            return Z[-1]
        else:
            raise SyntaxWarning(f"Length of statevector should be {len_Z} but is {len(Z)}")

    @staticmethod
    def calc_NSE(Qo, Qm):
        Qo[Qo == 0] = 1e-8
        Qm[Qm == 0] = 1e-8
        QoAv = np.mean(Qo)
        ErrUp = np.sum((Qo - Qm) ** 2)
        ErrDo = np.sum((Qo - QoAv) ** 2)
        return 1 - (ErrUp / ErrDo)

    @staticmethod
    def calc_log_NSE(Qo, Qm):
        Qo[Qo == 0] = 1e-8
        Qm[Qm == 0] = 1e-8
        QoAv = np.mean(Qo)
        ErrUp = np.sum((np.log(Qo) - np.log(Qm)) ** 2)
        ErrDo = np.sum((np.log(Qo) - np.log(QoAv)) ** 2)
        return 1 - (ErrUp / ErrDo)

    def set_up_forcing(self):
        self.make_paths()
        forcing_path = self.paths[0]

        ### no longer needed likely
        # origin_camels_path = forcing_path / f'{self.HRU_id}_lump_cida_forcing_leap.txt'
        # lst_camels_file_path = []
        # for n in range(self.n_particles):
        #     camels_file_path = forcing_path / f'{self.HRU_id}_lump_cida_forcing_leap_{n}.txt'
        #     if not camels_file_path.is_file():
        #         shutil.copy(origin_camels_path, camels_file_path)
        #     lst_camels_file_path.append(camels_file_path.name)

        for n in range(self.n_particles):
            self.lst_camels_forcing.append(
                sources.HBVForcing(start_time=self.experiment_start_date,
                                   end_time=self.experiment_end_date,
                                   directory=forcing_path,
                                   camels_file=f'{self.HRU_id}_lump_cida_forcing_leap.txt',  # lst_camels_file_path[n]
                                   alpha=self.alpha,
                                   ))
        # del lst_camels_file_path
        # gc.collect()

    def make_paths(self):
        path = Path.cwd().parent
        forcing_path = path / "Forcing"
        observations_path = path / "Observations"
        output_path = path / "Output"

        self.paths = forcing_path, output_path, observations_path

        for path_i in list(self.paths):
            path_i.mkdir(exist_ok=True)

    def initialize(self):
        """Contains actual code to run the experiment"""

        p_min_initial, p_max_initial, s_max_initial, s_min_initial = self.storage_parameter_bounds

        # set up ensemble
        self.ensemble = DA.Ensemble(N=self.n_particles)
        self.ensemble.setup()

        # initial values
        array_random_num = np.array(
            [[np.random.random() for _ in range(len(p_max_initial))] for _ in range(self.n_particles)])
        p_initial = p_min_initial + array_random_num * (p_max_initial - p_min_initial)

        # values which you
        setup_kwargs_lst = []
        for index in range(self.n_particles):
            setup_kwargs_lst.append({'parameters': ','.join([str(p) for p in p_initial[index]]),
                                     'initial_storage': ','.join([str(s) for s in self.s_0]),
                                     })

        # this initializes the models for all ensemble members.
        self.ensemble.initialize(model_name=[self.model_name] * self.n_particles,
                                 forcing=self.lst_camels_forcing,
                                 setup_kwargs=setup_kwargs_lst)

        del setup_kwargs_lst, p_initial, array_random_num
        gc.collect()

    def load_obs(self):
        forcing_path, output_path, observations_path = self.paths
        # create a reference model
        self.ref_model = self.ensemble.ensemble_list[0].model

        # load observations
        self.ds_obs_dir = observations_path / f'{self.HRU_id}_streamflow_qc.nc'
        if not self.ds_obs_dir.exists():
            ds = xr.open_dataset(forcing_path / self.ref_model.forcing.pr)
            basin_area = ds.attrs['area basin(m^2)']
            ds.close()

            observations = observations_path / f'{self.HRU_id}_streamflow_qc.txt'
            cubic_ft_to_cubic_m = 0.0283168466

            new_header = ['GAGEID', 'Year', 'Month', 'Day', 'Streamflow(cubic feet per second)', 'QC_flag']
            new_header_dict = dict(list(zip(range(len(new_header)), new_header)))
            df_q = pd.read_fwf(observations, delimiter=' ', encoding='utf-8', header=None)
            df_q = df_q.rename(columns=new_header_dict)
            df_q['Streamflow(cubic feet per second)'] = df_q['Streamflow(cubic feet per second)'].apply(
                lambda x: np.nan if x == -999.00 else x)
            df_q['Q (m3/s)'] = df_q['Streamflow(cubic feet per second)'] * cubic_ft_to_cubic_m
            df_q['Q'] = df_q['Q (m3/s)'] / basin_area * 3600 * 24 * 1000  # m3/s -> m/s ->m/d -> mm/d
            df_q.index = df_q.apply(lambda x: pd.Timestamp(f'{int(x.Year)}-{int(x.Month)}-{int(x.Day)}'), axis=1)
            df_q.index.name = "time"
            df_q.drop(columns=['Year', 'Month', 'Day', 'Streamflow(cubic feet per second)'], inplace=True)
            df_q = df_q.dropna(axis=0)

            ds_obs = xr.Dataset(data_vars=df_q[['Q']])
            ds_obs.to_netcdf(self.ds_obs_dir)
            ds_obs.close()
            del df_q, ds, ds_obs
            gc.collect()

    def initialize_da_method(self):
        # set up hyperparameters
        sigma_pp, sigma_ps, sigma_w, sigma_p_Sf = self.sigma_tuple
        p_min_initial, p_max_initial, s_max_initial, s_min_initial = self.storage_parameter_bounds
        # "Imax", "Ce", "Sumax", "Beta", "Pmax", "Tlag", "Kf", "Ks", "FM" "Si", "Su", "Sf", "Ss", "Sp + Q
        # p_mean = (p_max_initial + p_min_initial) / 2
        p_sig = np.sqrt((p_max_initial - p_min_initial) ** 2 / 12)
        s_sig = np.sqrt((s_max_initial - s_min_initial) ** 2 / 12)

        lst_like_sigma = (list(sigma_p_Sf * p_sig) +  # parameters
                          list(sigma_p_Sf * s_sig) +  # states
                          [0])  # Q
        hyper_parameters = {'like_sigma_weights': sigma_w,
                            'like_sigma_state_vector': lst_like_sigma,
                            'f_n_particles': self.f_n_particles,

                            }
        print(f'init_da', end=" ")

        self.ensemble.initialize_da_method(ensemble_method_name="PF",
                                           hyper_parameters=hyper_parameters,
                                           state_vector_variables="all",
                                           # the next three are keyword arguments but are needed.
                                           observation_path=self.ds_obs_dir,
                                           observed_variable_name="Q",
                                           measurement_operator=self.H,
                                           )
        # extract units for later
        state_vector_variables = self.ensemble.ensemble_list[0].variable_names

        for var in state_vector_variables:
            self.units.update({var: self.ref_model.bmi.get_var_units(var)})

    def assimilate(self):
        ## run!
        n_timesteps = int((self.ref_model.end_time - self.ref_model.start_time) /
                          self.ref_model.time_step)

        lst_state_vector = []
        try:
            for i in tqdm(range(n_timesteps)):
                self.time.append(pd.Timestamp(self.ref_model.time_as_datetime.date()))
                # update every 3 steps
                if i % self.assimilate_window == 0:
                    assimilate = True
                else:
                    assimilate = False

                # obs = self.ensemble.ensemble_method.obs
                current_time = np.datetime64(self.ensemble.ensemble_list[0].model.time_as_datetime)
                obs = self.ensemble.observations.sel(time=current_time, method="nearest").values

                self.ensemble.update(assimilate=assimilate)

                state_vector = self.ensemble.get_state_vector()
                sv_min = state_vector.T.min(axis=1)
                sv_max = state_vector.T.max(axis=1)
                sv_mean = state_vector.T.mean(axis=1)

                state_q = state_vector[:,-1]
                diff = abs(obs - state_q)
                index_best_fit = diff.argmin()
                sv_best_fit = state_vector[index_best_fit,:]

                summarised_state_vector = np.array([sv_min, sv_max, sv_mean, sv_best_fit])
                lst_state_vector.append(summarised_state_vector)

                del state_vector, sv_min, sv_max, sv_mean, summarised_state_vector
                gc.collect()

                self.lst_N_eff.append(self.ensemble.ensemble_method.N_eff)
                if self.ensemble.ensemble_method.resample:
                    self.lst_n_resample_indexes.append(
                        len(set(self.ensemble.ensemble_method.resample_indices)))

                else:
                    self.lst_n_resample_indexes.append(np.nan)
        except KeyboardInterrupt:  # saves deleting N folders if quit manually
            self.ensemble.finalize()

        self.ensemble.finalize()

        self.state_vector_arr = np.array(lst_state_vector)
        del lst_state_vector, self.ensemble
        gc.collect()

    def create_combined_ds(self):
        data_vars = {}
        for i, name in enumerate(param_names + stor_names + ["Q"]):
            storage_terms_i = xr.DataArray(self.state_vector_arr[:, :, i].T,
                                           name=name,
                                           dims=["summary_stat", "time"],
                                           coords=[['min', 'max', 'mean','best'],
                                                   self.time],
                                           attrs={
                                               "title": f"HBV storage terms data over time for {self.n_particles} particles ",
                                               "history": f"Storage term results from ewatercycle_HBV.model",
                                               "description": "Modeled values",
                                               "units": f"{self.units[name]}"})
            data_vars[name] = storage_terms_i

        sigma_pp, sigma_ps, sigma_w, sigma_p_Sf = self.sigma_tuple
        p_min_initial, p_max_initial, s_max_initial, s_min_initial = self.storage_parameter_bounds
        ds_combined = xr.Dataset(data_vars,
                                 attrs={
                                     "title": f"HBV storage & parameter terms data over time for {self.n_particles} particles ",
                                     "history": f"Storage term results from ewatercycle_HBV.model",
                                     "sigma_pp": sigma_pp,
                                     "sigma_ps": sigma_ps,
                                     "sigma_w": sigma_w,
                                     "sigma_p_Sf": sigma_p_Sf,
                                     "assimilate_window": self.assimilate_window,
                                     "n_particles": self.n_particles,
                                     "HRU_id": self.HRU_id,
                                     "p_min_initial":p_min_initial,
                                     "p_max_initial":p_max_initial,
                                     "s_max_initial":s_max_initial,
                                     "s_min_initial":s_min_initial
                                 }
                                 )

        ds_obs = xr.open_dataset(self.ds_obs_dir)
        ds_observations = ds_obs['Q'].sel(time=self.time)
        ds_obs.close()
        ds_combined['Q_obs'] = ds_observations
        ds_combined['Q_obs'].attrs.update({
            'history': 'USGS streamflow data obtained from CAMELS dataset',
            'url': 'https://dx.doi.org/10.5065/D6MW2F4D'})

        df_n_eff = pd.DataFrame(index=self.time,
                                data=self.lst_N_eff,
                                columns=['Neff'])
        df_n_eff.index.name = 'time'
        ds_combined['Neff'] = df_n_eff['Neff']
        ds_combined['Neff'].attrs.update({
            'info': 'DA debug: 1/sum(weights^2): measure for effective ensemble size'})

        df_n_resample = pd.DataFrame(index=self.time,
                                     data=self.lst_n_resample_indexes,
                                     columns=['n_resample'])
        df_n_resample.index.name = 'time'
        ds_combined['n_resample'] = df_n_resample['n_resample']
        ds_combined['n_resample'].attrs.update({
            'info': 'DA debug: number of uniquely resampled particles'})

        current_time = str(datetime.now())[:-10].replace(":", "_")
        sigma_pp, sigma_ps, sigma_w, sigma_p_Sf = self.sigma_tuple
        if self.save:
            forcing_path, output_path, observations_path = self.paths
            file_dir = output_path / (
                f'{self.HRU_id}_psf-{sigma_p_Sf}'
                f'_w-{sigma_w}_N-{self.n_particles}_'
                f'{current_time}.nc')
            ds_combined.to_netcdf(file_dir)


        del (self.time, ds_obs, self.lst_n_resample_indexes,
             self.lst_N_eff, df_n_eff, df_n_resample)

        gc.collect()

        return ds_combined

    def finalize(self):
        forcing_path = self.paths[0]
        # remove temp file once run - in case of camels just one file
        for index, forcing in enumerate(self.lst_camels_forcing):
            forcing_file = forcing_path / forcing.pr
            forcing_file.unlink(missing_ok=True)

        # catch to remove forcing objects if not already.
        try:
            self.ensemble.finalize()
            del self.ensemble.finalize
        except AttributeError:
            pass  # already deleted previously


def run_experiment(HRU_id_int: Any,
                   storage_parameter_bounds: tuple,
                   experiment_start_date: str,
                   experiment_end_date: str,
                   spin_up:bool,
                   sigma_w: float,
                   sigma_p_Sf: float,
                   ) -> xr.Dataset | None:
    # """Contains iterables for experiment"""

    model_name = "HBVLocal"

    s_0 = np.array([0, 100, 0, 5, 0])
    n_particles = 500

    # Hyper parameters
    sigma_pp = 0
    sigma_ps = 0
    assimilate_window = 3  # after how many time steps to run the assimilate steps
    f_n_particles = 0.975

    alpha = 1.26
    print_ending = "\n"  # should be \n to show in promt, can be removed here by changing to ' '
    returned = None

    HRU_id = f'{HRU_id_int}'
    if len(HRU_id) < 8:
        HRU_id = '0' + HRU_id

    if spin_up:
        save = False
    else:
        save = True

    current_time = str(datetime.now())[:-10].replace(":", "_")
    sigma_tuple = sigma_pp, sigma_ps, sigma_w, sigma_p_Sf
    experiment = Experiment(n_particles=n_particles,
                            storage_parameter_bounds=storage_parameter_bounds,
                            s_0=s_0,
                            model_name=model_name,
                            HRU_id=HRU_id,
                            sigma_tuple=sigma_tuple,
                            assimilate_window=assimilate_window,
                            experiment_start_date=experiment_start_date,
                            experiment_end_date=experiment_end_date,
                            alpha=alpha,
                            save=save,
                            f_n_particles=f_n_particles)
    try:
        if spin_up:
            print(f'starting spinup of {HRU_id} w{sigma_w}s{sigma_p_Sf}at {current_time}', end="\n")
        else:
            print(f'starting {HRU_id} w{sigma_w}s{sigma_p_Sf}at {current_time}', end="\n")
        experiment.set_up_forcing()

        print(f'init ', end=print_ending)
        experiment.initialize()

        print(f'load obs ', end=print_ending)
        experiment.load_obs()

        print(f'init da ', end=print_ending)
        experiment.initialize_da_method()

        print(f'assimilate ', end=print_ending)
        experiment.assimilate()

        print(f'output ', end=print_ending)
        ds_combined = experiment.create_combined_ds()

        if spin_up:
            returned = ds_combined
        else:
            ds_combined.close()
            del ds_combined
            gc.collect()

    except Exception as e:
        print(e)

    finally:
        print(f'cleanup ', end=print_ending)
        experiment.finalize()

    del experiment

    gc.collect()

    return returned



"""
Check list for a new experiment:

    - All values passed correctly from main -> run ->  experiment_run
        preferably don't change the actual values passed experiment_run:
        if needed refactor to list/tuple
    - preferably keep iterable set in main
    - add iterable in attrs 
    - Meaningful file path

"""


def main():
    """Main script"""
    forcing_path = Path.cwd() / "Forcing"
    #HRU_ids = [path.name[1:8] for path in
    #           forcing_path.glob("*_lump_cida_forcing_leap.txt")]
    HRU_ids =  [7083000,
 7068000,
 4015330,
 1411300,
 7145700,
 7196900,
 6903400,
 7208500,
 5584500,
 6803510,
 7067000]
    n_start_skip = 0
    n_end_skip = 0
    sigma_w = 2
    sigma_p_Sf = 1e-3 # in the report this is epsilon_p

    # total_nruns = len(HRU_ids) * len(sigma_w_lst) * len(sigma_p_Sf_lst)
    total_nruns = len(HRU_ids) - n_start_skip - n_end_skip 
    avg_run_length = 0.3  # hr
    total_hrs = total_nruns * avg_run_length
    estimated_finish = datetime.now() + timedelta(hours=total_hrs)
    print(
        f'based on {total_nruns}run @ {avg_run_length}hrs/run = est.finish: {estimated_finish.strftime("%Y-%m-%d %H:%M")}')

    for index, HRU_id_int in enumerate(HRU_ids):
        if index < n_start_skip or index > (len(HRU_ids)-n_end_skip):
            pass
        else:
            # initial guess
            try:
                p_min_initial_spinup = np.array([0, 0.2, 40, .5, .001, 1, .01, .0001, 6])
                p_max_initial_spinup = np.array([8, 1, 800, 4, .3, 10, .1, .01, 0.1])

                s_max_initial_spinup = np.array([10, 250, 100, 40, 150])
                s_min_initial_spinup = np.array([0, 150, 0, 0, 0])

                storage_parameter_bounds_spinup = (p_min_initial_spinup,
                                                   p_max_initial_spinup,
                                                   s_max_initial_spinup,
                                                   s_min_initial_spinup)
                experiment_start_date = "1997-08-01T00:00:00Z"
                experiment_end_date = "1999-09-01T00:00:00Z"

                spin_up = True
                ds_spinup = run_experiment(HRU_id_int,
                                           storage_parameter_bounds_spinup,
                                           experiment_start_date,
                                           experiment_end_date,
                                           spin_up,
                                           sigma_w,
                                           sigma_p_Sf,
                                          )
                ds_cropped = ds_spinup.sel(time=ds_spinup.time[90:]) # 3 months of spinup time

                # use the result to get a ballpark max/min
                param_names = ["Imax", "Ce", "Sumax", "Beta", "Pmax", "Tlag", "Kf", "Ks", "FM"]
                p_min_initial = np.zeros(len(param_names))
                p_max_initial = np.zeros(len(param_names))
                for index, param in enumerate(param_names):
                    p_min_initial[index] = ds_cropped[param].sel(summary_stat='min').mean().to_numpy()
                    p_max_initial[index] = ds_cropped[param].sel(summary_stat='max').mean().to_numpy()

                stor_names = ["Si", "Su", "Sf", "Ss", "Sp"]
                s_min_initial = np.zeros(len(stor_names))
                s_max_initial = np.zeros(len(stor_names))
                for index, stor in enumerate(stor_names):
                    s_min_initial[index] = ds_cropped[stor].sel(summary_stat='min').mean().to_numpy()
                    s_max_initial[index] = ds_cropped[stor].sel(summary_stat='max').mean().to_numpy()

                storage_parameter_bounds = (p_min_initial,
                                            p_max_initial,
                                            s_max_initial,
                                            s_min_initial)

                # Run longer
                experiment_start_date = "1997-08-01T00:00:00Z"
                experiment_end_date = "2002-09-01T00:00:00Z"

                spin_up = False
                run_experiment(HRU_id_int,
                               storage_parameter_bounds,
                               experiment_start_date,
                               experiment_end_date,
                               spin_up,
                               sigma_w,
                               sigma_p_Sf,
                               )

            except Exception as e:
                print(e)



if __name__ == "__main__":
    gc.enable()
    main()
