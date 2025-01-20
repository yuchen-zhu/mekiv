from typing import Dict, Any, Optional
from pathlib import Path
import logging
from miv.util import dotdict, make_dotdict
from miv.models.base_KIV.trainer import BaseKIVTrainer

logger = logging.getLogger()



class KIV_MTrainer(BaseKIVTrainer):

    def __init__(self, data_configs: dotdict, train_params: dotdict,
                 gpu_flg: bool = False, dump_folder: Optional[Path] = None):
        super(KIV_MTrainer, self).__init__(data_configs, train_params)

        self.which_regressor = 'M'

    def train(self, rand_seed: int = 42, verbose: int = 0) -> float:
        """

        Parameters
        ----------
        rand_seed: int
            random seed
        verbose : int
            Determine the level of logging
        Returns
        -------
        oos_result : float
            The performance of model evaluated by oos
        """

        return self._train(which_regressor=self.which_regressor, rand_seed=rand_seed, verbose=verbose)



