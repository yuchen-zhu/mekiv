from miv.models.KIV.base_trainer import BaseKIVTrainer


class KIV_MTrainer(BaseKIVTrainer):
    which_regressor = "M"


class KIV_MNTrainer(BaseKIVTrainer):
    which_regressor = "avMN"


class KIV_NTrainer(BaseKIVTrainer):
    which_regressor = "N"


class KIV_XTrainer(BaseKIVTrainer):
    which_regressor = "X_hidden"
