from losses.B_Focal_loss import focal_loss_binary
from losses.C_Focal_loss import focal_loss_multiclasses
from losses.Dice_loss import DiceLoss
from losses.BCE_Dice_loss import BCE_DiceLoss
from losses.CE_Dice_loss import CE_DiceLoss
from losses.Tversky_loss import TverskyLoss
from losses.Focal_Tversky_loss import FocalTverskyLoss
from losses.Weighted_Categorical_loss import Weighted_Categorical_CrossEntropy_Loss
from losses.Generalized_Dice_loss import GeneralizedDiceLoss
from losses.Jaccard_loss import JaccardLoss
from losses.BCE_Jaccard_Loss import BCE_JaccardLoss
from losses.CE_Jaccard_Loss import CE_JaccardLoss


LOSS_FACTORY = {
    'ce': 'categorical_crossentropy',
    'weighted_ce': Weighted_Categorical_CrossEntropy_Loss(dataset='DRIVE'),
    'b_focal': focal_loss_binary(),
    'c_focal': focal_loss_multiclasses(dataset='DRIVE'),
    'dice': DiceLoss(),
    'bce_dice': BCE_DiceLoss(),
    'ce_dice': CE_DiceLoss(),
    'g_dice': GeneralizedDiceLoss(),
    'jaccard': JaccardLoss(),
    'bce_jaccard': BCE_JaccardLoss(),
    'ce_jaccard': CE_JaccardLoss(),
    'tversky': TverskyLoss(),
    'f_tversky': FocalTverskyLoss()
}
