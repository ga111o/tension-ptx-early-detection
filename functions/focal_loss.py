import numpy as np
from scipy import special

class UnifiedFocalLoss:
    """Focal Loss compatible with both LightGBM and XGBoost"""
    
    def __init__(self, gamma=2.0, alpha=0.25):
        self.gamma = gamma
        self.alpha = alpha
    
    def _compute_focal_loss_grad_hess(self, y_true, preds):
        p = special.expit(preds)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        
        y = 2 * y_true - 1
        alpha_t = np.where(y_true == 1, self.alpha, 1 - self.alpha)
        p_t = np.where(y_true == 1, p, 1 - p)
        
        g = self.gamma
        
        u = alpha_t * y * (1 - p_t) ** g
        du = -alpha_t * y * g * (1 - p_t) ** (g - 1)
        
        v = g * p_t * np.log(p_t) + p_t - 1
        dv = g * np.log(p_t) + g + 1
        
        grad = u * v
        hess = (du * v + u * dv) * y * p * (1 - p)
        
        hess = np.abs(hess) + 1e-15
        
        return grad, hess

    
    def lgb_obj(self, preds, train_data):
        """
        LightGBM objective function
        
        Note:
        Native API: (preds, train_data) where train_data is Dataset object
        Sklearn API: (y_true, y_pred) where both are numpy arrays
        """
        if hasattr(train_data, 'get_label'):
            y_true = train_data.get_label()
            y_logits = preds
        else:
            y_true = preds
            y_logits = train_data
            
        return self._compute_focal_loss_grad_hess(y_true, y_logits)
    
    def lgb_eval(self, preds, train_data):
        """LightGBM evaluation metric"""
        if hasattr(train_data, 'get_label'):
            y_true = train_data.get_label()
            y_logits = preds
        else:
            y_true = preds
            y_logits = train_data

        p = special.expit(y_logits)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        
        alpha_t = np.where(y_true == 1, self.alpha, 1 - self.alpha)
        p_t = np.where(y_true == 1, p, 1 - p)
        
        loss = -alpha_t * (1 - p_t) ** self.gamma * np.log(p_t)
        return 'focal_loss', loss.mean(), False
    
    def xgb_obj(self, preds, dtrain):
        """XGBoost objective function"""
        y_true = dtrain.get_label()
        return self._compute_focal_loss_grad_hess(y_true, preds)
    
    def xgb_eval(self, preds, dtrain):
        """XGBoost evaluation metric"""
        y_true = dtrain.get_label()
        p = special.expit(preds)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        
        alpha_t = np.where(y_true == 1, self.alpha, 1 - self.alpha)
        p_t = np.where(y_true == 1, p, 1 - p)
        
        loss = -alpha_t * (1 - p_t) ** self.gamma * np.log(p_t)
        return 'focal_loss', loss.mean()