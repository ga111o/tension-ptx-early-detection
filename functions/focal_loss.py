import numpy as np
from scipy import special

class UnifiedFocalLoss:
    """LightGBM과 XGBoost 모두 사용 가능한 Focal Loss"""
    
    def __init__(self, gamma=2.0, alpha=0.25):
        self.gamma = gamma
        self.alpha = alpha
    
    def _compute_focal_loss_grad_hess(self, y_true, preds):
        """
        핵심 gradient/hessian 계산 로직 (프레임워크 독립적)
        
        Parameters:
        -----------
        y_true : array-like, shape = [n_samples]
            실제 레이블 (0 or 1)
        preds : array-like, shape = [n_samples]
            raw prediction (logits)
        
        Returns:
        --------
        grad : array, shape = [n_samples]
        hess : array, shape = [n_samples]
        """
        # Sigmoid 변환
        p = special.expit(preds)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        
        # y를 {-1, 1}로 변환
        y = 2 * y_true - 1
        
        # alpha_t와 p_t 계산
        alpha_t = np.where(y_true == 1, self.alpha, 1 - self.alpha)
        p_t = np.where(y_true == 1, p, 1 - p)
        
        # Gradient 계산
        # d/dp FL = alpha_t * (1-p_t)^gamma * [gamma*p_t*log(p_t) + p_t - 1]
        focal_term = (1 - p_t) ** self.gamma
        log_term = self.gamma * p_t * np.log(p_t) + p_t - 1
        grad = alpha_t * y * focal_term * log_term
        
        # Hessian 계산 (곱의 미분법 적용)
        # d/dp[(1-p_t)^gamma]
        dfocal_dp = -self.gamma * (1 - p_t) ** (self.gamma - 1)
        # d/dp[gamma*p_t*log(p_t) + p_t - 1]
        dlog_dp = self.gamma * np.log(p_t) + self.gamma + 1
        
        # Chain rule: d/dz = d/dp * dp/dz, where dp/dz = p(1-p)
        hess = alpha_t * y * (
            dfocal_dp * log_term + focal_term * dlog_dp
        ) * y * p * (1 - p)
        
        return grad, hess
    
    # LightGBM용 wrapper
    def lgb_obj(self, preds, train_data):
        """LightGBM objective function"""
        y_true = train_data.get_label()
        return self._compute_focal_loss_grad_hess(y_true, preds)
    
    def lgb_eval(self, preds, train_data):
        """LightGBM evaluation metric"""
        y_true = train_data.get_label()
        p = special.expit(preds)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        
        alpha_t = np.where(y_true == 1, self.alpha, 1 - self.alpha)
        p_t = np.where(y_true == 1, p, 1 - p)
        
        loss = -alpha_t * (1 - p_t) ** self.gamma * np.log(p_t)
        return 'focal_loss', loss.mean(), False
    
    # XGBoost용 wrapper
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
