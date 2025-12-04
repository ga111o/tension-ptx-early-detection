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
        p = np.clip(p, 1e-7, 1 - 1e-7)
        
        # alpha_t와 p_t 계산
        alpha_t = np.where(y_true == 1, self.alpha, 1 - self.alpha)
        p_t = np.where(y_true == 1, p, 1 - p)
        p_t = np.clip(p_t, 1e-7, 1 - 1e-7)
        
        # Focal Loss: FL = -alpha_t * (1-p_t)^gamma * log(p_t)
        
        # 기본 cross-entropy gradient: p - y
        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Gradient 계산
        # grad = alpha_t * [(1-p_t)^gamma * (p - y) - gamma * (1-p_t)^(gamma-1) * (p - y) * p_t * log(p_t)]
        # 단순화: grad = (p - y) * alpha_t * (1-p_t)^(gamma-1) * [(1-p_t) - gamma * p_t * log(p_t)]
        grad = p - y_true
        modulating_factor = focal_weight * (1 - self.gamma * p_t * np.log(p_t) / (1 - p_t + 1e-7))
        grad = alpha_t * grad * modulating_factor
        
        # Hessian 계산 (항상 양수여야 함)
        # Focal Loss의 hessian은 복잡하므로 근사값 사용
        # hess ≈ alpha_t * focal_weight * p * (1 - p)
        hess = alpha_t * focal_weight * p * (1 - p)
        hess = np.maximum(hess, 1e-7)  # 수치 안정성을 위해 최소값 보장
        
        return grad, hess
    
    # LightGBM용 wrapper
    def lgb_obj(self, preds, train_data):
        """
        LightGBM objective function
        
        Note:
        - Native API (lgb.train): (preds, train_data) -> train_data는 Dataset 객체
        - Sklearn API (LGBMClassifier): (y_true, y_pred) -> 둘 다 numpy array
        변수명이 preds, train_data로 되어 있지만 Sklearn 사용 시 순서가 바뀜에 주의
        """
        if hasattr(train_data, 'get_label'):
            # Case 1: Native API (lgb.train)
            y_true = train_data.get_label()
            y_logits = preds
        else:
            # Case 2: Sklearn API (LGBMClassifier)
            # 인자 순서가 (y_true, y_pred)로 들어옴
            # 함수 정의(preds, train_data)와 매핑하면: preds->y_true, train_data->y_logits
            y_true = preds
            y_logits = train_data
            
        return self._compute_focal_loss_grad_hess(y_true, y_logits)
    
    def lgb_eval(self, preds, train_data):
        """LightGBM evaluation metric"""
        if hasattr(train_data, 'get_label'):
            # Case 1: Native API
            y_true = train_data.get_label()
            y_logits = preds
        else:
            # Case 2: Sklearn API
            # 인자 순서가 (y_true, y_pred)
            y_true = preds
            y_logits = train_data

        p = special.expit(y_logits)
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