import os
import random
import numpy as np
import torch
import copy
from typing import Optional

import hydra
from omegaconf import DictConfig, open_dict
from src.train import train
from src.utils import extras, get_metric_value

##############################################
########## run k_fold_cross_validation########
##############################################

@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # apply extra utilities
    extras(cfg)

    # 1. TÁI TẠO CHÍNH XÁC MÔI TRƯỜNG CỦA CODE GỐC: Set seed đúng 1 lần ở đầu
    random_seed = 30
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    
    # 2. XÓA CẤU HÌNH SEED TRONG HYDRA ĐỂ NÓ KHÔNG TỰ ĐỘNG RESET SAU MỖI FOLD
    with open_dict(cfg):
        cfg.seed = None 
        # Đảm bảo giữ cấu hình linh hoạt hơn cho CUDNN y hệt code gốc
        if hasattr(cfg, "trainer") and hasattr(cfg.trainer, "deterministic"):
            cfg.trainer.deterministic = False

    folds = [0, 1, 2, 3, 4]
    fold_metrics = []
    
    # Lấy tên metric cần tối ưu (ví dụ: "val_acc")
    optimized_metric = cfg.get("optimized_metric")
    
    # 3. VÒNG LẶP LIÊN TỤC 5 FOLD TRONG CÙNG 1 TIẾN TRÌNH (để cho phép RNG trôi dạt)
    for current_test_fold in folds:
        print(f"\n\n{'#'*60}")
        print(f"###### ĐANG HUẤN LUYỆN FOLD TEST = {current_test_fold} ######")
        print(f"{'#'*60}\n")
        
        # Copy config để tránh bị sửa đổi chéo
        fold_cfg = copy.deepcopy(cfg)
        
        # Thay đổi linh động fold test và fold train
        with open_dict(fold_cfg):
            fold_cfg.data.fold_test = current_test_fold
            fold_cfg.data.fold_train = [f for f in folds if f != current_test_fold]
            
        # Gọi hàm train() của src.train
        try:
            metric_dict, _ = train(fold_cfg)
            
            # Thu thập metric tối ưu từ fold này
            if optimized_metric and optimized_metric in metric_dict:
                value = metric_dict[optimized_metric].item()
                fold_metrics.append(value)
                print(f"Fold {current_test_fold}: {optimized_metric} = {value:.4f}")
        except Exception as e:
            print(f"Lỗi ở fold {current_test_fold}: {e}")

    # 4. Trả về giá trị trung bình metric qua tất cả các fold cho Optuna
    if fold_metrics:
        avg_metric = sum(fold_metrics) / len(fold_metrics)
        print(f"\n{'='*60}")
        print(f"TRUNG BÌNH {optimized_metric} qua {len(fold_metrics)} fold: {avg_metric:.4f}")
        print(f"{'='*60}")
        return avg_metric
    
    return None

if __name__ == "__main__":
    main()
