import os
import random
import numpy as np
import torch
import copy

import hydra
from omegaconf import DictConfig, open_dict
from src.train import train

@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
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
            train(fold_cfg)
        except Exception as e:
            print(f"Lỗi ở fold {current_test_fold}: {e}")

if __name__ == "__main__":
    main()
