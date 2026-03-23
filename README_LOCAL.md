# 本機執行說明

## 環境需求
- Python 3.10.6（透過 pyenv 管理）
- NVIDIA GPU（已測試 RTX 3060 12GB）
- PyTorch 2.10.0+cu128（與 Colab 版本對齊）

## 環境建立

```powershell
# 建立虛擬環境（僅需執行一次）
& "$env:USERPROFILE\.pyenv\pyenv-win\versions\3.10.6\python.exe" -m venv venv

# 啟動虛擬環境
.\venv\Scripts\Activate.ps1

# 安裝 PyTorch（CUDA 12.8，與 Colab 一致）
pip install torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# 安裝其他依賴套件
pip install terminaltables tqdm matplotlib tensorboard gdown
```

## 使用方式

每次使用前先啟動虛擬環境（提示符出現 `(venv)` 表示已啟動）：

```powershell
.\venv\Scripts\Activate.ps1
```

---

### 4c（Multi-Class，4 類別：blue_sedan / red_sedan / blue_SUV / red_SUV）

#### 1. 訓練模型

```powershell
python train_4c.py
```

訓練 6 epochs，checkpoint 存放於 `checkpoints/4c/`。

#### 2. 評估模型

```powershell
python test_4c.py
```

預設使用 `checkpoints/4c/yolov3_ckpt_5.pth`，可用 `--weights_path` 指定其他 checkpoint：

```powershell
python test_4c.py --weights_path checkpoints/4c/yolov3_ckpt_4.pth
```

#### 3. 偵測圖片

```powershell
python detect_4c.py
```

偵測結果存放於 `output/4c/`。

#### 4. 檢視結果圖片

```powershell
python showimg_4c.py
```

---

### 2n2c（Multi-Label，2 色 + 2 車型：blue / red / sedan / SUV）

#### 1. 訓練模型

```powershell
python train_2n2c.py
```

訓練 6 epochs，checkpoint 存放於 `checkpoints/2n2c/`。

#### 2. 評估模型

```powershell
python test_2n2c.py
```

預設使用 `checkpoints/2n2c/yolov3_ckpt_5.pth`，可用 `--weights_path` 指定其他 checkpoint：

```powershell
python test_2n2c.py --weights_path checkpoints/2n2c/yolov3_ckpt_4.pth
```

#### 3. 偵測圖片

```powershell
python detect_2n2c.py
```

偵測結果存放於 `output/2n2c/`。

#### 4. 檢視結果圖片

```powershell
python showimg_2n2c.py
```

---

## 程式碼修改紀錄（相較於 Colab 版本）

1. **utils/logger.py** — Logger 從 TensorFlow 改為使用 PyTorch 內建的 `torch.utils.tensorboard`
2. **utils/utils.py** — 修正 Windows 路徑問題：`{:s}` 改為 `{}`（Windows 的 `os.path.join` 會將冒號視為磁碟代號）

## 備註
- `checkpoints/`、`weights/`、`logs/`、`__pycache__/`、`venv/` 已加入 `.gitignore`，不會上傳至 GitHub
- 每次開新終端機都需要重新執行 `.\venv\Scripts\Activate.ps1` 啟動虛擬環境
