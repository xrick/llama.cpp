---
title: "ggml backend 抽象層 — 程式碼與文件解析報告"
subtitle: "面試準備：ggml-backend.h / ggml-backend.cpp / ggml-backend-dl.cpp / docs/backend/"
date: "2026-06-08"
---

# 0. 總覽：這四個目標在 llama.cpp 中的角色

llama.cpp 的計算核心是 **ggml**（一個輕量的 tensor / 計算圖函式庫）。
真正執行運算的硬體可能是 CPU、CUDA GPU、Metal、Vulkan、SYCL、CANN(NPU)…等。
**ggml「backend 抽象層」就是介於「計算圖」與「各種硬體實作」之間的那一層介面（interface layer）**。

它的設計目標可以用一句話概括：

> 讓上層（llama.cpp 模型推論）用同一套 API 描述計算圖，而不需要知道底下到底跑在哪種硬體上；
> 同時允許「多個 backend 同時協作」（例如 GPU 跑大部分、CPU 跑 GPU 不支援的算子）。

四個目標各自的定位：

| 目標 | 類型 | 一句話定位 |
|------|------|-----------|
| `ggml/include/ggml-backend.h` | 標頭檔（API 契約） | 定義所有對外公開的型別與函式宣告，是「介面規格書」 |
| `ggml/src/ggml-backend.cpp` | 實作（核心邏輯） | 介面的通用實作 + **多後端排程器 (scheduler)** + CPU buffer 實作 |
| `ggml/src/ggml-backend-dl.cpp` | 實作（OS 抽象） | 跨平台「動態函式庫載入」薄封裝（dlopen / LoadLibrary） |
| `docs/backend/` | 文件資料夾 | 各硬體 backend 的安裝/建置/使用說明（非程式碼） |

**關鍵心智模型（面試一定要會講）**：整個 backend 層採用 **「C ABI + 不透明指標 (opaque pointer) + 介面結構 (vtable / iface struct)」** 的物件導向模擬手法。
公開的 `ggml-backend.h` 只給你一堆 `typedef struct ... * xxx_t`（你看不到內部），所有實際行為透過每個物件內含的 `iface`（一組函式指標）來分派 (dispatch)。這就是用 C 寫出多型 (polymorphism) 的經典做法。

---

# 1. `ggml-backend.h` — 介面與 API 契約

## 1.1 核心作用
這是整個 backend 系統的「對外規格」。它**只宣告、不實作**，定義了六層核心抽象、相關的 enum/struct，以及可被各硬體 backend 實作的函式指標型別 (`*_t`)。

檔頭的 `GGML_BACKEND_API` 巨集處理 Windows DLL 的 `dllexport/dllimport` 與 GCC/Clang 的 `visibility("default")`，並用 `extern "C"` 包住，確保是 **穩定的 C ABI**（可被動態載入、可跨編譯器/語言）。

## 1.2 六層核心抽象（由小到大）

這是理解整個檔案的骨架，建議用「資源 → 串流 → 裝置 → 註冊表 → 排程器」的層次去記：

1. **Backend buffer type (`ggml_backend_buffer_type_t`, 簡稱 buft)**
   描述「一種記憶體的種類」，例如「CUDA device memory」「CPU host memory」「pinned memory」。
   提供：對齊需求 `get_alignment`、最大配置量 `get_max_size`、某 tensor 實際要多少 bytes `get_alloc_size`、是否為 host 記憶體 `is_host`、以及最重要的 `alloc_buffer`（配置出一塊 buffer）。

2. **Backend buffer (`ggml_backend_buffer_t`)**
   一塊「實際配置出來的記憶體」。
   提供：`get_base`（基底位址）、`init_tensor`、`set_tensor/get_tensor`（資料搬進搬出）、`memset_tensor`、`cpy_tensor`、`clear`。
   有 `usage` 標記（`ANY` / `WEIGHTS` / `COMPUTE`）→ 排程器會用它來決定權重應該擺哪。

3. **Backend (stream) (`ggml_backend_t`)**
   代表「一個可以執行運算的執行串流」（類似 CUDA stream）。
   提供：`graph_compute`（執行計算圖）、非同步資料搬移 `tensor_set/get_async`、`synchronize`（等待完成）、graph plan、event 相關等。
   *注意：`backend` 是「串流」概念，`device` 才是「硬體裝置」概念，兩者分開。*

4. **Backend device (`ggml_backend_dev_t`)**
   代表「一個實體硬體裝置」（一顆 GPU、一顆 NPU、CPU）。
   提供：名稱/描述、記憶體查詢 `get_memory`、能力查詢 `get_props`(`ggml_backend_dev_caps`：async / host_buffer / events…)、`init_backend`（從裝置產生一個 backend 串流）、`supports_op`/`supports_buft`/`offload_op`（查詢這顆裝置支不支援某算子/某記憶體型別）。
   裝置型別 `ggml_backend_dev_type`：`CPU / GPU / IGPU(整合顯卡) / ACCEL(加速器，如 BLAS、AMX) / META(包裝多裝置做張量平行)`。

5. **Backend registry (`ggml_backend_reg_t`)**
   代表「一個 backend 函式庫/模組」（例如整個 CUDA backend）。
   提供：列舉它底下有幾個 device、`get_proc_address`（用名字取得額外功能函式指標，類似 OpenGL 的 extension 機制）。
   全域註冊表 API：`ggml_backend_reg_count/get/by_name`、`ggml_backend_dev_count/get/by_name/by_type`。
   還有 **動態載入 API**：`ggml_backend_load(path)`、`ggml_backend_unload`、`ggml_backend_load_all()`（→ 這裡會用到 `ggml-backend-dl.cpp`）。

6. **Backend scheduler (`ggml_backend_sched_t`)**
   **整個系統最精華的部分**：把一張計算圖自動切分並分配到多個 backend 上協同執行。
   它負責三件事：①決定每個算子(node)該在哪個 backend 跑 ②管理 compute buffer 配置 ③在 backend 之間搬 tensor。
   標頭檔內有一大段註解示範標準用法（`sched_new → reserve → graph_compute`）。

## 1.3 其他重要宣告
- **`get_proc_address` 可取得的「common functions」**：用函式指標型別宣告，例如
  `ggml_backend_set_n_threads_t`（設執行緒數）、`ggml_backend_dev_get_extra_bufts_t`（額外 buffer type）、
  `ggml_backend_set_abort_callback_t`、`ggml_backend_get_features_t`（回報 feature flags）、以及張量平行用的 split buffer / comm 介面。
  → 這是一種「不破壞 ABI 也能擴充功能」的設計：新功能不加進固定 vtable，而是透過字串查詢。
- **Events**：`event_new/free/record/synchronize/wait` — 跨串流同步原語，用於 pipeline 平行。
- **Meta backend**：`GGML_BACKEND_META_MAX_DEVICES=16`、`ggml_backend_meta_split_axis`（沿哪個維度切張量：AXIS_0..3、MIRRORED 全複製、PARTIAL 部分和）、`ggml_backend_meta_device(...)` → 把多個「簡單裝置」包成一個「meta 裝置」做 **tensor parallelism（張量平行）**。
- **Utils**：`ggml_backend_graph_copy`（把圖複製到另一 backend）、`ggml_backend_compare_graph_backend`（比對兩個 backend 的輸出，常用於除錯/驗證數值正確性）、`ggml_backend_tensor_alloc/view_init`、以及永遠可用的 `ggml_backend_cpu_buffer_type()`。

## 1.4 設計理念小結（面試金句）
- **C ABI + opaque pointer**：穩定、可動態載入、語言無關。
- **iface（函式指標表）= 手寫 vtable**：用 C 達成多型。
- **選用 (optional) 介面函式**：很多 iface 欄位可為 `NULL`，`.cpp` 會檢查並提供預設行為（見 §2.2）。
- **能力查詢式設計**：上層透過 `supports_op/supports_buft` 詢問，而非硬編碼，達成硬體無關。

---

# 2. `ggml-backend.cpp` — 通用實作 + 排程器

檔案開頭就註明「移植到 C++ 仍在進行中」，所以風格是 C-style 為主、夾雜少量 C++（`std::vector`、`new/delete`、lambda）。
全檔約 2,400 行，可分成六大區塊：

| 區塊 | 行數範圍(約) | 內容 |
|------|------------|------|
| A. buffer type / buffer 包裝 | 33–211 | 把對外 API 轉成 `buft->iface.xxx()` / `buffer->iface.xxx()` 的薄封裝 |
| B. backend(stream) / device / reg 包裝 | 213–665 | 同上，分派到對應 iface；含 tensor copy（同步/非同步） |
| C. multi-buffer | 667–735 | 把多個 buffer 包成一個邏輯 buffer |
| D. **Scheduler（核心）** | 750–1976 | 圖切分 5-pass 演算法 + split 執行 + pipeline 平行 |
| E. Utils | 1978–2209 | graph copy / graph compare |
| F. **CPU backend buffer 實作** | 2211–2371 | 唯一「內建」的具體 backend buffer（memcpy/memset） |

## 2.1 區塊 A/B：薄封裝層（thin wrapper）
幾乎每個公開函式都是同一個模式：

```cpp
size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_alignment(buft);   // 把呼叫轉發給該物件自己的實作
}
```

**核心觀念**：`ggml-backend.cpp` 大多數函式「自己不做事」，只負責
①參數檢查 (`GGML_ASSERT`)、②處理選用函式為 NULL 的預設行為、③呼叫物件內的 `iface` 函式指標。
真正的硬體實作在各 backend 的檔案裡（`ggml-cuda.cu`、`ggml-metal.m`…）。

## 2.2 「選用介面 + 預設值」模式（很好用的面試例子）
- `get_max_size` 為 NULL → 預設回 `SIZE_MAX`。
- `get_alloc_size` 為 NULL → 預設回 `ggml_nbytes(tensor)`（並 assert 實際值 ≥ 理論值）。
- `is_host` 為 NULL → 預設 `false`。
- `set_tensor_async` 為 NULL → 自動退化 (fallback) 成「先 synchronize 再做同步 set」。

## 2.3 `ggml_backend_tensor_copy`：跨 backend 搬資料的三段式策略
（行 477–498）展示了「能快就快、不能快就退化」的工程取捨：
1. 來源在 host → 直接 `tensor_set` 到目的。
2. 目的在 host → 直接 `tensor_get` 從來源。
3. 兩邊都在裝置 → 試 `cpy_tensor`（裝置間直拷）；**若失敗，最後手段**：malloc 一塊主機暫存 → get 出來 → set 回去（debug 模式會印 "slow copy" 警告）。

非同步版 `ggml_backend_tensor_copy_async` 同理，但若裝置不支援 async 拷貝，會 synchronize 兩邊後退化成同步拷貝。

## 2.4 區塊 D：**Scheduler — 全檔最重要的核心運算**

### 2.4.1 它解決什麼問題？
你有一張計算圖 + 多個 backend（GPU0、GPU1、CPU）。
排程器要自動回答：**每個 node 在哪跑？哪些 tensor 要跨裝置複製？compute buffer 怎麼配置？** 並盡量讓「需要連續在同一裝置的 node」聚成一段 (split)，減少跨裝置搬運。

### 2.4.2 關鍵資料結構
- `struct ggml_backend_sched`：持有所有 backend、buffer types、galloc(圖記憶體配置器)、hash set（把 tensor 映射到 backend id / copies）、splits 陣列、pipeline 用的 events 等。
- `struct ggml_backend_sched_split`：一段「同一 backend 連續執行的子圖」，記錄 `backend_id`、`[i_start, i_end)`、需要的 `inputs[]`、以及該段的 graph view。
- 三個 macro：`tensor_backend_id(t)`（查某 tensor 被指派到哪個 backend）、`tensor_copy(...)`（查某 tensor 在某 backend/某 copy 的複本）。

### 2.4.3 核心演算法：`ggml_backend_sched_split_graph()` 的「五遍掃描 (5 passes)」
這是面試最可能被深入追問的部分，務必能講出每一遍在幹嘛：

- **Pass 1 — 依「預先配置的位置」指派 backend**
  若 node 的輸入/輸出已經被配置在某塊 buffer（典型是「權重」），就把該 node 指派到支援該 buffer type 且支援該算子的最高優先序 backend。
  另外：圖的「輸入 (INPUT flag)」預設丟給最後一個 backend（慣例上是 CPU）。
  *優先序規則：backends 陣列中 index 越小 = 優先序越高（見 `ggml_backend_sched_backend_id`）。*

- **Pass 2 — 擴張 (expand) 指派**
  把已指派的 backend「沿圖向上/向下傳染」給相鄰未指派的 node。
  特別處理：先擴張 GPU（非最後優先序），**刻意跳過 CPU**，目的是「除非權重在 CPU 或 GPU 之間夾著 CPU 算子，否則盡量不用 CPU」→ 減少 GPU↔CPU 來回搬運。分成 expand-gpu-down/up、expand-rest-down/up 四個子迴圈。

- **Pass 3 — 升級到更高優先序 backend + 收尾未指派**
  若某 node 目前的 backend 與某更高優先序 backend「buffer type 相同」且該 backend 也支援此算子且其來源相容，就升級過去。
  仍未指派的 node → 指派給「支援此算子且其已定位輸入最多」的 backend。

- **Pass 4 — 把剩餘的 src / view 指派**
  view 類 tensor 永遠跟著 `view_src` 同一 backend；其餘 src 跟隨其 dst node。最後 assert 每個 node 都已指派。

- **Pass 5 — 切分 split + 找出需要複製的 tensor**
  依「相鄰 node 的 backend 是否相同」把圖切成多段 split。
  觸發「開新 split」的情況：backend 改變、或權重在不相容的 backend、或單一 split 的 input 數超過上限 `GGML_SCHED_MAX_SPLIT_INPUTS(30)`。
  對於「來源在別的 backend 且目標 backend 不支援該 buffer」的 src → **建立一份複本 (copy)**，把 `node->src[j]` 改指向複本，並登記為此 split 的 input。
  若開了 pipeline 平行 (`n_copies > 1`)，每個 input 會建立多份複本以利雙緩衝。

掃描完後，組出一張新的 `sched->graph`（含為了搬資料而插入的 input dependency / input copy node），供後續配置與執行。

### 2.4.4 執行：`ggml_backend_sched_compute_splits()`
逐一執行每個 split：
1. **把該 split 的 input 複製到目標 backend**：使用者輸入要「立即複製」（避免使用者覆寫），其餘用 event 等待後再覆寫。
2. **MoE 權重最佳化（亮點）**：當算子是 `GGML_OP_MUL_MAT_ID`（Mixture-of-Experts）且權重在 host，會先讀出 ids 張量、用 bitset 找出「真正被用到的 experts」，**只複製用到的 expert 權重**（還會把連續 expert 合併成一次拷貝），大幅減少 PCIe 傳輸量。
3. 呼叫 `graph_compute_async` 執行該段子圖；若使用者設了 eval callback，則逐 node 觀察。
4. split 結束時 record event，供下一段/下一輪 pipeline 同步。

### 2.4.5 生命週期 API
`sched_new`（配置、建立 hash set、pipeline events）→ `reserve`（用「最大尺寸的量測圖」預先配置 buffer）→ `alloc_graph` → `graph_compute(_async)` → `synchronize` → `reset`（釋放上一張圖的配置）→ `free`。
另有 `set_tensor_backend`（手動指定某 node 的 backend）、`get_n_splits/n_copies`（除錯與效能觀察）。

### 2.4.6 Pipeline parallelism（管線平行）
`parallel=true` 時 `n_copies = GGML_SCHED_MAX_COPIES(4)`：對輸入做多份複本 + 用 events 在多份 copy 間輪替 (`cur_copy/next_copy`)，讓「資料搬移」與「計算」重疊，提高吞吐。

## 2.5 區塊 E：Utils（除錯/驗證工具）
- `ggml_backend_graph_copy`：把整張圖（含資料）深拷貝到另一個 backend，回傳新 ctx + buffer + graph。
- `ggml_backend_compare_graph_backend`：在兩個 backend 上逐 node 執行同一張圖並用 callback 比對輸出（算 RMS 等）→ 驗證新 backend 數值正確性的標準工具。

## 2.6 區塊 F：CPU backend buffer（唯一內建的具體實作）
為什麼放這裡？註解說明：「CPU buffer type 永遠可用，定義在這裡讓所有 backend 都能取用」。
實作非常直接（這也是理解 iface 怎麼被填的最佳範例）：
- `set_tensor` = `memcpy` 進去、`get_tensor` = `memcpy` 出來、`memset_tensor` = `memset`、`clear` = `memset` 整塊。
- `get_base` 會把位址對齊到 `TENSOR_ALIGNMENT`。
- 兩種變體：`ggml_backend_cpu_buffer_i`（自己 malloc/free）與 `..._from_ptr_i`（包裝外部指標，`free_buffer=NULL` 不負責釋放，名字叫 "CPU_Mapped"）。
- `is_host` 回 `true`（這就是為什麼 §2.3 的 host 快路徑能成立）。

---

# 3. `ggml-backend-dl.cpp` — 跨平台動態載入薄封裝

## 3.1 核心作用
**只有 48 行**，作用單純：把「載入動態函式庫」這件事在 Windows 與 POSIX 之間統一成三個函式，供 backend 註冊表 (`ggml-backend-reg.cpp`) 動態載入各硬體 backend 的 `.so/.dll` 使用。

它配合標頭 `ggml-backend-dl.h`，後者定義了：
- `dl_handle` 型別（Windows = `HMODULE`、POSIX = `void`）。
- `dl_handle_deleter` + `dl_handle_ptr`（`std::unique_ptr` 包裝，RAII 自動 `FreeLibrary`/`dlclose`）。
- 用 `std::filesystem::path`（`namespace fs`）表示路徑。

## 3.2 三個函式 × 兩個平台
| 函式 | Windows 實作 | POSIX 實作 |
|------|-------------|-----------|
| `dl_load_library(path)` | `LoadLibraryW`（並用 `SetErrorMode` 抑制「缺少 DLL」的彈窗錯誤對話框） | `dlopen(path, RTLD_NOW \| RTLD_LOCAL)` |
| `dl_get_sym(handle, name)` | `GetProcAddress` | `dlsym` |
| `dl_error()` | 回空字串 `""` | `dlerror()`（NULL 時回 `""`） |

**幾個值得講的細節**：
- Windows 用 `SetErrorMode(SEM_FAILCRITICALERRORS)`：嘗試載入某 backend DLL 失敗時不要跳系統錯誤視窗，讓程式能安靜地嘗試下一個 backend。
- POSIX 用 `RTLD_NOW`（立即解析所有符號）+ `RTLD_LOCAL`（符號不外洩到全域，避免不同 backend 符號衝突）。

## 3.3 它在整體架構中的位置
這支檔案是**「外掛式 (plug-in) backend 系統」的最底層 OS 抽象**：

```
ggml_backend_load_all()         (宣告於 ggml-backend.h)
   └─ ggml-backend-reg.cpp      掃描目錄、找 libggml-cuda.so 等
        └─ dl_load_library()    (本檔) 真正去 dlopen / LoadLibrary
        └─ dl_get_sym("ggml_backend_..._reg")  取得進入點並註冊
```

→ 這就是為什麼 llama.cpp 可以「編譯時不綁死硬體」，執行期才動態挑選可用的 GPU backend。
（`dl_load_library` 也被 `ggml-hexagon` 等 backend 內部重用。）

---

# 4. `docs/backend/` — 各硬體 backend 說明文件

這個資料夾**不是程式碼**，而是「每一種硬體 backend 的安裝/建置/使用手冊」。它正好對應 §1 提到的「外掛式 backend 生態」——每個檔案講一種後端怎麼啟用。掌握「每個檔案是哪種硬體 / 哪家廠商」即可：

| 檔案 | 對應硬體 / 廠商 | 重點一句話 |
|------|----------------|-----------|
| `BLIS.md` | CPU（開源 BLAS 函式庫，FLAME 專案） | 用高效能 BLAS 加速 CPU 上的 GEMM；`-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=FLAME` |
| `CANN.md` | 華為 **Ascend NPU**（昇騰） | CANN 異構運算架構，支援 Atlas/910C 等 NPU；`-DGGML_CANN` |
| `CUDA-FEDORA.md` | NVIDIA **CUDA**（Fedora 環境） | 在 Fedora toolbox 容器內安裝 CUDA 的教學（偏環境設定，非 backend 設計） |
| `OPENCL.md` | **Qualcomm Adreno GPU**（也可 Intel GPU） | 跨平台 OpenCL backend，主打 Snapdragon Android/Windows on Arm |
| `OPENVINO.md` | **Intel CPU/GPU/NPU** | 把 GGML 圖翻譯成 `ov::Model`，用 OpenVINO 推論引擎；支援 GGUF 生態 |
| `SYCL.md` | **Intel GPU**（oneAPI/DPC++） | 最完整的文件之一；SYCL 單一原始碼跨硬體，主攻 Intel Arc |
| `ZenDNN.md` | **AMD EPYC/Ryzen CPU** | AMD 的深度學習函式庫，用 LowOHA MatMul 加速 CPU GEMM |
| `zDNN.md` | **IBM Z / Telum 大型主機 NPU** | 注意：與 ZenDNN 不同！這是 IBM z17 的 NNPA 加速器 |
| `VirtGPU.md` | **虛擬機 GPU 遠端執行** | guest VM 內呼叫、host 真正算（virtio-gpu + APIR），零拷貝共享記憶體 |
| `VirtGPU/`（子夾） | VirtGPU 補充 | `configuration.md`（環境變數）、`development.md`（開發測試） |
| `snapdragon/`（子夾） | **Qualcomm Snapdragon 整機** | `README/developer/linux/windows.md` + CMake preset；整合 CPU+OpenCL+Hexagon |

**面試可講的整體觀察**：
- 文件涵蓋面非常廣 → 印證 ggml backend 抽象層的價值：**同一份 GGUF 模型、同一套上層程式碼，可跑在 NVIDIA/AMD/Intel/Qualcomm/Huawei/IBM 等截然不同的硬體上**。
- 每份文件結構類似：Background → OS/Hardware 支援表 → DataType 支援 → CMake 選項 → 建置步驟 → 環境變數 → Known issues。
- 有兩組「容易混淆」的命名特別被文件強調：**zDNN(IBM) ≠ ZenDNN(AMD)**；OpenCL 與 SYCL 都能跑 Intel GPU 但定位不同。

---

# 5. 面試問答重點整理（速記版）

**Q：ggml backend 抽象層想解決什麼問題？**
A：硬體無關性 + 多後端協作。上層用統一 API 描述計算圖，底層可換成任意硬體；且能把一張圖自動切分到多個 backend 協同跑。

**Q：用 C 怎麼做到「多型」？**
A：opaque pointer（對外只給 `*_t`）+ 每個物件內含 `iface`（一組函式指標，即手寫 vtable）。`.cpp` 的公開函式大多只是 `obj->iface.fn(obj)` 的轉發，外加參數檢查與「選用函式為 NULL 時的預設行為」。

**Q：backend、device、buffer、buffer_type、reg 有何差別？**
A：buffer_type=記憶體種類；buffer=實際配置出的記憶體；backend=執行串流(stream)；device=實體硬體；reg=一個 backend 模組(函式庫)。層次由小到大。

**Q：Scheduler 的核心演算法？**
A：`split_graph` 五遍掃描——(1)依預配置位置指派(權重)、(2)向相鄰節點擴張並刻意避開 CPU、(3)升級到更高優先序 backend 並收尾、(4)指派剩餘 src/view、(5)切 split + 為跨裝置來源建立複本。之後 `compute_splits` 逐段執行，期間用 events 同步、對 MoE 只搬用到的 expert。

**Q：為什麼擴張時要「跳過 CPU」？**
A：CPU 是最低優先序 backend，盡量讓連續算子留在 GPU，避免 GPU↔CPU 反覆搬資料的開銷。

**Q：`ggml-backend-dl.cpp` 的角色？**
A：跨平台動態載入薄封裝（dlopen vs LoadLibrary），是「外掛式 backend」最底層 OS 抽象，被註冊表 `ggml_backend_load_all` 用來執行期載入各硬體 `.so/.dll`。

**Q：跨 backend 拷貝 tensor 的策略？**
A：host→直接 set / 從 host→直接 get / 都在裝置→試裝置間直拷，失敗才退化成「經主機暫存」的慢路徑。

**Q：怎麼驗證一個新 backend 算對了？**
A：`ggml_backend_compare_graph_backend` 把圖在參考 backend(通常 CPU)與新 backend 各跑一遍，逐 node 比對輸出。

**Q：pipeline 平行怎麼做？**
A：`parallel=true` → `n_copies=4`，輸入做多份複本，用 events 在多 copy 間輪替，讓資料搬移與計算重疊。
