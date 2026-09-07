# Kế hoạch làm tiếp trên Codex — ComfyUI AI Match Color

Tài liệu bàn giao để một agent khác (Codex) tiếp tục dự án. Đọc file này +
[README.md](README.md) trước khi sửa code.

- **Repo:** https://github.com/soul0710/comfyui-ai-match-color
- **Remote SSH đã cấu hình qua port 443** (`ssh://git@ssh.github.com:443/...`) vì
  port 22 bị chặn không ổn định. Cứ `git push origin main` là chạy.
- **Nguồn gốc:** port lõi color-transfer của desktop app *AI Match Color*
  (mô tả đầy đủ ở `../README.md` và `../PROJECT_STATUS.md` của repo cha).

## 1. Trạng thái hiện tại (đã xong & đã kiểm chứng)

| Hạng mục | Trạng thái |
| --- | --- |
| Lõi màu numpy (LAB, histogram, smart, adjustments, protections) | ✅ có test |
| Node `AIMatchColor` (global/histogram/smart) | ✅ |
| Node `AIMatchColorSemantic` (23 nhóm ADE20K) | ✅ |
| Tự tải model lần đầu → `ComfyUI/models/ai_match_color/` | ✅ **đã chạy thật** (SegFormer-B2, 4.6s) |
| SegFormer-B2 segmentation + semantic_match + reuse frame | ✅ **đã chạy thật CPU** |
| Unit test | ✅ 8/8 pass (`pytest tests/`) |
| Đóng gói (`pyproject.toml`, LICENSE, requirements) | ✅ |

**Chưa kiểm chứng thật:** đường `oneformer_swin_large`, chạy trên **GPU/CUDA**,
và chạy bên trong **ComfyUI thật** (mới chỉ chạy qua venv CPU độc lập).

## 2. Bản đồ file

| File | Vai trò |
| --- | --- |
| `__init__.py` | Entry point ComfyUI, export NODE_CLASS_MAPPINGS |
| `ai_match_color/nodes.py` | 2 node ComfyUI, tiền/hậu xử lý, batch/video |
| `ai_match_color/color_core.py` | Toàn bộ toán màu thuần numpy |
| `ai_match_color/semantic.py` | Segmentation + ánh xạ ADE20K→23 nhóm + semantic match |
| `ai_match_color/model_manager.py` | Registry model + tải lần đầu qua huggingface_hub |
| `tests/test_core.py` | Test numpy + mapping (không tải model) |

## 3. Việc cần làm — theo ưu tiên

### P1 — Đưa lên mức "dùng thật ổn định"

1. **Fallback chain như app gốc: OneFormer → SegFormer → Smart Match.**
   - Hiện `nodes.py` chỉ chạy đúng model được chọn; nếu model lỗi/OOM sẽ raise.
   - Thêm try/except trong `AIMatchColorSemantic.run`: nếu `sem.semantic_match`
     ném lỗi (load/inference/OOM), tự thử `segformer_b2`, rồi rơi về
     `cc.smart_match`. Ghi cảnh báo rõ ra `print`/log, **không nuốt lỗi im lặng**.
   - Acceptance: mô phỏng lỗi (model_key rác) vẫn ra ảnh Smart Match, có log.

2. **Kiểm chứng OneFormer thật.**
   - `oneformer_swin_large` dùng `OneFormerProcessor`/
     `OneFormerForUniversalSegmentation` (đã viết trong `semantic._load_segmenter`).
   - Chạy thử tải + inference; DiNAT/Ultra cần `natten` (chưa thêm). Nếu chỉ làm
     Swin-Large thì không cần natten.
   - Acceptance: script giống `scratchpad/test_semantic_real.py` nhưng model
     `oneformer_swin_large` chạy tới "PASSED".

3. **Giữ alpha & dtype.**
   - `_to_np` đang cắt về 3 kênh (`[..., :3]`). Nếu Source có alpha (RGBA), cần
     giữ và ghép lại ở output (ComfyUI tách alpha qua MASK, nhưng nên an toàn).
   - Acceptance: input 4 kênh không crash; alpha được bảo toàn hoặc tách đúng.

4. **Chạy trong ComfyUI thật + GPU.**
   - Test trên máy có CUDA: kiểm `_cuda_available()`, VRAM, unload model sau chạy.
   - Thêm giải phóng VRAM: sau segment, `torch.cuda.empty_cache()`; cân nhắc
     unload model khỏi `_SEG_CACHE` khi cần (low-VRAM).

### P2 — Tính năng còn thiếu so với app gốc

5. **Node xuất LUT `.cube`** (17/33/65) từ phép match toàn cục — app gốc có.
6. **Xuất semantic mask ra `MASK`** để người dùng dùng tiếp; và **nhận MASK vào**
   để giới hạn vùng match.
7. **Example workflow** `example_workflows/ai_match_color.json` để kéo-thả trong
   ComfyUI (một cho basic, một cho semantic).
8. **Cải thiện ánh xạ nhóm:** Wood/Skin/Hair không có class ADE20K riêng nên hiện
   luôn rỗng. Cân nhắc thêm face/skin detector (mediapipe/insightface) cho
   Person/Skin/Hair, hoặc ghi rõ giới hạn (README đã ghi).

### P3 — Chất lượng & phát hành

9. **Cache segmentation theo fingerprint ảnh** (path/size/hash) như app gốc, để
   đổi slider không chạy lại model. Trong ComfyUI có thể cache ở class-level dict.
10. **Chuyển toán màu sang torch** (GPU) cho ảnh lớn/video dài thay vì numpy CPU.
11. **UI 23 slider gọn hơn**: hiện là 23 widget FLOAT rất dài. Cân nhắc gộp thành
    1 input JSON hoặc web extension panel, hoặc tách node "group override".
12. **Đăng ký ComfyUI-Manager / registry**: điền `PublisherId` trong
    `pyproject.toml`, thêm node vào danh sách của Manager.

## 4. Cách test nhanh

```bash
# unit (chỉ cần numpy + pytest)
python -m pytest tests/ -q

# end-to-end semantic thật (cần torch + transformers + torchvision + huggingface_hub + pillow)
#   tham khảo scratchpad/test_semantic_real.py: monkeypatch model_manager._models_root
#   để tải model vào thư mục tạm, rồi gọi sem.segment / sem.semantic_match.
```

Lưu ý transformers **>=5** bắt buộc `torchvision` cho image processor; SegFormer
được load bằng `AutoImageProcessor(use_fast=False)` trong `semantic.py`.

## 5. Quy ước

- Không xóa logic cũ chỉ để rút gọn (giữ tinh thần "bảo toàn tính năng" của app gốc).
- IMAGE của ComfyUI: `torch.float32 (B,H,W,3)` trong `[0,1]`. Batch = frame video.
- Import torch/transformers **lazy trong hàm**, không import ở top-level module
  (để `pytest` chạy được khi chỉ có numpy).
- Model tải lần đầu vào `ComfyUI/models/ai_match_color/<key>/`, lần sau load từ đĩa.
