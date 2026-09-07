# ComfyUI — AI Match Color

Custom node **color transfer / color matching** cục bộ cho ComfyUI. Node chuyển
**bảng màu, white balance, sắc độ, tương phản và cảm giác ánh sáng** của một
**Reference image** sang **Source image (hoặc batch nhiều frame)** — không dùng
mô hình tạo ảnh, không đổi bố cục, vật thể hay độ phân giải của Source.

> Source và Reference được phép khác kích thước, tỷ lệ, góc máy và chủ thể. Node
> không pixel-match, không crop, không resize Source. Ảnh kết quả giữ nguyên kích
> thước Source.

Đây là bản port lõi xử lý màu của desktop app **AI Match Color** sang ComfyUI.

## Node

| Node | Công dụng | Cần model? |
| --- | --- | --- |
| **AI Match Color** | Global LAB / Histogram / Smart match + chỉnh màu + protection | Không |
| **AI Match Color (Semantic)** | Phân vùng ADE20K, khớp màu theo 23 nhóm với strength riêng | Có — tải lần đầu chạy |

Cả hai đều nằm trong category **`AI Match Color`** của ComfyUI.

### AI Match Color (không cần model)

- **method**
  - `smart` — kết hợp LAB + phân bố luminance, ưu tiên kết quả tự nhiên (mặc định).
  - `global_lab` — khớp mean/std trong CIE-LAB (Reinhard transfer).
  - `histogram` — khớp histogram; chọn `histogram_space` = `rgb` / `lab` / `luminance`.
- **strength** `0..1` — 0 giữ nguyên Source, 1 là ảnh hưởng tối đa từ Reference.
- **Chỉnh màu live**: `exposure`, `contrast`, `temperature`, `tint`, `hue` (độ),
  `saturation`.
- **Protection**: `preserve_luminance`, `protect_highlights`, `protect_shadows`,
  `preserve_neutral`, `gamut_compression`.

### AI Match Color (Semantic)

Segment Source và Reference **độc lập** bằng model ADE20K, rồi khớp màu **theo
từng nhóm** với cường độ riêng. Vùng nào không có nhóm tương ứng trong Reference
sẽ giữ **Smart Match** toàn cục, đúng như app gốc.

- **model**
  - `segformer_b2` — *Fast/Compatible — SegFormer-B2 ADE20K* (mặc định, nhẹ,
    chạy CPU hoặc CUDA). Tự tải ~110 MB lần đầu.
  - `oneformer_swin_large` — *High Quality — OneFormer Swin-Large* (nặng hơn,
    ~840 MB, cần thêm dependency của OneFormer).
- **semantic_strength** `0..1` — hệ số nhân chung áp lên mọi group strength.
- **23 group strength** (`0..100`) — Sky, Water, Foliage/Trees, Flowers,
  Grass/Moss, Building, Wall/Ceiling, Window/Door, Ground/Floor/Path, Wood,
  Stone/Rock, Furniture, Fabric/Curtain/Rug, Glass/Reflective, Fire/Flame,
  Candle/Artificial Light, Seasonal Decorations, Small Objects/Tableware, Other,
  Person, Skin, Hair, Clothing. Mặc định lấy đúng theo spec app gốc.
- **reuse_first_frame** — khi Source là batch (video), phân đoạn frame đầu được
  đóng băng và dùng lại cho mọi frame để màu ổn định, không nhấp nháy và render
  nhanh (First-Frame Semantic Field).
- **proxy_max_edge** — cạnh dài tối đa khi chạy segmentation (giữ VRAM ổn định).
- Node Semantic cũng có đủ bộ chỉnh màu + protection như node cơ bản.

> ADE20K không có class riêng cho **Wood / Skin / Hair**, nên các nhóm này thường
> không nhận pixel và vùng đó giữ Smart Match — đúng như giới hạn đã ghi trong
> app gốc.

## Tải model lần đầu (First-run download)

Lần **đầu tiên** node Semantic chạy, model được tải tự động qua
`huggingface_hub` vào:

```
ComfyUI/models/ai_match_color/<model_key>/
```

Những lần sau load thẳng từ đĩa, không tải lại. Ba method không semantic
(`smart` / `global_lab` / `histogram`) **không cần Internet và không tải gì**.

## Cài đặt

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/OWNER/comfyui-ai-match-color.git
cd comfyui-ai-match-color
pip install -r requirements.txt
```

Rồi khởi động lại ComfyUI. `numpy`/`torch` đã có sẵn trong ComfyUI; `requirements.txt`
chỉ bổ sung `transformers` + `huggingface_hub` (cho Semantic) và `Pillow`.

> Có thể cài qua **ComfyUI-Manager** nếu repo đã được đăng ký.

## Cách dùng nhanh

1. Đưa Source và Reference vào (`Load Image` → IMAGE). Video/batch: nối chuỗi
   frame vào `source`.
2. Nối vào **AI Match Color** hoặc **AI Match Color (Semantic)**.
3. Chọn method / model, chỉnh `strength` và các slider.
4. Lấy output `IMAGE` để `Preview` / `Save` / ghép lại thành video.

## Video / batch

`IMAGE` trong ComfyUI là tensor `(B, H, W, 3)` nên một batch được coi là các
frame video. Reference là một ảnh (frame đầu của batch reference) và được áp dụng
nhất quán cho mọi frame Source. Node Semantic mặc định `reuse_first_frame=True`
để giữ màu ổn định theo thời gian.

## Kiểm thử

```bash
pip install numpy pytest
python -m pytest tests/ -q
```

Test bao phủ round-trip sRGB↔LAB, hướng dịch chuyển của Global LAB match,
histogram ở cả 3 không gian màu, endpoint của strength blend, tính bất biến của
adjustments ở giá trị 0, feather mask và ánh xạ ADE20K → 23 nhóm. Phần tải model
segmentation không chạy trong test.

## License

Code: **MIT** (xem [LICENSE](LICENSE)). Model segmentation (SegFormer-B2 /
OneFormer, ADE20K) tải lúc chạy theo **license riêng của nhà phát hành** trên
Hugging Face Hub — một số hạn chế dùng thương mại. Global / Histogram / Smart
match **không phụ thuộc model**.
