# Báo Cáo Giải Thích Các Trường Thông Tin — Credit Scoring Web App

> Tài liệu hướng dẫn người dùng điền thông tin trên form đánh giá tín dụng.
> Dành cho khách hàng **chưa có lịch sử tín dụng**.

---

## 👤 Nhóm 1: Thông tin cá nhân

| Trường | Giải thích | Ví dụ |
|--------|-----------|-------|
| **Giới tính** | Chọn Nam hoặc Nữ | Nam |
| **Tuổi** | Tuổi hiện tại của bạn (18–80). Người lớn tuổi hơn thường có điểm tín dụng tốt hơn do cuộc sống ổn định hơn | 30 tuổi |
| **Số con** | Số con phụ thuộc đang nuôi dưỡng. Nhiều con hơn nghĩa là gánh nặng tài chính lớn hơn, có thể ảnh hưởng nhẹ đến rủi ro | 1 |
| **Thành viên gia đình** | Tổng số người cùng sống trong hộ gia đình (bao gồm bản thân). Dùng để tính thu nhập trên đầu người | 3 |
| **Trình độ học vấn** | Cấp học cao nhất đã hoàn thành. Học vấn cao hơn tương quan với khả năng trả nợ tốt hơn | Đại học |
| **Tình trạng hôn nhân** | Tình trạng gia đình hiện tại. Đã kết hôn thường có điểm tốt hơn (ổn định tài chính hộ gia đình) | Đã kết hôn |

---

## 💰 Nhóm 2: Thông tin khoản vay

| Trường | Giải thích | Ví dụ |
|--------|-----------|-------|
| **Thu nhập hàng năm (VNĐ)** | Tổng thu nhập trước thuế trong 1 năm. Model sử dụng con số này để tính **tỷ lệ vay/thu nhập (DTI)** — đây là chỉ số rủi ro quan trọng bậc nhất. Thu nhập càng cao so với khoản vay → điểm càng tốt | 150,000,000 VNĐ |
| **Số tiền muốn vay (VNĐ)** | Tổng số tiền bạn muốn vay. Vay càng lớn so với thu nhập → rủi ro càng cao | 300,000,000 VNĐ |
| **Trả góp hàng tháng (VNĐ)** | Số tiền phải trả mỗi tháng (gốc + lãi). Model tính tỷ lệ trả góp/thu nhập để đánh giá khả năng chi trả | 10,000,000 VNĐ |
| **Giá trị hàng hóa / mục đích vay (VNĐ)** | Giá trị thực tế của tài sản/hàng hóa bạn định mua. Nếu số tiền vay > giá trị hàng → dấu hiệu rủi ro (vay dư, dùng tiền vào việc khác) | 280,000,000 VNĐ |
| **Loại hợp đồng** | *Vay tiền mặt* = vay 1 lần cố định, trả góp đều hàng tháng; *Vay tín dụng tuần hoàn* = hạn mức quay vòng, trả rồi vay lại (giống thẻ tín dụng) | Vay tiền mặt |

---

## 💼 Nhóm 3: Việc làm & Nghề nghiệp

| Trường | Giải thích | Ví dụ |
|--------|-----------|-------|
| **Số năm đi làm** | Thâm niên ở công việc hiện tại (tính bằng năm). Làm việc càng lâu = càng ổn định = điểm tín dụng càng tốt | 5 năm |
| **Loại thu nhập** | Nguồn thu nhập chính: Đi làm (Working), Kinh doanh (Businessman), Hưu trí (Pensioner), Công chức (State servant), v.v. Mỗi loại có mức rủi ro khác nhau | Working |
| **Nghề nghiệp** | Không bắt buộc. Nếu chọn, model dùng để đánh giá mức ổn định và thu nhập kỳ vọng của nghề (VD: IT staff vs Low-skill Laborers) | Core staff |

---

## 🏠 Nhóm 4: Tài sản & Nhà ở

| Trường | Giải thích | Ví dụ |
|--------|-----------|-------|
| **Sở hữu ô tô** | Bạn có đang sở hữu ô tô không? Có xe = bằng chứng về tài sản tích lũy → giảm rủi ro. Nếu chọn "Có", sẽ hiện thêm trường "Tuổi xe" | Có |
| **Tuổi xe (năm)** | Xe đã sử dụng bao nhiêu năm? Xe mới hơn = giá trị tài sản cao hơn | 3 năm |
| **Sở hữu bất động sản** | Bạn có nhà/đất đứng tên không? Có BĐS = tài sản thế chấp tiềm năng → giảm rủi ro đáng kể | Có |
| **Loại nhà ở** | Bạn đang ở đâu? Nhà riêng/chung cư sở hữu được đánh giá tốt hơn nhà thuê hoặc ở cùng bố mẹ | Nhà riêng / căn hộ |

---

## 📊 Nhóm 5: Điểm tín dụng thay thế (External Sources)

> ⚠️ **Đây là nhóm quan trọng nhất**, chiếm hơn 50% ảnh hưởng đến kết quả.
> Scale: **0.0** (rất kém) → **1.0** (rất tốt).

| Trường | Nguồn dữ liệu | Giải thích | Ví dụ |
|--------|---------------|-----------|-------|
| **Viễn thông (Nguồn 1)** | Nhà mạng (Viettel, VNPT, Mobifone) | Điểm dựa trên: thời gian sử dụng thuê bao, tần suất nạp tiền, mức chi tiêu di động hàng tháng, có trả cước đúng hạn không. Dùng SIM lâu năm + thanh toán đều = điểm cao | 0.65 |
| **Tiện ích & Hóa đơn (Nguồn 2)** | Công ty điện (EVN), nước, internet | Điểm dựa trên: lịch sử thanh toán hóa đơn điện/nước/internet, mức tiêu thụ ổn định. Trả đúng hạn nhiều tháng liên tục = điểm cao | 0.70 |
| **Thương mại điện tử (Nguồn 3)** | Sàn TMĐT (Shopee, Lazada, Tiki) | Điểm dựa trên: tuổi tài khoản, tần suất mua hàng, tỷ lệ hoàn thành đơn, có sử dụng "mua trước trả sau" và trả đúng hạn không | 0.55 |

---

## 📱 Nhóm 6: Thông tin liên lạc

| Trường | Giải thích | Ví dụ |
|--------|-----------|-------|
| **SĐT cơ quan** | Bạn có cung cấp được số điện thoại bàn của công ty/cơ quan đang làm việc không? | ✅ Có |
| **SĐT nơi làm việc** | Số điện thoại trực tiếp tại bộ phận/phòng ban nơi bạn làm việc | ❌ Không |
| **SĐT nhà** | Bạn có số điện thoại cố định tại nhà riêng không? | ✅ Có |
| **Có email** | Bạn có địa chỉ email cá nhân không? Có email = khả năng tiếp cận kỹ thuật số, thường tương quan với hồ sơ ít rủi ro hơn | ✅ Có |

> 💡 Cung cấp càng **nhiều kênh liên lạc** → model đánh giá bạn là **đáng tin cậy hơn** (dễ xác minh, dễ liên lạc).

---

## 📈 Kết quả trả về

Sau khi nhấn **Đánh Giá**, hệ thống sẽ trả về:

| Kết quả | Ý nghĩa |
|---------|---------|
| **FICO Score (300-850)** | Điểm tín dụng theo thang chuẩn Mỹ. Càng cao = càng tốt |
| **Xác suất vỡ nợ (%)** | Khả năng bạn không trả được nợ, đã hiệu chỉnh (calibrated) |
| **Xếp hạng tín dụng** | 5 mức: 🌟 Exceptional (800+) · ✅ Very Good (740-799) · 👍 Good (670-739) · ⚠️ Fair (580-669) · ❌ Poor (<580) |
| **Giải thích SHAP** | Top 10 yếu tố ảnh hưởng nhất đến kết quả của bạn (đỏ = tăng rủi ro, xanh = giảm rủi ro) |
| **Đề xuất cải thiện** | Gợi ý cụ thể để nâng cao điểm tín dụng |
| **Hạn mức đề xuất (VNĐ)** | Số tiền tối đa hệ thống đề xuất cho vay, dựa trên thu nhập và mức rủi ro |

---

## 🔢 Công thức tính FICO Score

### Nguyên lý

Hệ thống sử dụng công thức **log-odds** chuẩn ngành ngân hàng để chuyển đổi xác suất vỡ nợ → điểm FICO:

```
FICO = Base_Score − Factor × ln(odds) − Penalties
```

Trong đó:
- `odds = P(default) / (1 − P(default))`
- `Factor = PDO / ln(2) ≈ 57.71`
- `PDO = 40` (Points to Double the Odds — mỗi khi odds tăng gấp đôi, điểm giảm 40)
- `Base_Score = 600` (điểm tại odds 1:1, tức P(default) = 50%)

### Tham số hệ thống

| Tham số | Giá trị | Mục đích |
|---------|---------|----------|
| PDO | 40 | Phạt mạnh — mỗi khi rủi ro tăng gấp đôi, điểm giảm 40 |
| Base Score | 600 | Điểm cơ sở tại odds = 1:1 |
| No-history penalty | −20 | Phạt cố định cho khách hàng chưa có lịch sử tín dụng |
| EXT default penalty | −15/nguồn | Phạt nếu điểm thay thế ở mức trung tính (≈ 0.5) |
| EXT perfect cap | 750 | FICO tối đa nếu ≥ 2 nguồn có điểm > 0.85 (chống gaming) |
| EXT high dampening | tối đa −30 | Giảm điểm khi trung bình EXT_SOURCE > 0.75 |

### Hệ thống phạt EXT_SOURCE

Vì **EXT_SOURCE chiếm hơn 50% ảnh hưởng** đến kết quả model, hệ thống áp dụng 3 cơ chế kiểm soát:

1. **Neutral penalty**: Nếu điểm thay thế = 0.5 (chưa có dữ liệu / mặc định), mỗi nguồn bị trừ 15 điểm
2. **Perfect cap**: Nếu ≥ 2 nguồn có điểm > 0.85, FICO bị giới hạn tối đa 750 (tránh lạm dụng)
3. **High dampening**: Nếu trung bình 3 nguồn > 0.75, trừ thêm tỷ lệ tối đa 30 điểm

---

## 📊 Phân phối xác suất dự đoán (OOF — 307,506 hồ sơ)

### Thống kê P(default)

| Metric | Giá trị |
|--------|---------|
| Count | 307,506 |
| Mean | 0.389 |
| Median | 0.358 |
| Std Dev | 0.216 |
| Min | 0.003 |
| P1 | 0.041 |
| P5 | 0.085 |
| Q25 | 0.212 |
| Q75 | 0.549 |
| P95 | 0.780 |
| P99 | 0.867 |
| Max | 0.958 |
| Skewness | +0.40 |
| Kurtosis | −0.79 |

### Phân phối theo nhóm thực tế

| Nhóm | N | Mean P | Median P | Std | Min | Max |
|------|---|--------|----------|-----|-----|-----|
| Good (TARGET=0) | 282,682 | 0.371 | 0.340 | 0.208 | 0.003 | 0.957 |
| Bad (TARGET=1) | 24,824 | 0.588 | 0.618 | 0.207 | 0.012 | 0.958 |

### Phân phối FICO Score

| Metric | Giá trị |
|--------|---------|
| Mean | 632.7 |
| Median | 633.7 |
| Std Dev | 63.6 |
| Min | 419.2 |
| P1 | 491.9 |
| P5 | 527.1 |
| Q25 | 588.6 |
| Q75 | 675.7 |
| P95 | 737.0 |
| P99 | 782.5 |
| Max | 850.0 |

### FICO theo nhóm thực tế

| Nhóm | Mean FICO | Median FICO | Std |
|------|-----------|-------------|-----|
| Good (TARGET=0) | 637.7 | 638.3 | 61.7 |
| Bad (TARGET=1) | 576.1 | 572.2 | 57.7 |

---

## 🏷️ Phân bổ theo Tier (FICO-based)

| Tier | Khoảng điểm | Số hồ sơ | Tỷ lệ | Tỷ lệ vỡ nợ thực tế |
|------|-------------|----------|--------|---------------------|
| 🌟 Exceptional | 800–850 | 1,517 | 0.5% | 0.73% |
| ✅ Very Good | 740–799 | 12,426 | 4.0% | 0.80% |
| 👍 Good | 670–739 | 71,875 | 23.4% | 1.95% |
| ⚠️ Fair | 580–669 | 154,590 | 50.3% | 6.20% |
| ❌ Poor | 300–579 | 63,627 | 20.7% | 21.25% |

> **Nhận xét**: Hệ thống phân tách rủi ro tốt — tỷ lệ vỡ nợ tăng đều từ 0.73% (Exceptional) lên 21.25% (Poor), cho thấy thang FICO phản ánh đúng mức độ rủi ro thực tế.

---

## 🔄 So sánh Old (0-1000) vs New FICO (300-850)

| Hồ sơ mẫu | Old Score | New FICO | Tier mới |
|-----------|----------|---------|----------|
| Khách hàng tốt (thu nhập cao, có BĐS, EXT ~0.65) | 994/1000 | **745/850** | Very Good |
| Khách hàng rủi ro (thu nhập thấp, trẻ, EXT ~0.25) | 964/1000 | **631/850** | Fair |
| EXT_SOURCE mặc định (0.5, 0.5, 0.5) | 987/1000 | **661/850** | Fair (−45 penalty) |
| EXT_SOURCE hoàn hảo (0.95, 0.95, 0.95) | 996/1000 | **727/850** | Good (capped) |

> **Cải thiện chính**: Thang cũ cho điểm quá cao (>960 cho mọi profile), không phân biệt rủi ro. Thang FICO mới phân tách rõ ràng và phạt hợp lý.

---


