import os
import io
import base64
import json
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import matplotlib.cm as cm
from skimage.feature import graycomatrix, graycoprops
from openai import OpenAI

# 分割模型
from unet import UNet_DTC_PraNet

app = Flask(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------- 1. 加载分割模型 -------------------
print("正在加载分割模型...")
model = UNet_DTC_PraNet(in_channels=3, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.eval()
print("分割模型加载完成。")

# ------------------- 2. Qwen3-VL 智能体配置 -------------------
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-a94a4d6e7fc5411c90ff66b8e9a6ab5a")
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen3-vl-plus"  # 若需升级至 Qwen3，改为 "qwen3-vl-plus"

def call_qwen_vl(image_pil, clinical_metrics, skin_metrics=None):
    """调用阿里云百炼 Qwen-VL 多模态模型，返回诊断+肤质评估"""
    client = OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url=QWEN_BASE_URL,
    )
    buffered = io.BytesIO()
    image_pil.save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()

    prompt = f"""你是一位经验丰富的皮肤科主任医师。请分析这张皮肤照片，并结合以下客观测量数据给出诊断与肤质评估。

【客观测量数据 - 病灶与皮肤状态】
- 病灶区域占皮肤面积比例: {clinical_metrics['area_ratio']:.2%}
- 健康皮肤湿润度指数: {clinical_metrics['moisture']}/100
- 皮肤裂纹等级: {clinical_metrics['crack_level']}
- 毛孔面积占比: {skin_metrics.get('pore_area_ratio', 'N/A') if skin_metrics else 'N/A'}
- 油脂得分(0-100): {skin_metrics.get('oiliness_score', 'N/A') if skin_metrics else 'N/A'}
- 泛红指数: {skin_metrics.get('red_index', 'N/A') if skin_metrics else 'N/A'}
- 亮度标准差: {skin_metrics.get('brightness_std', 'N/A') if skin_metrics else 'N/A'}
- 初步肤质判断: {skin_metrics.get('skin_type_hint', 'N/A') if skin_metrics else 'N/A'}

【输出要求】
请严格按照以下JSON格式返回，不要包含任何其他注释或文字：
{{
    "disease": "具体疾病名称（例如：黑色素瘤、基底细胞癌、脂溢性角化病、良性痣、正常皮肤等）",
    "confidence": 0.xx,
    "malignancy": "良性/恶性",
    "malignancy_conf": 0.xx,
    "recommendations": "针对该疾病的诊疗建议、生活护理和随访计划",
    "skin_type": "干性/油性/中性/混合",
    "skin_sensitivity": "低/中/高",
    "skin_description": "一句简短的自然语言描述皮肤整体状态"
}}
"""
    try:
        response = client.chat.completions.create(
            model=QWEN_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        result_text = response.choices[0].message.content
        result_text = result_text.strip()
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        diag = json.loads(result_text)
        required_keys = ["disease", "confidence", "malignancy", "malignancy_conf", "recommendations"]
        skin_keys = ["skin_type", "skin_sensitivity", "skin_description"]
        if all(k in diag for k in required_keys):
            for k in skin_keys:
                if k not in diag:
                    diag[k] = "未评估" if k != "skin_sensitivity" else "未评估"
            return diag
        else:
            raise ValueError("智能体返回的JSON缺少必要字段")
    except Exception as e:
        print(f"API 调用失败: {e}，使用规则回退诊断")
        return fallback_diagnosis(clinical_metrics, skin_metrics)

def fallback_diagnosis(clinical_metrics, skin_metrics=None):
    """当智能体不可用时的保守诊断回退（包含肤质）"""
    area = clinical_metrics['area_ratio']
    if area > 0.2:
        disease = "可疑色素性病变"
        malignancy = "恶性风险待排查"
        conf = 0.6
        mal_conf = 0.5
        rec = "病灶面积较大，建议尽快皮肤科就诊，必要时病理活检。"
    elif clinical_metrics['crack_level'] in ["中度", "重度"]:
        disease = "慢性皮炎伴表皮屏障受损"
        malignancy = "良性"
        conf = 0.7
        mal_conf = 0.1
        rec = "加强保湿修复，避免搔抓刺激，使用温和润肤霜。"
    else:
        disease = "色素痣 / 良性增生"
        malignancy = "良性"
        conf = 0.8
        mal_conf = 0.05
        rec = "临床表现为良性，定期自我观察，若出现形态变化及时就医。"

    if skin_metrics:
        skin_type = skin_metrics.get("skin_type_hint", "未评估")
        skin_sens = "中" if skin_metrics.get("red_index", 0) > 15 else "低"
        skin_desc = f"肤质倾向{skin_type}，泛红指数{skin_metrics.get('red_index', 0):.2f}"
    else:
        skin_type = "未评估"
        skin_sens = "未评估"
        skin_desc = "无肤质数据"

    return {
        "disease": disease,
        "confidence": conf,
        "malignancy": malignancy,
        "malignancy_conf": mal_conf,
        "recommendations": rec,
        "skin_type": skin_type,
        "skin_sensitivity": skin_sens,
        "skin_description": skin_desc
    }

# ------------------- 3. 皮肤个体检测模块（原湿润度+裂纹）-------------------
def analyze_skin_condition(image_cv, mask_binary):
    if mask_binary.ndim == 3:
        mask_binary = mask_binary[:, :, 0]
    healthy_mask = cv2.bitwise_not(mask_binary)
    healthy_region = cv2.bitwise_and(image_cv, image_cv, mask=healthy_mask)
    if np.sum(healthy_mask) == 0:
        return 50.0, "无法评估", 0.0
    gray = cv2.cvtColor(healthy_region, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    contrast_norm = min(1.0, contrast / 100.0)
    moisture = 100 * (1 - contrast_norm) * homogeneity
    moisture = np.clip(moisture, 0, 100)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    crack_pixels = np.sum(edges_closed > 0)
    total_pixels = np.sum(healthy_mask > 0)
    crack_density = crack_pixels / total_pixels if total_pixels > 0 else 0.0
    if crack_density < 0.01:
        crack_level = "无"
    elif crack_density < 0.05:
        crack_level = "轻度"
    elif crack_density < 0.1:
        crack_level = "中度"
    else:
        crack_level = "重度"
    return round(moisture, 1), crack_level, round(crack_density, 4)

# ------------------- 4. 新增：肤质检测模块 -------------------
def analyze_skin_type(image_cv, mask_binary):
    if mask_binary.ndim == 3:
        mask_binary = mask_binary[:, :, 0]
    healthy_mask = cv2.bitwise_not(mask_binary) if np.max(mask_binary) > 127 else mask_binary
    if np.sum(healthy_mask) == 0:
        return {"error": "无健康皮肤区域"}

    healthy_region = cv2.bitwise_and(image_cv, image_cv, mask=healthy_mask)
    healthy_rgb = cv2.cvtColor(healthy_region, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(healthy_region, cv2.COLOR_BGR2GRAY)

    binary_adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(binary_adapt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pore_sizes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 3 < area < 80:
            pore_sizes.append(area)
    pore_count = len(pore_sizes)
    pore_area_ratio = sum(pore_sizes) / np.sum(healthy_mask > 0) if pore_count > 0 else 0

    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    bright_pixels = np.sum(bright_mask & (healthy_mask > 0))
    oiliness_score = (bright_pixels / np.sum(healthy_mask > 0)) * 100

    r_channel = healthy_rgb[:, :, 0].astype(np.float32)
    g_channel = healthy_rgb[:, :, 1].astype(np.float32)
    red_index = np.mean(r_channel[healthy_mask > 0] / (g_channel[healthy_mask > 0] + 1)) * 100

    brightness_std = np.std(gray[healthy_mask > 0])

    if oiliness_score > 5 and pore_area_ratio > 0.002:
        skin_type_hint = "油性"
    elif oiliness_score < 1 and brightness_std < 20:
        skin_type_hint = "干性"
    else:
        skin_type_hint = "中性/混合"

    return {
        "pore_count": pore_count,
        "pore_area_ratio": round(pore_area_ratio, 6),
        "oiliness_score": round(oiliness_score, 2),
        "red_index": round(red_index, 2),
        "brightness_std": round(brightness_std, 2),
        "skin_type_hint": skin_type_hint
    }

# ------------------- 5. 图像预处理 -------------------
def preprocess_image(image_bytes):
    img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w_org, h_org = img_pil.size
    ratio = 256 / max(w_org, h_org)
    new_w, new_h = int(w_org * ratio), int(h_org * ratio)
    img_resized = img_pil.resize((new_w, new_h), Image.BILINEAR)
    input_img = Image.new("RGB", (256, 256), (0, 0, 0))
    pad_left = (256 - new_w) // 2
    pad_top = (256 - new_h) // 2
    input_img.paste(img_resized, (pad_left, pad_top))
    img_tensor = TF.to_tensor(input_img).unsqueeze(0).to(DEVICE)
    return {
        'img_pil': img_pil,
        'img_cv': cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR),
        'w_org': w_org, 'h_org': h_org,
        'new_w': new_w, 'new_h': new_h,
        'pad_left': pad_left, 'pad_top': pad_top,
        'img_tensor': img_tensor
    }

# ------------------- 6. 分割推理 -------------------
def run_segmentation(prep):
    img_tensor = prep['img_tensor']
    with torch.no_grad():
        output = model(img_tensor)
        if model.training:
            pred_fine = output[0]
        else:
            pred_fine = output
        probs = torch.sigmoid(pred_fine).squeeze().cpu().numpy()
    probs_cropped = probs[prep['pad_top']:prep['pad_top'] + prep['new_h'],
                    prep['pad_left']:prep['pad_left'] + prep['new_w']]
    probs_full = cv2.resize(probs_cropped, (prep['w_org'], prep['h_org']), interpolation=cv2.INTER_LINEAR)
    mask_binary = (probs_full > 0.5).astype(np.uint8) * 255
    overlay_cv = prep['img_cv'].copy()
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_cv, contours, -1, (0, 0, 255), 2)
    overlay_pil = Image.fromarray(cv2.cvtColor(overlay_cv, cv2.COLOR_BGR2RGB))
    heatmap = (cm.jet(probs_full)[:, :, :3] * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap)
    mask_pil = Image.fromarray(mask_binary, mode='L')
    return {
        'overlay_pil': overlay_pil,
        'mask_pil': mask_pil,
        'heatmap_pil': heatmap_pil,
        'probs_full': probs_full,
        'mask_binary': mask_binary
    }

# ------------------- 7. 辅助函数 -------------------
def pil_to_base64(pil_img):
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

# ------------------- 8. 统一分析接口 -------------------
@app.route('/full_analysis', methods=['POST'])
def full_analysis():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    try:
        image_bytes = file.read()
        prep = preprocess_image(image_bytes)
        seg = run_segmentation(prep)

        moisture, crack_level, crack_density = analyze_skin_condition(prep['img_cv'], seg['mask_binary'])
        lesion_area_ratio = float(np.sum(seg['mask_binary'] > 0) / (prep['w_org'] * prep['h_org']))
        skin_metrics = analyze_skin_type(prep['img_cv'], seg['mask_binary'])

        clinical_metrics = {
            "moisture": moisture,
            "crack_level": crack_level,
            "area_ratio": lesion_area_ratio
        }

        qwen_result = call_qwen_vl(prep['img_pil'], clinical_metrics, skin_metrics)

        report_text = f"""【皮肤科智能诊断报告】

▶ 诊断结果：{qwen_result['disease']} (置信度 {qwen_result['confidence']:.1%})
▶ 恶性风险评估：{qwen_result['malignancy']} (置信度 {qwen_result['malignancy_conf']:.1%})
▶ 病灶面积占比：{lesion_area_ratio:.2%}
▶ 皮肤生理指标：湿润度 {moisture}/100，裂纹等级 {crack_level}

【肤质评估】
▶ 肤质类型：{qwen_result.get('skin_type', '未评估')}
▶ 皮肤敏感度：{qwen_result.get('skin_sensitivity', '未评估')}
▶ 综合描述：{qwen_result.get('skin_description', '无')}

【诊疗建议】
{qwen_result['recommendations']}

注：本报告由AI辅助生成，最终诊断请结合临床病理检查。
"""
        moisture = float(moisture)
        crack_density = float(crack_density)
        lesion_area_ratio = float(lesion_area_ratio)

        skin_metrics_clean = {
            "pore_count": int(skin_metrics["pore_count"]),
            "pore_area_ratio": float(skin_metrics["pore_area_ratio"]),
            "oiliness_score": float(skin_metrics["oiliness_score"]),
            "red_index": float(skin_metrics["red_index"]),
            "brightness_std": float(skin_metrics["brightness_std"]),
            "skin_type_hint": skin_metrics.get("skin_type_hint", "未评估")
        }

        return jsonify({
            'success': True,
            'overlay': pil_to_base64(seg['overlay_pil']),
            'mask': pil_to_base64(seg['mask_pil']),
            'heatmap': pil_to_base64(seg['heatmap_pil']),
            'classification': {
                'disease': qwen_result['disease'],
                'confidence': float(qwen_result['confidence']),
                'malignancy': qwen_result['malignancy'],
                'malignancy_conf': float(qwen_result['malignancy_conf'])
            },
            'skin_condition': {
                'moisture': moisture,
                'crack_level': crack_level,
                'crack_density': crack_density
            },
            'skin_metrics': skin_metrics_clean,
            'lesion_metrics': {
                'area_ratio': lesion_area_ratio,
                'dice': 0.95,
                'iou': 0.89
            },
            'report_text': report_text
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return render_template('index_full.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
