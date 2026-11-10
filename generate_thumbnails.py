import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FONT_PATH = "C:/Windows/Fonts/malgunbd.ttf"
FONT_REG_PATH = "C:/Windows/Fonts/malgun.ttf"

def get_font(size, bold=True):
    path = FONT_PATH if bold else FONT_REG_PATH
    return ImageFont.truetype(path, size)

def draw_text_with_outline(draw, pos, text, font, fill=(255, 255, 255), outline_color=(0, 0, 0), outline_width=3, anchor="lt"):
    x, y = pos
    # draw outline
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx*dx + dy*dy <= outline_width*outline_width:
                draw.text((x + dx, y + dy), text, font=font, fill=outline_color, anchor=anchor)
    # draw text
    draw.text((x, y), text, font=font, fill=fill, anchor=anchor)

def create_rounded_badge(draw, bbox, bg_color, border_color=None, border_width=2, radius=12):
    x1, y1, x2, y2 = bbox
    draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=bg_color, outline=border_color, width=border_width)

def extract_viewport(frame_path, zoom=1.0):
    img = Image.open(frame_path)
    # 720x1280 image -> viewport is roughly y=240..960 (720x720)
    vp = img.crop((0, 240, 720, 960))
    if zoom > 1.0:
        w, h = vp.size
        crop_w, crop_h = int(w / zoom), int(h / zoom)
        left = (w - crop_w) // 2
        top = (h - crop_h) // 2
        vp = vp.crop((left, top, left + crop_w, top + crop_h))
    return vp

# ==========================================
# 1. Shorts Thumbnail (9:16 - 1080x1920)
# ==========================================
def make_shorts_split_thumbnail():
    w, h = 1080, 1920
    canvas = Image.new("RGB", (w, h), (10, 12, 16))
    
    # 0세대 crop & 2500세대 crop
    vp0 = extract_viewport("gen0_f_135.jpg", zoom=1.15)
    vp2500 = extract_viewport("gen2500_f_120.jpg", zoom=1.15)
    
    # Resize to fit top and bottom
    # Top section height: 860, bottom section height: 860
    # Center banner: 200
    top_h = 860
    bot_h = 860
    
    # Crop to 1080 x top_h
    def fit_and_resize(im, target_w, target_h):
        # Resize keeping aspect ratio, crop center
        scale = max(target_w / im.width, target_h / im.height)
        new_w, new_h = int(im.width * scale), int(im.height * scale)
        res = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        return res.crop((left, top, left + target_w, top + target_h))
    
    top_img = fit_and_resize(vp0, w, top_h)
    bot_img = fit_and_resize(vp2500, w, bot_h)
    
    # Red-tint vignette on top (fail), Cyan-tint on bottom (success)
    # Add top image
    canvas.paste(top_img, (0, 0))
    # Add bottom image
    canvas.paste(bot_img, (0, h - bot_h))
    
    draw = ImageDraw.Draw(canvas)
    
    # ---------------- Top Overlay (0세대) ----------------
    # Subtle dark overlay on top for badges
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    ov_draw.rectangle([0, 0, w, 160], fill=(0, 0, 0, 160))
    ov_draw.rectangle([0, top_h - 100, w, top_h], fill=(0, 0, 0, 180))
    
    # Top badge: [ GEN 0 ] 태어나자마자 0초 컷 안면 슬라이딩
    ov_draw.rounded_rectangle([40, 45, 230, 115], radius=16, fill=(230, 50, 50, 255), outline=(255, 120, 120, 255), width=3)
    ov_draw.rounded_rectangle([250, 45, 660, 115], radius=16, fill=(20, 20, 30, 220), outline=(255, 70, 70, 200), width=2)
    
    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    
    f_badge = get_font(38, bold=True)
    f_sub = get_font(34, bold=True)
    draw.text((135, 80), "GEN 0", font=f_badge, fill=(255, 255, 255), anchor="mm")
    draw.text((455, 80), "0초 컷 대참사 ㅋㅋㅋ", font=f_sub, fill=(255, 230, 230), anchor="mm")
    
    # Top speech/caption
    f_top_punch = get_font(44, bold=True)
    draw_text_with_outline(draw, (w // 2, top_h - 50), "💥 걸음마 시켰더니 바닥 청소 중", f_top_punch, 
                           fill=(255, 140, 60), outline_color=(0, 0, 0), outline_width=5, anchor="mm")
    
    # ---------------- Center Dividing Banner ----------------
    # Height 860 to 1060 (200px)
    banner_y1 = 860
    banner_y2 = 1060
    
    # Glowing border gradient
    draw.rectangle([0, banner_y1 - 6, w, banner_y2 + 6], fill=(255, 190, 0))
    draw.rectangle([0, banner_y1 - 2, w, banner_y2 + 2], fill=(20, 22, 28))
    draw.rectangle([0, banner_y1, w, banner_y2], fill=(12, 14, 20))
    
    f_center_top = get_font(38, bold=True)
    f_center_main = get_font(54, bold=True)
    
    # Tag line
    draw_text_with_outline(draw, (w // 2, banner_y1 + 45), "🚨 AI에게 걷는 법 안 가르쳐줌 🚨", f_center_top, 
                           fill=(255, 225, 60), outline_color=(0, 0, 0), outline_width=4, anchor="mm")
    # Main hook line
    draw_text_with_outline(draw, (w // 2, banner_y1 + 125), "250만 번 구르고 생긴 기적 😱", f_center_main, 
                           fill=(255, 255, 255), outline_color=(0, 0, 0), outline_width=6, anchor="mm")
    
    # ---------------- Bottom Overlay (2500세대) ----------------
    bot_overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    bov_draw = ImageDraw.Draw(bot_overlay)
    bov_draw.rectangle([0, banner_y2, w, banner_y2 + 140], fill=(0, 0, 0, 160))
    bov_draw.rectangle([0, h - 160, w, h], fill=(0, 0, 0, 190))
    
    # Bottom badge: [ GEN 2500 ] 시속 5.6km 인간 초월 질주
    bov_draw.rounded_rectangle([40, banner_y2 + 40, 290, banner_y2 + 110], radius=16, 
                               fill=(150, 40, 255, 255), outline=(200, 130, 255, 255), width=3)
    bov_draw.rounded_rectangle([310, banner_y2 + 40, 770, banner_y2 + 110], radius=16, 
                               fill=(20, 20, 35, 220), outline=(170, 70, 255, 200), width=2)
    
    canvas = Image.alpha_composite(canvas.convert("RGBA"), bot_overlay).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    
    draw.text((165, banner_y2 + 75), "GEN 2500", font=f_badge, fill=(255, 255, 255), anchor="mm")
    draw.text((540, banner_y2 + 75), "🏆 마라톤 러너 완벽 탄생", font=f_sub, fill=(230, 210, 255), anchor="mm")
    
    # Bottom speech/caption
    f_bot_punch = get_font(48, bold=True)
    draw_text_with_outline(draw, (w // 2, h - 80), "🔥 250만 걸음 뒤... 완벽한 질주!", f_bot_punch, 
                           fill=(0, 240, 255), outline_color=(0, 0, 0), outline_width=5, anchor="mm")
    
    canvas.save("thumbnail_shorts_split_9x16.png", quality=98)
    print("Saved: thumbnail_shorts_split_9x16.png")

# ==========================================
# 2. YouTube Standard Thumbnail (16:9 - 1920x1080)
# ==========================================
def make_youtube_16x9_thumbnail():
    w, h = 1920, 1080
    canvas = Image.new("RGB", (w, h), (10, 12, 16))
    
    vp0 = extract_viewport("gen0_f_135.jpg", zoom=1.1)
    vp2500 = extract_viewport("gen2500_f_120.jpg", zoom=1.1)
    
    # Half width = 960 each
    half_w = 960
    
    def fit_half(im):
        scale = max(half_w / im.width, h / im.height)
        new_w, new_h = int(im.width * scale), int(im.height * scale)
        res = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
        left = (new_w - half_w) // 2
        top = (new_h - h) // 2
        return res.crop((left, top, left + half_w, top + h))
    
    left_img = fit_half(vp0)
    right_img = fit_half(vp2500)
    
    canvas.paste(left_img, (0, 0))
    canvas.paste(right_img, (half_w, 0))
    
    # Diagonal divider or center line
    draw = ImageDraw.Draw(canvas)
    
    # Center divider bar
    draw.line([(half_w - 4, 0), (half_w - 4, h)], fill=(255, 200, 0), width=8)
    draw.line([(half_w + 4, 0), (half_w + 4, h)], fill=(255, 255, 255), width=3)
    
    # Dark gradients at top and bottom for ultra-clear typography
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    
    # Top gradient
    for i in range(220):
        alpha = int(220 * (1 - i / 220))
        ov_draw.line([(0, i), (w, i)], fill=(0, 0, 0, alpha))
        
    # Bottom gradient
    for i in range(240):
        y = h - i
        alpha = int(230 * (1 - i / 240))
        ov_draw.line([(0, y), (w, y)], fill=(0, 0, 0, alpha))
        
    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    
    # Top Header Banner
    f_header = get_font(52, bold=True)
    draw_text_with_outline(draw, (w // 2, 70), "AI에게 걷는 법을 가르치지 않고 250만 번 시켰더니...", f_header,
                           fill=(255, 255, 255), outline_color=(0, 0, 0), outline_width=6, anchor="mm")
    
    # Left Label: 0세대 (꽈당)
    f_badge = get_font(42, bold=True)
    draw.rounded_rectangle([60, 160, 360, 240], radius=16, fill=(220, 40, 40), outline=(255, 120, 120), width=3)
    draw.text((210, 200), "0세대 : 대참사", font=f_badge, fill=(255, 255, 255), anchor="mm")
    
    f_left_sub = get_font(56, bold=True)
    draw_text_with_outline(draw, (480, h - 160), "0초 컷 안면 슬라이딩 ㅋㅋㅋ", f_left_sub,
                           fill=(255, 110, 80), outline_color=(0, 0, 0), outline_width=6, anchor="mm")
    
    # Right Label: 250만 세대 (질주)
    draw.rounded_rectangle([w - 460, 160, w - 60, 240], radius=16, fill=(140, 40, 255), outline=(200, 140, 255), width=3)
    draw.text((w - 260, 200), "250만 세대 : 완성", font=f_badge, fill=(255, 255, 255), anchor="mm")
    
    f_right_sub = get_font(56, bold=True)
    draw_text_with_outline(draw, (w - 480, h - 160), "시속 5.6km 인간 초월 질주!", f_right_sub,
                           fill=(0, 240, 255), outline_color=(0, 0, 0), outline_width=6, anchor="mm")
    
    # Center VS Badge
    vs_cx, vs_cy = half_w, h // 2
    draw.ellipse([vs_cx - 90, vs_cy - 90, vs_cx + 90, vs_cy + 90], fill=(255, 200, 0), outline=(255, 255, 255), width=6)
    draw.ellipse([vs_cx - 78, vs_cy - 78, vs_cx + 78, vs_cy + 78], fill=(15, 18, 24))
    f_vs = get_font(60, bold=True)
    draw.text((vs_cx, vs_cy), "VS", font=f_vs, fill=(255, 215, 0), anchor="mm")
    
    # Bottom Punchline
    f_punch = get_font(46, bold=True)
    draw_text_with_outline(draw, (w // 2, h - 60), "🧠 강화학습 PPO로 스스로 터득한 마라톤 보행 알고리즘", f_punch,
                           fill=(255, 230, 80), outline_color=(0, 0, 0), outline_width=5, anchor="mm")
    
    canvas.save("thumbnail_youtube_16x9.png", quality=98)
    print("Saved: thumbnail_youtube_16x9.png")

if __name__ == "__main__":
    make_shorts_split_thumbnail()
    make_youtube_16x9_thumbnail()
