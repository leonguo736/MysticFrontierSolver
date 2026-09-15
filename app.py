import re
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageGrab, ImageTk
from rapidfuzz import fuzz
import customtkinter as ctk
import os

# Set theme and appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Uncomment and adjust if Tesseract is not in your system PATH:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def preprocess_digit_crop(img_crop: np.ndarray) -> np.ndarray:
    """Isolates bright white/cream numbers, inverts, and adds a white border."""
    hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
    
    # Isolate the bright white/cream numerals
    mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 90, 255]))
    
    # Scale up using INTER_NEAREST to keep crisp edges
    scaled = cv2.resize(mask, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_NEAREST)
    
    # Invert to black numbers on white background
    inverted = cv2.bitwise_not(scaled)
    
    # CRITICAL: Pad with 25px of solid white on all sides so Tesseract doesn't fail
    padded = cv2.copyMakeBorder(
        inverted, 
        top=25, bottom=25, left=25, right=25, 
        borderType=cv2.BORDER_CONSTANT, 
        value=[255, 255, 255]
    )
    return padded


def preprocess_text_crop(img_crop: np.ndarray) -> np.ndarray:
    """Grayscale + Otsu thresholding optimized for attribute text blocks."""
    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, (0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def evaluate_attribute_rule(line: str, dice: list[int]) -> tuple[bool, int, float, str]:
    text = line.lower().strip()
    mult_match = re.search(r"final multiplier:\s*\+?([\d\.]+)x?", text)
    flat_match = re.search(r"dice total:\s*([+-]?\d+)", text)

    multiplier = float(mult_match.group(1)) if mult_match else 0.0
    flat = int(flat_match.group(1)) if flat_match else 0

    dice_tokens = ["dice", "die", "roll", "ait"]
    has_dice_keyword = any(fuzz.partial_ratio(kw, text) >= 80 for kw in dice_tokens)

    if not has_dice_keyword:
        return True, flat, multiplier, "Passive condition (assumed True)"

    # Sum pattern matches variations including OCR artifacts like "ait to"
    sum_match = re.search(r"(?:add\s*up|ait|add|all)\s*(?:to|up\s*to)?\s*(\d+)\s*(?:or more|or higher|\+)?", text)
    if sum_match:
        threshold = int(sum_match.group(1))
        dice_sum = sum(dice)
        passed = dice_sum >= threshold
        reason = f"Sum {dice_sum} >= {threshold}" if passed else f"Sum {dice_sum} < {threshold}"
        return passed, flat, multiplier, reason

    index_map = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2}
    for word, idx in index_map.items():
        if word in text:
            target_val = dice[idx]
            if "odd" in text:
                passed = (target_val % 2 != 0)
                return passed, flat, multiplier, f"Die #{idx+1} ({target_val}) is {'odd' if passed else 'even'}"
            if "even" in text:
                passed = (target_val % 2 == 0)
                return passed, flat, multiplier, f"Die #{idx+1} ({target_val}) is {'even' if passed else 'odd'}"

            comp_match = re.search(r"(?:greater than|at least|or more than|>=)\s*(\d+)", text)
            if comp_match:
                thresh = int(comp_match.group(1))
                passed = target_val >= thresh
                return passed, flat, multiplier, f"Die #{idx+1} ({target_val}) >= {thresh}"

    if "all" in text and "even" in text:
        passed = all(d % 2 == 0 for d in dice)
        return passed, flat, multiplier, "All dice even" if passed else "Not all dice even"

    if "all" in text and "odd" in text:
        passed = all(d % 2 != 0 for d in dice)
        return passed, flat, multiplier, "All dice odd" if passed else "Not all dice odd"

    if any(k in text for k in ["identical", "same", "triple", "match"]):
        passed = (len(set(dice)) == 1)
        return passed, flat, multiplier, "All dice match" if passed else "Dice do not match"

    return False, flat, multiplier, "Unrecognized dice rule"


class MysticFrontierApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("MapleStory - Mystic Frontier Dice Solver")
        self.geometry("960x640")
        self.minsize(800, 560)

        self.bind("<Control-v>", lambda e: self.paste_image())
        self.bind("<Command-v>", lambda e: self.paste_image())

        self._build_ui()

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Left Panel: Image Drop / Preview
        left_frame = ctk.CTkFrame(self, corner_radius=12)
        left_frame.grid(row=0, column=0, padx=16, pady=16, sticky="nsew")
        left_frame.grid_rowconfigure(1, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)

        paste_btn = ctk.CTkButton(
            left_frame,
            text="📋 Paste Screenshot from Clipboard (Ctrl+V)",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
            command=self.paste_image
        )
        paste_btn.grid(row=0, column=0, padx=16, pady=(16, 8), sticky="ew")

        self.preview_label = ctk.CTkLabel(
            left_frame,
            text="No image loaded.\nPress Ctrl+V or click Paste above.",
            font=ctk.CTkFont(size=13),
            fg_color="#1e1e24",
            corner_radius=8
        )
        self.preview_label.grid(row=1, column=0, padx=16, pady=12, sticky="nsew")

        # Right Panel: Output & Controls
        right_frame = ctk.CTkFrame(self, corner_radius=12)
        right_frame.grid(row=0, column=1, padx=(0, 16), pady=16, sticky="nsew")
        right_frame.grid_columnconfigure(0, weight=1)

        self.result_banner = ctk.CTkLabel(
            right_frame,
            text="READY",
            font=ctk.CTkFont(size=22, weight="bold"),
            fg_color="#333333",
            corner_radius=8,
            height=50
        )
        self.result_banner.grid(row=0, column=0, padx=16, pady=(16, 12), sticky="ew")

        form_frame = ctk.CTkFrame(right_frame, fg_color="transparent")
        form_frame.grid(row=1, column=0, padx=16, pady=4, sticky="ew")
        form_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        ctk.CTkLabel(form_frame, text="Difficulty:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, sticky="w")
        self.diff_var = ctk.StringVar(value="0")
        self.diff_var.trace_add("write", lambda *args: self.recalculate())
        self.diff_entry = ctk.CTkEntry(form_frame, textvariable=self.diff_var, width=60, justify="center")
        self.diff_entry.grid(row=1, column=0, padx=(0, 8), pady=(2, 8), sticky="w")

        self.dice_vars = [ctk.StringVar(value="1") for _ in range(3)]
        for i, var in enumerate(self.dice_vars):
            var.trace_add("write", lambda *args: self.recalculate())
            ctk.CTkLabel(form_frame, text=f"Die #{i+1}:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=i+1, sticky="w")
            entry = ctk.CTkEntry(form_frame, textvariable=var, width=50, justify="center")
            entry.grid(row=1, column=i+1, padx=4, pady=(2, 8), sticky="w")

        self.stats_label = ctk.CTkLabel(
            right_frame,
            text="Base Sum: 0  |  Flat: +0  |  Multiplier: 1.00x\nFinal Calculated Score: 0",
            font=ctk.CTkFont(size=14, weight="bold"),
            justify="center"
        )
        self.stats_label.grid(row=2, column=0, padx=16, pady=8)

        ctk.CTkLabel(right_frame, text="Detected Attributes (Editable):", font=ctk.CTkFont(weight="bold")).grid(
            row=3, column=0, padx=16, sticky="w"
        )
        self.attr_textbox = ctk.CTkTextbox(right_frame, height=120, font=ctk.CTkFont(size=12))
        self.attr_textbox.grid(row=4, column=0, padx=16, pady=(4, 8), sticky="nsew")
        self.attr_textbox.bind("<KeyRelease>", lambda e: self.recalculate())

        self.log_textbox = ctk.CTkTextbox(right_frame, height=140, font=ctk.CTkFont(family="Consolas", size=11))
        self.log_textbox.grid(row=5, column=0, padx=16, pady=(0, 16), sticky="nsew")
        self.log_textbox.configure(state="disabled")

    def paste_image(self):
        grabbed = ImageGrab.grabclipboard()
        if not isinstance(grabbed, Image.Image):
            self.result_banner.configure(text="CLIPBOARD IS NOT AN IMAGE", fg_color="#b83232")
            return

        img = cv2.cvtColor(np.array(grabbed), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]

        # Create debug output directory
        debug_dir = "debug_crops"
        os.makedirs(debug_dir, exist_ok=True)

        # Coordinates: (y1, y2, x1, x2)
        # Recalibrated ROIs: shifted up and widened
        roi_defs = {
            "difficulty": (int(h * 0.055), int(h * 0.155), int(w * 0.455), int(w * 0.545)),
            "dice1":      (int(h * 0.380), int(h * 0.560), int(w * 0.230), int(w * 0.320)),
            "dice2":      (int(h * 0.380), int(h * 0.560), int(w * 0.450), int(w * 0.540)),
            "dice3":      (int(h * 0.380), int(h * 0.560), int(w * 0.670), int(w * 0.760)),
            "attributes": (int(h * 0.680), int(h * 0.810), int(w * 0.080), int(w * 0.850)),
        }

        # Extract crops
        rois = {name: img[y1:y2, x1:x2] for name, (y1, y2, x1, x2) in roi_defs.items()}

        # --- DEBUG STEP: Save raw crops and preprocessed images to disk ---
        for name, crop in rois.items():
            cv2.imwrite(os.path.join(debug_dir, f"{name}_raw.png"), crop)
            if name in ["difficulty", "dice1", "dice2", "dice3"]:
                proc = preprocess_digit_crop(crop)
                cv2.imwrite(os.path.join(debug_dir, f"{name}_proc.png"), proc)

        # --- DEBUG STEP: Draw bounding boxes onto the UI preview ---
        debug_preview = img.copy()
        for name, (y1, y2, x1, x2) in roi_defs.items():
            color = (0, 0, 255) if "dice" in name else (0, 255, 0)
            cv2.rectangle(debug_preview, (x1, y1), (x2, y2), color, 2)
            cv2.putText(debug_preview, name, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Render debug preview on the left panel
        preview_rgb = cv2.cvtColor(debug_preview, cv2.COLOR_BGR2RGB)
        preview_pil = Image.fromarray(preview_rgb)
        preview_pil.thumbnail((420, 420))
        ctk_img = ctk.CTkImage(light_image=preview_pil, dark_image=preview_pil, size=preview_pil.size)
        self.preview_label.configure(image=ctk_img, text="")
        self.preview_label.image = ctk_img

        preview_img = grabbed.copy()
        preview_img.thumbnail((420, 420))
        ctk_img = ctk.CTkImage(light_image=preview_img, dark_image=preview_img, size=preview_img.size)
        self.preview_label.configure(image=ctk_img, text="")
        self.preview_label.image = ctk_img

        img = cv2.cvtColor(np.array(grabbed), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]

        # Tightly cropped ROIs targeting the exact glyph coordinates
        rois = {
            "difficulty": img[int(h * 0.065):int(h * 0.150), int(w * 0.465):int(w * 0.535)],
            "dice1":      img[int(h * 0.450):int(h * 0.585), int(w * 0.245):int(w * 0.310)],
            "dice2":      img[int(h * 0.450):int(h * 0.585), int(w * 0.465):int(w * 0.530)],
            "dice3":      img[int(h * 0.450):int(h * 0.585), int(w * 0.685):int(w * 0.750)],
            "attributes": img[int(h * 0.680):int(h * 0.810), int(w * 0.080):int(w * 0.850)],
        }

        # OCR Configurations
        diff_config = "--psm 8 -c tessedit_char_whitelist=0123456789"
        dice_config = "--psm 10 -c tessedit_char_whitelist=123456"
        text_config = "--psm 6"

        # 1. OCR Difficulty (Contains 1 or 2 digits -> use PSM 7 or 8, NOT PSM 10)
        diff_config = "--psm 7 -c tessedit_char_whitelist=0123456789"
        diff_proc = preprocess_digit_crop(rois["difficulty"])
        diff_raw = pytesseract.image_to_string(diff_proc, config=diff_config).strip()
        
        # Keep only digits without regex
        clean_diff = "".join([c for c in diff_raw if c.isdigit()])
        self.diff_var.set(clean_diff if clean_diff else "0")

        # 2. OCR Dice (Single isolated digit 1-6 -> PSM 10 with padded border)
        dice_config = "--psm 10 -c tessedit_char_whitelist=123456"
        for idx, key in enumerate(["dice1", "dice2", "dice3"]):
            dice_proc = preprocess_digit_crop(rois[key])
            val_raw = pytesseract.image_to_string(dice_proc, config=dice_config).strip()
            
            valid_digits = [c for c in val_raw if c in "123456"]
            self.dice_vars[idx].set(valid_digits[0] if valid_digits else "1")

        # 3. OCR Attributes
        attr_proc = preprocess_text_crop(rois["attributes"])
        attr_raw = pytesseract.image_to_string(attr_proc, config=text_config)
        clean_lines = [line.strip() for line in attr_raw.splitlines() if len(line.strip()) > 3]

        self.attr_textbox.delete("1.0", "end")
        self.attr_textbox.insert("1.0", "\n".join(clean_lines))

        self.recalculate()

    def recalculate(self):
        try:
            target_diff = int(self.diff_var.get())
        except ValueError:
            target_diff = 0

        dice_vals = []
        for var in self.dice_vars:
            try:
                dice_vals.append(int(var.get()))
            except ValueError:
                dice_vals.append(1)

        raw_attr_text = self.attr_textbox.get("1.0", "end")
        lines = [l.strip() for l in raw_attr_text.splitlines() if len(l.strip()) > 2]

        dice_sum = sum(dice_vals)
        total_flat = 0
        total_mult = 1.0
        applied_log = []

        for line in lines:
            active, flat, mult, reason = evaluate_attribute_rule(line, dice_vals)
            status_icon = "✓" if active else "✗"
            mod_str = f"+{mult:.1f}x" if mult else f"{flat:+d} flat" if flat else "0"
            applied_log.append(f"[{status_icon}] [{mod_str}] {line}\n    ↳ {reason}")
            if active:
                total_flat += flat
                total_mult += mult

        final_score = int(np.floor((dice_sum + total_flat) * total_mult))

        self.stats_label.configure(
            text=f"Base Sum: {dice_sum}  |  Flat: {total_flat:+d}  |  Multiplier: {total_mult:.2f}x\n"
                 f"Final Score: ⌊({dice_sum} {total_flat:+d}) × {total_mult:.2f}⌋ = {final_score}"
        )

        if target_diff > 0:
            if final_score >= target_diff:
                self.result_banner.configure(
                    text=f"PASSED! ({final_score} / {target_diff})",
                    fg_color="#2e7d32"
                )
            else:
                self.result_banner.configure(
                    text=f"FAILED - REROLL RECOMMENDED ({final_score} / {target_diff})",
                    fg_color="#c62828"
                )
        else:
            self.result_banner.configure(text=f"SCORE: {final_score}", fg_color="#333333")

        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("1.0", "end")
        self.log_textbox.insert("1.0", "\n\n".join(applied_log) if applied_log else "No potentials detected.")
        self.log_textbox.configure(state="disabled")


if __name__ == "__main__":
    app = MysticFrontierApp()
    app.mainloop()