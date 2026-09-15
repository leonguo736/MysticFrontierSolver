import re
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageGrab, ImageTk
from rapidfuzz import fuzz
import customtkinter as ctk

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Set theme and appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Uncomment and adjust if Tesseract is not in your system PATH:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def preprocess_for_ocr(img_crop: np.ndarray, invert: bool = False) -> np.ndarray:
    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, (0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    if invert:
        scaled = cv2.bitwise_not(scaled)
    _, thresh = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def evaluate_attribute_rule(line: str, dice: list[int]) -> tuple[bool, int, float, str]:
    text = line.lower().strip()
    mult_match = re.search(r"final multiplier:\s*\+?([\d\.]+)x?", text)
    flat_match = re.search(r"dice total:\s*([+-]?\d+)", text)

    multiplier = float(mult_match.group(1)) if mult_match else 0.0
    flat = int(flat_match.group(1)) if flat_match else 0

    dice_tokens = ["dice", "die", "roll"]
    has_dice_keyword = any(fuzz.partial_ratio(kw, text) >= 80 for kw in dice_tokens)

    if not has_dice_keyword:
        return True, flat, multiplier, "Passive condition (assumed True)"

    sum_match = re.search(r"add up to (\d+)\s*(?:or more|or higher|\+)?", text)
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

        # Global hotkey binding for paste
        self.bind("<Control-v>", lambda e: self.paste_image())
        self.bind("<Command-v>", lambda e: self.paste_image())

        self._build_ui()

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ----------------- Left Panel: Image Drop / Preview -----------------
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

        # Preview Container
        self.preview_label = ctk.CTkLabel(
            left_frame,
            text="No image loaded.\nPress Ctrl+V or click Paste above.",
            font=ctk.CTkFont(size=13),
            fg_color="#1e1e24",
            corner_radius=8
        )
        self.preview_label.grid(row=1, column=0, padx=16, pady=12, sticky="nsew")

        # ----------------- Right Panel: Detection & Breakdown -----------------
        right_frame = ctk.CTkFrame(self, corner_radius=12)
        right_frame.grid(row=0, column=1, padx=(0, 16), pady=16, sticky="nsew")
        right_frame.grid_columnconfigure(0, weight=1)

        # Big Result Banner
        self.result_banner = ctk.CTkLabel(
            right_frame,
            text="READY",
            font=ctk.CTkFont(size=22, weight="bold"),
            fg_color="#333333",
            corner_radius=8,
            height=50
        )
        self.result_banner.grid(row=0, column=0, padx=16, pady=(16, 12), sticky="ew")

        # Input Form Grid
        form_frame = ctk.CTkFrame(right_frame, fg_color="transparent")
        form_frame.grid(row=1, column=0, padx=16, pady=4, sticky="ew")
        form_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        ctk.CTkLabel(form_frame, text="Difficulty:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, sticky="w")
        self.diff_var = ctk.StringVar(value="0")
        self.diff_var.trace_add("write", lambda *args: self.recalculate())
        self.diff_entry = ctk.CTkEntry(form_frame, textvariable=self.diff_var, width=60, justify="center")
        self.diff_entry.grid(row=1, column=0, padx=(0, 8), pady=(2, 8), sticky="w")

        # Dice 1, 2, 3
        self.dice_vars = [ctk.StringVar(value="1") for _ in range(3)]
        for i, var in enumerate(self.dice_vars):
            var.trace_add("write", lambda *args: self.recalculate())
            ctk.CTkLabel(form_frame, text=f"Die #{i+1}:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=i+1, sticky="w")
            entry = ctk.CTkEntry(form_frame, textvariable=var, width=50, justify="center")
            entry.grid(row=1, column=i+1, padx=4, pady=(2, 8), sticky="w")

        # Score Breakdown Stats
        self.stats_label = ctk.CTkLabel(
            right_frame,
            text="Dice Sum: 0 | Flat Bonus: +0 | Multiplier: 1.00x\nFinal Calculated Score: 0",
            font=ctk.CTkFont(size=14, weight="bold"),
            justify="center"
        )
        self.stats_label.grid(row=2, column=0, padx=16, pady=8)

        # Attribute Lines Text Box
        ctk.CTkLabel(right_frame, text="Detected Attributes (Editable):", font=ctk.CTkFont(weight="bold")).grid(
            row=3, column=0, padx=16, sticky="w"
        )
        self.attr_textbox = ctk.CTkTextbox(right_frame, height=120, font=ctk.CTkFont(size=12))
        self.attr_textbox.grid(row=4, column=0, padx=16, pady=(4, 8), sticky="nsew")
        self.attr_textbox.bind("<KeyRelease>", lambda e: self.recalculate())

        # Log breakdown list
        self.log_textbox = ctk.CTkTextbox(right_frame, height=140, font=ctk.CTkFont(family="Consolas", size=11))
        self.log_textbox.grid(row=5, column=0, padx=16, pady=(0, 16), sticky="nsew")
        self.log_textbox.configure(state="disabled")

    def paste_image(self):
        grabbed = ImageGrab.grabclipboard()
        if not isinstance(grabbed, Image.Image):
            self.result_banner.configure(text="CLIPBOARD IS NOT AN IMAGE", fg_color="#b83232")
            return

        # Resize for preview thumbnail
        preview_img = grabbed.copy()
        preview_img.thumbnail((420, 420))
        ctk_img = ctk.CTkImage(light_image=preview_img, dark_image=preview_img, size=preview_img.size)
        self.preview_label.configure(image=ctk_img, text="")
        self.preview_label.image = ctk_img

        # Process OpenCV frame
        img = cv2.cvtColor(np.array(grabbed), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]

        # Tightly focused relative coordinates (y1, y2, x1, x2)
        rois = {
            # Difficulty: tightly on the '15' inside the crest
            "difficulty": img[int(h * 0.065):int(h * 0.150), int(w * 0.465):int(w * 0.535)],
            
            # Dice: tight center crop on just the numeral inside each isometric die
            "dice1":      img[int(h * 0.450):int(h * 0.585), int(w * 0.245):int(w * 0.310)],
            "dice2":      img[int(h * 0.450):int(h * 0.585), int(w * 0.465):int(w * 0.530)],
            "dice3":      img[int(h * 0.450):int(h * 0.585), int(w * 0.685):int(w * 0.750)],
            
            # Attributes box
            "attributes": img[int(h * 0.680):int(h * 0.810), int(w * 0.080):int(w * 0.850)],
        }

        digit_config = "--psm 7 -c tessedit_char_whitelist=0123456789"
        text_config = "--psm 6"

        # OCR Difficulty
        diff_raw = pytesseract.image_to_string(preprocess_for_ocr(rois["difficulty"]), config=digit_config).strip()
        self.diff_var.set(diff_raw if diff_raw.isdigit() else "0")

        # OCR Dice
        for idx, key in enumerate(["dice1", "dice2", "dice3"]):
            val_raw = pytesseract.image_to_string(preprocess_for_ocr(rois[key]), config=digit_config).strip()
            self.dice_vars[idx].set(val_raw if (val_raw.isdigit() and 1 <= int(val_raw) <= 6) else "1")

        # OCR Attributes
        attr_raw = pytesseract.image_to_string(preprocess_for_ocr(rois["attributes"]), config=text_config)
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

        # Update stats
        self.stats_label.configure(
            text=f"Base Sum: {dice_sum}  |  Flat: {total_flat:+d}  |  Multiplier: {total_mult:.2f}x\n"
                 f"Final Score: ⌊({dice_sum} {total_flat:+d}) × {total_mult:.2f}⌋ = {final_score}"
        )

        # Update Banner
        if target_diff > 0:
            if final_score >= target_diff:
                self.result_banner.configure(
                    text=f"PASSED! ({final_score} / {target_diff})",
                    fg_color="#2e7d32"  # Green
                )
            else:
                self.result_banner.configure(
                    text=f"FAILED - REROLL RECOMMENDED ({final_score} / {target_diff})",
                    fg_color="#c62828"  # Red
                )
        else:
            self.result_banner.configure(text=f"SCORE: {final_score}", fg_color="#333333")

        # Update Log Box
        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("1.0", "end")
        self.log_textbox.insert("1.0", "\n\n".join(applied_log) if applied_log else "No potentials detected.")
        self.log_textbox.configure(state="disabled")


if __name__ == "__main__":
    app = MysticFrontierApp()
    app.mainloop()