import os
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageGrab, ImageTk
from rapidfuzz import fuzz
import customtkinter as ctk

# Set theme and appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Uncomment and adjust if Tesseract is not in your system PATH:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

TEMPLATE_DIR = "templates/dice"
os.makedirs(TEMPLATE_DIR, exist_ok=True)

HOLY_DICE_COLUMNS = ["Grey", "Blue", "Purple", "Orange", "Green"]
HOLY_DICE_ROWS = ["+", "-x", "x", "+x"]
HOLY_DICE_COLUMN_COLORS = ["#a6a6a6", "#5b9bd5", "#b07ad1", "#e89545", "#64ad72"]


def holy_dice_effect(row: str, column_index: int) -> tuple[int, float]:
    """Return the flat and multiplier contributions for one holy-dice cell."""
    if row == "+":
        return 3 * (column_index + 1), 0.0
    if row == "-x":
        return -1, 1.6 + 0.2 * column_index
    if row == "x":
        return 0, 1.4 + 0.2 * column_index
    return 1, 1.4 + 0.2 * column_index


def load_dice_templates() -> dict[int, np.ndarray]:
    """Loads reference binary glyphs for dice numbers 1-6."""
    templates = {}
    if os.path.exists(TEMPLATE_DIR):
        for fname in os.listdir(TEMPLATE_DIR):
            if fname.endswith(".png"):
                try:
                    val = int(os.path.splitext(fname)[0])
                    img = cv2.imread(os.path.join(TEMPLATE_DIR, fname), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        templates[val] = img
                except ValueError:
                    continue
    return templates


def match_digit_template(binary_crop: np.ndarray, templates_dict: dict) -> int:
    """Finds the best matching digit via normalized cross-correlation."""
    if not templates_dict:
        return 1

    # Extract tight bounding box of the black glyph
    black_pixels = np.where(binary_crop == 0)
    if len(black_pixels[0]) == 0:
        return 1

    y_min, y_max = np.min(black_pixels[0]), np.max(black_pixels[0])
    x_min, x_max = np.min(black_pixels[1]), np.max(black_pixels[1])
    glyph = binary_crop[y_min:y_max + 1, x_min:x_max + 1]

    best_digit = 1
    highest_score = -1.0

    for digit, tmpl in templates_dict.items():
        t_black = np.where(tmpl == 0)
        if len(t_black[0]) == 0:
            continue
        t_ymin, t_ymax = np.min(t_black[0]), np.max(t_black[0])
        t_xmin, t_xmax = np.min(t_black[1]), np.max(t_black[1])
        clean_tmpl = tmpl[t_ymin:t_ymax + 1, t_xmin:t_xmax + 1]

        # Resize template to match current glyph dimensions
        resized_tmpl = cv2.resize(clean_tmpl, (glyph.shape[1], glyph.shape[0]), interpolation=cv2.INTER_AREA)

        # Cross-correlation matching
        res = cv2.matchTemplate(glyph, resized_tmpl, cv2.TM_CCOEFF_NORMED)
        score = res[0][0]

        if score > highest_score:
            highest_score = score
            best_digit = digit

    return best_digit


def preprocess_dice_crop(img_crop: np.ndarray) -> np.ndarray:
    """Isolates dice numerals, closes internal gaps, and produces solid black glyphs."""
    hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 140]), np.array([180, 110, 255]))
    scaled = cv2.resize(mask, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(scaled, cv2.MORPH_CLOSE, kernel)

    # Fill internal contours so numbers become solid glyphs
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(closed)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

    inverted = cv2.bitwise_not(filled)
    return cv2.copyMakeBorder(inverted, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=[255, 255, 255])


def preprocess_difficulty_crop(img_crop: np.ndarray) -> np.ndarray:
    """Preprocesses difficulty badge without merging digits together."""
    hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 160]), np.array([180, 80, 255]))

    # Smooth cubic upscale + blur to prevent jagged pixel stairs
    scaled = cv2.resize(mask, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(scaled, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    inverted = cv2.bitwise_not(thresh)
    return cv2.copyMakeBorder(inverted, 35, 35, 35, 35, cv2.BORDER_CONSTANT, value=[255, 255, 255])

def clean_and_crop_tight(diff_proc: np.ndarray) -> np.ndarray:
    """Crops tight to the black pixels to eliminate misleading whitespace."""
    black_pixels = np.where(diff_proc == 0)
    if len(black_pixels[0]) == 0:
        return diff_proc
    
    y_min, y_max = np.min(black_pixels[0]), np.max(black_pixels[0])
    x_min, x_max = np.min(black_pixels[1]), np.max(black_pixels[1])
    tight = diff_proc[y_min:y_max+1, x_min:x_max+1]
    
    # Pad with only 12px of margin (not 35px)
    return cv2.copyMakeBorder(tight, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=[255, 255, 255])


def preprocess_text_crop(img_crop: np.ndarray) -> np.ndarray:
    """Standard thresholding for attribute text area."""
    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, (0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def extract_numeric_value(segment: str) -> float:
    """Extracts first valid number from a string without regex."""
    clean_chars = [c if (c.isdigit() or c in ".-") else " " for c in segment]
    tokens = "".join(clean_chars).split()
    for token in tokens:
        try:
            return float(token)
        except ValueError:
            continue
    return 0.0


def evaluate_attribute_rule(line: str, dice: list[int]) -> tuple[bool, int, float, str]:
    """Evaluates attribute line without regex."""
    text = line.lower().strip()
    multiplier = 0.0
    flat = 0

    if "final multiplier:" in text:
        multiplier = extract_numeric_value(text.split("final multiplier:")[1])

    if "dice total:" in text:
        flat = int(extract_numeric_value(text.split("dice total:")[1]))

    # Check if dice are referenced
    dice_keywords = ["dice", "die", "roll", "ait"]
    if not any(fuzz.partial_ratio(kw, text) >= 80 for kw in dice_keywords):
        return True, flat, multiplier, "Passive condition (assumed True)"

    dice_sum = sum(dice)

    # Condition: Sum of all dice
    for marker in ["add up to", "ait to", "add to"]:
        if marker in text:
            threshold = int(extract_numeric_value(text.split(marker)[1]))
            passed = dice_sum >= threshold
            reason = f"Sum {dice_sum} >= {threshold}" if passed else f"Sum {dice_sum} < {threshold}"
            return passed, flat, multiplier, reason

    # Condition: Specific die position checks
    indexed_checks = [
        (["first", "1st"], 0),
        (["second", "2nd"], 1),
        (["third", "3rd"], 2),
    ]
    for aliases, idx in indexed_checks:
        if any(alias in text for alias in aliases):
            target_val = dice[idx]
            if "odd" in text:
                passed = (target_val % 2 != 0)
                return passed, flat, multiplier, f"Die #{idx+1} ({target_val}) is {'odd' if passed else 'even'}"
            if "even" in text:
                passed = (target_val % 2 == 0)
                return passed, flat, multiplier, f"Die #{idx+1} ({target_val}) is {'even' if passed else 'odd'}"

            for comp in ["greater than", "at least", "or more than", ">="]:
                if comp in text:
                    threshold = int(extract_numeric_value(text.split(comp)[1]))
                    passed = target_val >= threshold
                    return passed, flat, multiplier, f"Die #{idx+1} ({target_val}) >= {threshold}"

    # Condition: Parity across all dice
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
        self.geometry("1080x800")
        self.minsize(900, 700)

        self.dice_templates = load_dice_templates()
        self.last_dice_procs = [None, None, None]

        self.bind("<Control-v>", lambda e: self.paste_image())
        self.bind("<Command-v>", lambda e: self.paste_image())

        self._build_ui()

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Left Panel
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

        # Right Panel
        right_frame = ctk.CTkFrame(self, corner_radius=12)
        right_frame.grid(row=0, column=1, padx=(0, 16), pady=16, sticky="nsew")
        right_frame.grid_columnconfigure(0, weight=1)
        right_frame.grid_rowconfigure((5, 6), weight=1)

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
            var.trace_add("write", lambda *args, idx=i: self._on_dice_manual_change(idx))
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

        holy_frame = ctk.CTkFrame(right_frame, fg_color="transparent")
        holy_frame.grid(row=3, column=0, padx=16, pady=(4, 8), sticky="ew")
        holy_frame.grid_columnconfigure((1, 2, 3, 4, 5), weight=1)
        ctk.CTkLabel(
            holy_frame,
            text="Holy Dice Potentials",
            font=ctk.CTkFont(weight="bold")
        ).grid(row=0, column=0, columnspan=6, sticky="w", pady=(0, 4))

        for column_index, column_name in enumerate(HOLY_DICE_COLUMNS):
            ctk.CTkLabel(
                holy_frame,
                text=column_name,
                text_color=HOLY_DICE_COLUMN_COLORS[column_index],
                font=ctk.CTkFont(size=11, weight="bold")
            ).grid(row=1, column=column_index + 1, padx=2, pady=2)

        self.holy_dice_vars = {}
        for row_index, row_name in enumerate(HOLY_DICE_ROWS):
            ctk.CTkLabel(
                holy_frame,
                text=row_name,
                font=ctk.CTkFont(size=12, weight="bold")
            ).grid(row=row_index + 2, column=0, padx=(0, 8), pady=3, sticky="w")

            for column_index in range(len(HOLY_DICE_COLUMNS)):
                flat, multiplier = holy_dice_effect(row_name, column_index)
                if row_name == "+":
                    cell_text = f"+{flat}"
                elif row_name == "-x":
                    cell_text = f"-1 / {multiplier:.1f}x"
                elif row_name == "x":
                    cell_text = f"{multiplier:.1f}x"
                else:
                    cell_text = f"+1 / {multiplier:.1f}x"

                variable = ctk.BooleanVar(value=False)
                self.holy_dice_vars[(row_name, column_index)] = variable
                ctk.CTkCheckBox(
                    holy_frame,
                    text=cell_text,
                    variable=variable,
                    command=self.recalculate,
                    width=76,
                    checkbox_width=16,
                    checkbox_height=16,
                    font=ctk.CTkFont(size=10),
                    fg_color=HOLY_DICE_COLUMN_COLORS[column_index],
                    border_color=HOLY_DICE_COLUMN_COLORS[column_index],
                    checkmark_color="#111111"
                ).grid(row=row_index + 2, column=column_index + 1, padx=2, pady=3, sticky="w")

        ctk.CTkLabel(right_frame, text="Detected Attributes (Editable):", font=ctk.CTkFont(weight="bold")).grid(
            row=4, column=0, padx=16, sticky="w"
        )
        self.attr_textbox = ctk.CTkTextbox(right_frame, height=105, font=ctk.CTkFont(size=12))
        self.attr_textbox.grid(row=5, column=0, padx=16, pady=(4, 8), sticky="nsew")
        self.attr_textbox.bind("<KeyRelease>", lambda e: self.recalculate())

        self.log_textbox = ctk.CTkTextbox(right_frame, height=110, font=ctk.CTkFont(family="Consolas", size=11))
        self.log_textbox.grid(row=6, column=0, padx=16, pady=(0, 16), sticky="nsew")
        self.log_textbox.configure(state="disabled")

    def _on_dice_manual_change(self, idx: int):
        """If user corrects a die manually, save the crop as a reference template."""
        val_str = self.dice_vars[idx].get().strip()
        if val_str in ["1", "2", "3", "4", "5", "6"] and self.last_dice_procs[idx] is not None:
            val = int(val_str)
            target_path = os.path.join(TEMPLATE_DIR, f"{val}.png")
            cv2.imwrite(target_path, self.last_dice_procs[idx])
            self.dice_templates[val] = self.last_dice_procs[idx]
        self.recalculate()

    def paste_image(self):
        grabbed = ImageGrab.grabclipboard()
        if not isinstance(grabbed, Image.Image):
            self.result_banner.configure(text="CLIPBOARD IS NOT AN IMAGE", fg_color="#b83232")
            return

        preview_img = grabbed.copy()
        preview_img.thumbnail((420, 420))
        ctk_img = ctk.CTkImage(light_image=preview_img, dark_image=preview_img, size=preview_img.size)
        self.preview_label.configure(image=ctk_img, text="")
        self.preview_label.image = ctk_img

        img = cv2.cvtColor(np.array(grabbed), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]

        os.makedirs("debug_crops", exist_ok=True)

        rois = {
            "difficulty": img[int(h * 0.055):int(h * 0.155), int(w * 0.455):int(w * 0.545)],
            "dice1":      img[int(h * 0.380):int(h * 0.560), int(w * 0.230):int(w * 0.320)],
            "dice2":      img[int(h * 0.380):int(h * 0.560), int(w * 0.450):int(w * 0.540)],
            "dice3":      img[int(h * 0.380):int(h * 0.560), int(w * 0.670):int(w * 0.760)],
            "attributes": img[int(h * 0.680):int(h * 0.810), int(w * 0.080):int(w * 0.850)],
        }

        # 1. OCR Difficulty
        diff_proc = preprocess_difficulty_crop(rois["difficulty"])
        cv2.imwrite("debug_crops/difficulty_proc.png", diff_proc)
        tight_img = clean_and_crop_tight(diff_proc)
        diff_raw = pytesseract.image_to_string(
            tight_img, 
            config="--psm 6 -c tessedit_char_whitelist=0123456789"
        ).strip()
        clean_diff = "".join([c for c in diff_raw if c.isdigit()])
        self.diff_var.set(clean_diff if clean_diff else "0")

        # 2. Template Match Dice
        for idx, key in enumerate(["dice1", "dice2", "dice3"]):
            d_proc = preprocess_dice_crop(rois[key])
            self.last_dice_procs[idx] = d_proc
            cv2.imwrite(f"debug_crops/{key}_proc.png", d_proc)

            if self.dice_templates:
                val = match_digit_template(d_proc, self.dice_templates)
                self.dice_vars[idx].set(str(val))

        # 3. OCR Attributes
        attr_proc = preprocess_text_crop(rois["attributes"])
        attr_raw = pytesseract.image_to_string(attr_proc, config="--psm 6")
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

        for (row_name, column_index), variable in self.holy_dice_vars.items():
            if not variable.get():
                continue

            flat, mult = holy_dice_effect(row_name, column_index)
            column_name = HOLY_DICE_COLUMNS[column_index]
            if flat and mult:
                mod_str = f"{flat:+d} flat, +{mult:.1f}x"
            elif flat:
                mod_str = f"{flat:+d} flat"
            else:
                mod_str = f"+{mult:.1f}x"
            applied_log.append(f"[✓] [Holy {column_name} {row_name}] {mod_str}")
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