"""
Assistente Gráfico de Recorte Facial (Tkinter + Pillow)
======================================================
Interface visual nativa do Windows (não depende do OpenCV HighGUI).
Abre cada foto, permite desenhar uma caixa sobre o rosto,
mostra preview em tempo real e salva em data/faunaBR/aligned_images (224x224).
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

RAW_DIR = os.path.join("data", "faunaBR", "raw_images")
ALIGNED_DIR = os.path.join("data", "faunaBR", "aligned_images")
TARGET_SIZE = 224
MAX_DISPLAY_W = 850
MAX_DISPLAY_H = 650

# Resampling compatível com qualquer versão do Pillow
LANCZOS_RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS

class FaceCropperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PetFace FaunaBR - Assistente de Recorte Facial")
        self.root.geometry("1180x750")
        self.root.minsize(950, 600)

        # Coletar arquivos
        self.files = self._collect_files()
        self.current_idx = 0

        # Variáveis de estado
        self.orig_img = None
        self.disp_img = None
        self.disp_photo = None
        self.scale = 1.0
        self.start_x = None
        self.start_y = None
        self.rect_id = None
        self.crop_coords = None
        self.preview_photo = None

        self._build_ui()
        self._load_next_image()

    def _collect_files(self):
        species_list = ["onca", "capivara", "arara"]
        files = []
        for sp in species_list:
            sp_dir = os.path.join(RAW_DIR, sp)
            if not os.path.isdir(sp_dir):
                continue
            for root, _, fs in os.walk(sp_dir):
                for f in sorted(fs):
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):
                        files.append((sp, os.path.join(root, f)))
        return files

    def _build_ui(self):
        # Barra superior com instruções
        top_frame = ttk.Frame(self.root, padding=8)
        top_frame.pack(fill=tk.X)

        self.lbl_progress = ttk.Label(
            top_frame, text="", font=("Segoe UI", 12, "bold"), foreground="#0055aa"
        )
        self.lbl_progress.pack(side=tk.LEFT, padx=10)

        lbl_hint = ttk.Label(
            top_frame,
            text="Instrução: Arraste o mouse na cabeça do animal | Tecle ENTER/ESPAÇO para salvar | 'C' para pular",
            font=("Segoe UI", 10),
            foreground="#555555",
        )
        lbl_hint.pack(side=tk.RIGHT, padx=10)

        # Área central dividida
        main_frame = ttk.Frame(self.root, padding=6)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Esquerda: Canvas principal
        self.canvas_frame = ttk.Frame(main_frame, relief=tk.GROOVE)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.canvas = tk.Canvas(self.canvas_frame, bg="#202020", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self._on_button_press)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_button_release)

        # Direita: Painel de Controle e Preview
        right_panel = ttk.Frame(main_frame, width=280)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        right_panel.pack_propagate(False)

        ttk.Label(
            right_panel, text="Preview da Face (224x224):", font=("Segoe UI", 10, "bold")
        ).pack(anchor=tk.W, pady=(5, 5))

        # Canvas do Preview 224x224
        preview_border = ttk.Frame(right_panel, relief=tk.SOLID, borderwidth=1)
        preview_border.pack(anchor=tk.CENTER, pady=5)
        self.preview_canvas = tk.Canvas(preview_border, width=224, height=224, bg="#111111")
        self.preview_canvas.pack()

        self.lbl_species = ttk.Label(right_panel, text="", font=("Segoe UI", 10))
        self.lbl_species.pack(anchor=tk.W, pady=(15, 2))

        self.lbl_dimensions = ttk.Label(right_panel, text="", font=("Segoe UI", 9), foreground="#666")
        self.lbl_dimensions.pack(anchor=tk.W, pady=(0, 15))

        # Botões de Ação
        btn_style = ttk.Style()
        btn_style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))

        self.btn_save = ttk.Button(
            right_panel,
            text="✔ Confirmar Recorte (Enter)",
            style="Accent.TButton",
            command=self._confirm_crop,
        )
        self.btn_save.pack(fill=tk.X, pady=4)

        self.btn_skip = ttk.Button(
            right_panel, text="⏩ Pular Foto (C)", command=self._skip_image
        )
        self.btn_skip.pack(fill=tk.X, pady=4)

        self.btn_reset = ttk.Button(
            right_panel, text="↺ Limpar Seleção", command=self._reset_selection
        )
        self.btn_reset.pack(fill=tk.X, pady=4)

        ttk.Separator(right_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)

        lbl_keys = ttk.Label(
            right_panel,
            text="Atalhos no Teclado:\n• ENTER / ESPAÇO: Salvar\n• C: Pular\n• ESC: Sair",
            font=("Segoe UI", 8),
            foreground="#777777",
            justify=tk.LEFT,
        )
        lbl_keys.pack(anchor=tk.W)

        # Atalhos de teclado
        self.root.bind("<Return>", lambda e: self._confirm_crop())
        self.root.bind("<space>", lambda e: self._confirm_crop())
        self.root.bind("<c>", lambda e: self._skip_image())
        self.root.bind("<C>", lambda e: self._skip_image())
        self.root.bind("<Escape>", lambda e: self.root.destroy())

    def _load_next_image(self):
        # Pular imagens que já foram recortadas anteriormente
        while self.current_idx < len(self.files):
            sp, path = self.files[self.current_idx]
            filename = os.path.splitext(os.path.basename(path))[0] + ".jpg"
            out_path = os.path.join(ALIGNED_DIR, sp, filename)
            if not os.path.exists(out_path):
                break
            self.current_idx += 1

        if self.current_idx >= len(self.files):
            messagebox.showinfo(
                "Concluído!",
                "Parabéns! Todas as fotos foram recortadas e salvas com sucesso em data/faunaBR/aligned_images!",
            )
            self.root.destroy()
            return

        sp, path = self.files[self.current_idx]
        filename = os.path.basename(path)

        try:
            self.orig_img = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Erro", f"Não foi possível abrir {path}:\n{e}")
            self.current_idx += 1
            self._load_next_image()
            return

        orig_w, orig_h = self.orig_img.size

        # Calcular escala para caber na tela
        scale_w = MAX_DISPLAY_W / orig_w
        scale_h = MAX_DISPLAY_H / orig_h
        self.scale = min(1.0, scale_w, scale_h)
        disp_w = int(orig_w * self.scale)
        disp_h = int(orig_h * self.scale)

        self.disp_img = self.orig_img.resize((disp_w, disp_h), resample=LANCZOS_RESAMPLE)
        self.disp_photo = ImageTk.PhotoImage(self.disp_img)

        # Centralizar a imagem no Canvas
        self.canvas.delete("all")
        self.offset_x = (MAX_DISPLAY_W - disp_w) // 2
        self.offset_y = (MAX_DISPLAY_H - disp_h) // 2
        self.canvas.create_image(
            self.offset_x, self.offset_y, image=self.disp_photo, anchor=tk.NW
        )

        # Atualizar Labels
        self.lbl_progress.config(
            text=f"Foto {self.current_idx + 1} de {len(self.files)}: {filename}"
        )
        self.lbl_species.config(text=f"Espécie: {sp.upper()}")
        self.lbl_dimensions.config(text=f"Tamanho original: {orig_w}x{orig_h} px")

        # Resetar preview
        self.crop_coords = None
        self.preview_canvas.delete("all")

    def _on_button_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(
            self.start_x,
            self.start_y,
            self.start_x,
            self.start_y,
            outline="#00FF66",
            width=2,
            dash=(4, 2),
        )

    def _on_mouse_drag(self, event):
        curr_x = event.x
        curr_y = event.y

        # Manter proporção perfeitamente quadrada durante o arrasto
        dx = curr_x - self.start_x
        dy = curr_y - self.start_y
        side = max(abs(dx), abs(dy))
        sign_x = 1 if dx >= 0 else -1
        sign_y = 1 if dy >= 0 else -1

        end_x = self.start_x + sign_x * side
        end_y = self.start_y + sign_y * side

        self.canvas.coords(self.rect_id, self.start_x, self.start_y, end_x, end_y)

    def _on_button_release(self, event):
        coords = self.canvas.coords(self.rect_id)
        if not coords:
            return
        x1, y1, x2, y2 = coords
        left = min(x1, x2)
        top = min(y1, y2)
        right = max(x1, x2)
        bottom = max(y1, y2)

        # Se a caixa for mínima, ignorar
        if (right - left) < 15 or (bottom - top) < 15:
            self._reset_selection()
            return

        # Ajustar para coordenadas relativas da imagem exibida
        img_x1 = left - self.offset_x
        img_y1 = top - self.offset_y
        img_x2 = right - self.offset_x
        img_y2 = bottom - self.offset_y

        # Converter de volta para o tamanho original da foto
        orig_w, orig_h = self.orig_img.size
        ox1 = max(0, int(img_x1 / self.scale))
        oy1 = max(0, int(img_y1 / self.scale))
        ox2 = min(orig_w, int(img_x2 / self.scale))
        oy2 = min(orig_h, int(img_y2 / self.scale))

        # Garantir que seja quadrado no tamanho original
        w = ox2 - ox1
        h = oy2 - oy1
        side = max(w, h)
        cx = ox1 + w // 2
        cy = oy1 + h // 2

        crop_x1 = max(0, cx - side // 2)
        crop_y1 = max(0, cy - side // 2)
        crop_x2 = min(orig_w, crop_x1 + side)
        crop_y2 = min(orig_h, crop_y1 + side)

        actual_side = min(crop_x2 - crop_x1, crop_y2 - crop_y1)
        if actual_side < 20:
            return

        self.crop_coords = (crop_x1, crop_y1, crop_x1 + actual_side, crop_y1 + actual_side)
        self._update_preview()

    def _update_preview(self):
        if not self.crop_coords or not self.orig_img:
            return
        crop = self.orig_img.crop(self.crop_coords)
        crop_resized = crop.resize((TARGET_SIZE, TARGET_SIZE), resample=LANCZOS_RESAMPLE)
        self.preview_photo = ImageTk.PhotoImage(crop_resized)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(0, 0, image=self.preview_photo, anchor=tk.NW)

    def _confirm_crop(self):
        if not self.crop_coords:
            # Se não desenhou caixa, avisar
            messagebox.showwarning(
                "Atenção", "Por favor, clique e arraste com o mouse sobre a face do animal antes de salvar."
            )
            return

        sp, path = self.files[self.current_idx]
        filename = os.path.splitext(os.path.basename(path))[0] + ".jpg"
        out_path = os.path.join(ALIGNED_DIR, sp, filename)

        crop = self.orig_img.crop(self.crop_coords)
        crop_resized = crop.resize((TARGET_SIZE, TARGET_SIZE), resample=LANCZOS_RESAMPLE)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        crop_resized.save(out_path, quality=95)
        print(f"[{self.current_idx + 1}/{len(self.files)}] Salvo: {sp}/{filename} (224x224)")

        self.current_idx += 1
        self._load_next_image()

    def _skip_image(self):
        print(f"[{self.current_idx + 1}/{len(self.files)}] Pulado pelo usuário.")
        self.current_idx += 1
        self._load_next_image()

    def _reset_selection(self):
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None
        self.crop_coords = None
        self.preview_canvas.delete("all")

def main():
    root = tk.Tk()
    app = FaceCropperApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
