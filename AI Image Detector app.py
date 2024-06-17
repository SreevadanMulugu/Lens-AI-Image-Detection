import tkinter as tk
from tkinter import Canvas, PhotoImage, Scrollbar, scrolledtext, Text, filedialog
from pathlib import Path
from PIL import Image, ImageTk
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tkinterdnd2 import DND_FILES, TkinterDnD

# Initialize the model
torch.set_default_device("cuda")
model = AutoModelForCausalLM.from_pretrained(
    "MILVLG/imp-v1-3b",
    torch_dtype=torch.float16,
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("MILVLG/imp-v1-3b", trust_remote_code=True)

# Global variable to store uploaded image path
uploaded_image_path = ""



# Function to display the uploaded image
def display_image(image_path):
    uploaded_image = Image.open(image_path)
    uploaded_image.thumbnail((450, 450))  # Resize image to fit in the left half
    uploaded_photo = ImageTk.PhotoImage(uploaded_image)

    canvas.create_image(250, 450, image=uploaded_photo)  # Adjusted coordinates to center in left rectangle
    canvas.image = uploaded_photo

# Function to handle drag-and-drop event
def on_drop(event):
    global uploaded_image_path
    uploaded_image_path = event.data.strip('{}')  # Remove extra curly braces if present
    if uploaded_image_path:
        canvas.delete(instruction_text)  # Delete the instruction text
        display_image(uploaded_image_path)

# Function to handle drag-enter event
def on_drag_enter(event):
    canvas.itemconfig(left_rect, fill="black")

def on_drag_leave(event):
    canvas.itemconfig(left_rect, fill="#FFFFFF")

# GUI setup
OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"frame0")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def upload_image():
    global uploaded_image_path
    uploaded_image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif")])
    if uploaded_image_path:
        canvas.delete(instruction_text)  # Delete the instruction text
        display_image(uploaded_image_path)

def on_text_change(event):
    text_widget.update_idletasks()
    height = text_widget.winfo_reqheight()
    text_widget.config(height=height)

def on_drag_start(event):
    global x, y
    x, y = event.x, event.y

def on_drag_motion(event):
    deltax = event.x - x
    deltay = event.y - y
    x_root = window.winfo_x() + deltax
    y_root = window.winfo_y() + deltay
    window.geometry(f"+{x_root}+{y_root}")

def update_canvas_dimensions():
    canvas_width = window.winfo_width()
    canvas_height = window.winfo_height()
    window.after(1000, update_canvas_dimensions)

window = TkinterDnD.Tk()
window.geometry("1080x768")
window.configure(bg="#FFFFFF")
window.overrideredirect(False)
window.outline = "white"

# Make the window resizable
window.resizable(False, False)

canvas_width = window.winfo_width()
canvas_height = window.winfo_height()

canvas = Canvas(
    window,
    bg="#FFFFFF",
    height=768,
    width=1080,
    bd=0,
    highlightthickness=0,
    relief="ridge"
)
canvas.place(x=0, y=0)

# Load the image
image_image_1 = PhotoImage(file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    720.0,
    520.0,
    image=image_image_1
)

canvas_width = 1080
canvas_height = 768

canvas.create_rectangle(
    10.0,
    10.0,
    947.0 * (canvas_width / 1080),
    133.0 * (canvas_height / 768),
    fill="#FFFFFF",
    outline="")

menu_text = canvas.create_text(
    245.75,
    53.0,
    anchor="nw",
    text="Lens: AI Image Detection",
    fill="#000000",
    font=("Gabriela Regular", 55 * -1),
)

canvas.create_rectangle(
    947.25,
    0.0,
    1080.0,
    768.0,
    fill="#FFFFFF",
    outline="black",
    width=2  # Adjust the width to control the thickness of the border
)

canvas.create_text(
    1050.0 * (canvas_width / 1080),
    680.5 * (canvas_height / 768),
    anchor="sw",  # Use "sw" (southwest) anchor for bottom-to-top text
    text="Image Detector",
    fill="#000000",
    font=("Gabriela Regular", 40),
    angle=90  # Rotate text by 90 degrees to the left
)

# New rectangles for the left half
left_rect = canvas.create_rectangle(
    17.0,
    140.0,
    500.0,
    750.0,
    fill="#FFFFFF",
    outline=""
)

canvas.create_rectangle(
    510.0,
    150.0,
    940.0,
    750.0,
    fill="#FFFFFF",
    outline=""
)

# Add the instruction text in italic
instruction_text = canvas.create_text(
    250, 450,  # Adjust coordinates to center the text
    text="Drag and drop your images here",
    fill="#000000",
    font=("Gabriela Regular", 20, "italic"),
    anchor="center"
)

text_widget = scrolledtext.ScrolledText(window, font=("Gabriela Regular", 15), wrap='word')
text_widget.place(x=510.0, y=680, width=430, height=70)

scrollbar = Scrollbar(window, command=text_widget.yview)
scrollbar.place(x=940, y=700, height=40)

text_widget.config(yscrollcommand=scrollbar.set)
text_widget.bind("<KeyRelease>", on_text_change)

# Create a button to upload images at the top right corner
upload_button = tk.Button(window, text="Upload Image", font=("Gabriela Regular", 20), command=upload_image,
                          bg="white", fg="black", relief="flat", activebackground="black", activeforeground="white")
upload_button.place(x=10.0, y=680, width=470, height=70)

# Function to invert colors on mouse enter
def invert_button_colors(event):
    upload_button.config(bg="black", fg="white")

# Function to revert colors on mouse leave
def revert_button_colors(event):
    upload_button.config(bg="white", fg="black")

# Bind events for color inversion on hover
upload_button.bind("<Enter>", invert_button_colors)
upload_button.bind("<Leave>", revert_button_colors)

def generate_response(event=None):
    text = "A chat between a curious user and an artificial intelligence assistant. " \
           "The assistant gives helpful, detailed, and polite answers to the user's questions. " \
           "USER: <image>\nAnalyse the image whether it is generated by any AI Image Generators out there?  ASSISTANT:"
    image = Image.open(uploaded_image_path)

    input_ids = tokenizer(text, return_tensors='pt').input_ids
    if uploaded_image_path:
        image_tensor = model.image_preprocess(image)
    else:
        image_tensor = None

    # Generate response
    output_ids = model.generate(
        input_ids,
        max_length=200,
        images=image_tensor,
        use_cache=True
    )[0]

    response = tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
    message_text.config(state=tk.NORMAL)
    message_text.delete("1.0", tk.END)
    message_text.insert(tk.END, response)
    message_text.config(state=tk.DISABLED)

# Create a frame to act as a button for the user text prompt
process_frame = tk.Frame(window, bg="white", cursor="hand2")
process_frame.place(x=510.0, y=680, width=430, height=70)

process_label = tk.Label(process_frame, text="Process", font=("Gabriela Regular", 20), fg="black", bg="white")
process_label.place(relx=0.5, rely=0.5, anchor="center")

def invert_colors(event):
    process_frame.config(bg="black")
    process_label.config(fg="white", bg="black")

def revert_colors(event):
    process_frame.config(bg="white")
    process_label.config(fg="black", bg="white")

process_frame.bind("<Enter>", invert_colors)
process_frame.bind("<Leave>", revert_colors)
process_frame.bind("<Button-1>", generate_response)

# Text widget to display output
message_text = Text(window, font=("Gabriela Regular", 15), wrap='word')
message_text.place(x=510, y=150, width=430, height=500)

window.resizable(False, False)
window.attributes("-topmost", False)
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
x_root = (screen_width - window.winfo_reqwidth()) // 2
y_root = (screen_height - window.winfo_reqheight()) // 2
window.geometry(f"+{x_root}+{y_root}")

canvas.bind("<ButtonPress-1>", on_drag_start)
canvas.bind("<B1-Motion>", on_drag_motion)

update_canvas_dimensions()

# Configure drag-and-drop
window.drop_target_register(DND_FILES)
window.dnd_bind('<<Drop>>', on_drop)
window.dnd_bind('<<DragEnter>>', on_drag_enter)
window.dnd_bind('<<DragLeave>>', on_drag_leave)

window.mainloop()
