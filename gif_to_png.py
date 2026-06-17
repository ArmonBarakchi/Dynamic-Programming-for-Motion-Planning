from PIL import Image, ImageSequence
import math

def gif_to_full_grid(
    gif_path,
    output_path,
    max_height=150,
    cols=5,
    padding=10,              
    bg_color=(255, 255, 255),
    border=2,
    border_color=(0, 0, 0)
):
    gif = Image.open(gif_path)
    frames = [frame.copy().convert("RGB") for frame in ImageSequence.Iterator(gif)]

    resized = []
    for f in frames:
        new_width = int(f.width * max_height / f.height)
        f_resized = f.resize((new_width, max_height))

        if border > 0:
            bordered = Image.new(
                "RGB",
                (f_resized.width + 2*border, f_resized.height + 2*border),
                border_color
            )
            bordered.paste(f_resized, (border, border))
            resized.append(bordered)
        else:
            resized.append(f_resized)

    rows = math.ceil(len(resized) / cols)
    w, h = resized[0].size

    grid_width = cols * w + (cols + 1) * padding
    grid_height = rows * h + (rows + 1) * padding

    grid = Image.new("RGB", (grid_width, grid_height), color=bg_color)

    for i, f in enumerate(resized):
        col = i % cols
        row = i // cols

        x = padding + col * (w + padding)
        y = padding + row * (h + padding)

        grid.paste(f, (x, y))

    grid.save(output_path, "PNG")

