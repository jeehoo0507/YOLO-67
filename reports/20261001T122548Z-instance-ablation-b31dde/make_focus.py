"""Crop diagnostic rows from the complete comparison; no box edits."""
from pathlib import Path

from PIL import Image

directory = Path(__file__).resolve().parent
chosen = [9, 15, 3]
canvas = Image.new('RGB', (1152, 56 + 320 * len(chosen)), '#10151e')
for output_row, number in enumerate(chosen):
    with Image.open(directory / f'comparison-{(number-1)//5+1}.jpg') as page:
        if not output_row:
            canvas.paste(page.crop((0, 0, 1152, 56)), (0, 0))
        top = 56 + ((number-1) % 5) * 320
        canvas.paste(page.crop((0, top, 1152, top+320)), (0, 56+output_row*320))
canvas.save(directory / 'diagnostic_examples.jpg', quality=90, optimize=True)
