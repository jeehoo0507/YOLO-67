"""Reproduce the fixed 20-image ablation; does not write training labels."""
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from solo.config import ROOT, load_config
from solo.dataset import deterministic_subset, scan_images
from solo.dino import DinoBackbone
from solo.key_preview import COLORS
from solo.maskcut import discover_masks
from solo.pipeline import decode_image
from solo.pseudo_labels import masks_to_boxes
from solo.reporting import new_report
from solo.storage import write_json
from solo.system_info import system_info

dataset = ROOT / 'data/coco-visual-20'
current = load_config(ROOT / 'configs/crowded.toml')
conservative = replace(current,
    maskcut=replace(current.maskcut, max_objects=3, min_mask_area=.01,
                    all_components=False),
    filtering=replace(current.filtering, min_box_area=.01))
split = replace(conservative,
    maskcut=replace(conservative.maskcut, split_depth=2, split_min_patches=12))
variants = [('current', current), ('conservative', conservative), ('conservative_split', split)]
directory, payload = new_report('instance-ablation', current, system_info(), dataset)
payload['command'] = 'project-local uv run --frozen -- python ' + str(Path(__file__).relative_to(ROOT))
payload['timing_notes'] = 'CPU batch1 qualitative ablation, shared DINO keys. Not a throughput benchmark.'
payload['configurations'] = {name: config.to_dict() for name, config in variants}
payload['samples'] = []
payload['warnings'] = ['No ground-truth annotations; counts are not accuracy. No full-dataset qualification.']
shutil.copy2(__file__, directory / 'experiment.py')
shutil.copy2(dataset / 'provenance.json', directory / 'provenance.json')
font = ImageFont.load_default(size=18)
small = ImageFont.load_default(size=14)
records = deterministic_subset(scan_images(dataset), 20, 0)
model = DinoBackbone(current.dino)
payload['backbone'] = model.metadata
pages = [Image.new('RGB', (1152, 56 + 5*320), '#10151e') for _ in range(4)]
for page in pages:
    d = ImageDraw.Draw(page)
    for i, (name, _) in enumerate(variants):
        d.text((384*i + 8, 12), name, fill='white', font=font)
    d.text((8, 36), 'Same 512px DINO v1 keys | all class 0 | no annotations or retraining', fill='#b9c6d8', font=small)
try:
    for i, record in enumerate(records):
        decoded = decode_image(record, current.dino.resolution)
        if 'error' in decoded:
            raise RuntimeError(decoded['error'])
        keys = model.extract(decoded['array'][None])[0]
        with Image.open(record.path) as source:
            original = source.convert('RGB')
        item = {'image': record.relative, 'variants': {}}
        for j, (name, config) in enumerate(variants):
            started = time.perf_counter()
            masks = discover_masks(keys, model.grid, config.maskcut)
            boxes = masks_to_boxes(masks, decoded['size'], config.filtering)
            item['variants'][name] = {'boxes': boxes, 'masks': len(masks), 'seconds': time.perf_counter()-started}
            picture = ImageOps.contain(original, (368, 272))
            d = ImageDraw.Draw(picture)
            for k, (x, y, w, h) in enumerate(boxes):
                corners = ((x-w/2)*picture.width, (y-h/2)*picture.height,
                           (x+w/2)*picture.width, (y+h/2)*picture.height)
                color = COLORS[k % len(COLORS)]
                d.rectangle(corners, outline=color, width=2)
                d.text((corners[0]+2, corners[1]+2), str(k+1), fill=color, font=small)
            tile = Image.new('RGB', (384,320), '#10151e')
            d = ImageDraw.Draw(tile)
            d.text((8,4), f'{i+1:02}. {record.relative} | {len(boxes)} boxes', fill='white', font=small)
            tile.paste(picture, ((384-picture.width)//2, 32+(272-picture.height)//2))
            pages[i//5].paste(tile, (j*384, 56+(i%5)*320))
        payload['samples'].append(item)
        print(f'{i+1}/20 {record.relative}: ' + ', '.join(f'{n}={len(item["variants"][n]["boxes"])}' for n,_ in variants), flush=True)
    payload['status'] = 'complete'
    payload['counts'] = {name: [len(s['variants'][name]['boxes']) for s in payload['samples']] for name,_ in variants}
    for i,page in enumerate(pages,1):
        page.save(directory / f'comparison-{i}.jpg', quality=88, optimize=True)
except BaseException as exc:
    payload['status'] = 'failed'
    payload['errors'].append(f'{type(exc).__name__}: {exc}')
    raise
finally:
    write_json(directory / 'ablation.json', payload)
    print(directory, flush=True)
    print(json.dumps(payload.get('counts', {})), flush=True)
