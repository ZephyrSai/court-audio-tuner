#!/usr/bin/env python3
"""Assembles index.html from src/head.html + an embedded sample WAV (base64) + src/app.js.
Usage: ./build.py [sample.wav]   (default: samples/court-original.wav)"""
import base64, os, sys
root = os.path.dirname(os.path.abspath(__file__))
sample = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, 'samples', 'court-original.wav')
head = open(os.path.join(root, 'src', 'head.html'), encoding='utf-8').read()
app = open(os.path.join(root, 'src', 'app.js'), encoding='utf-8').read()
b64 = base64.b64encode(open(sample, 'rb').read()).decode('ascii')
i = head.index('</style>') + len('</style>')
head_part, body_part = head[:i], head[i:]
page = (
    '<!doctype html>\n<html lang="en">\n<head>\n'
    '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">\n'
    '<meta name="color-scheme" content="light dark">\n'
    '<style>html,body{margin:0}[hidden]{display:none!important}</style>\n'
    + head_part + '\n</head>\n<body>' + body_part +
    '\n<script id="wavdata" type="text/plain">' + b64 + '</script>\n'
    '<script>\n' + app + '\n</script>\n</body>\n</html>\n'
)
out = os.path.join(root, 'index.html')
open(out, 'w', encoding='utf-8').write(page)
print(f'wrote index.html ({len(page.encode()) } bytes) with sample {os.path.relpath(sample, root)}')
