lines = open('multitask.py', 'r', encoding='utf-8').readlines()
new_lines = []
skip = False
for line in lines:
    if 'Step 3 from README' in line:
        skip = True
        new_lines.append('        if not os.path.exists(_CLASSIFIER_PATH) or not os.path.exists(_LOCALIZER_PATH) or not os.path.exists(_UNET_PATH):\n')
        new_lines.append('            import gdown\n')
        new_lines.append('            gdown.download(id="1TUHvSfGm1nOs6g-tNwjCUwOaa9rx7XkR", output=_CLASSIFIER_PATH, quiet=False)\n')
        new_lines.append('            gdown.download(id="1siFAVhefFU90IdnnMFKYDlBCWHaTj8JC", output=_LOCALIZER_PATH, quiet=False)\n')
        new_lines.append('            gdown.download(id="1Ao6RdfllVhrXwwBKUEFrOyysAC9JsENa", output=_UNET_PATH, quiet=False)\n')
        continue
    if skip and 'gdown.download' in line:
        continue
    if skip and 'import gdown' in line:
        continue
    skip = False
    new_lines.append(line)
open('multitask.py', 'w', encoding='utf-8').writelines(new_lines)
print('Fixed multitask.py')

lines = open('models/multitask.py', 'r', encoding='utf-8').readlines()
new_lines = []
skip = False
for line in lines:
    if 'Step 3 from README' in line:
        skip = True
        new_lines.append('        if not os.path.exists(_CLASSIFIER_PATH) or not os.path.exists(_LOCALIZER_PATH) or not os.path.exists(_UNET_PATH):\n')
        new_lines.append('            import gdown\n')
        new_lines.append('            gdown.download(id="1TUHvSfGm1nOs6g-tNwjCUwOaa9rx7XkR", output=_CLASSIFIER_PATH, quiet=False)\n')
        new_lines.append('            gdown.download(id="1siFAVhefFU90IdnnMFKYDlBCWHaTj8JC", output=_LOCALIZER_PATH, quiet=False)\n')
        new_lines.append('            gdown.download(id="1Ao6RdfllVhrXwwBKUEFrOyysAC9JsENa", output=_UNET_PATH, quiet=False)\n')
        continue
    if skip and 'gdown.download' in line:
        continue
    if skip and 'import gdown' in line:
        continue
    skip = False
    new_lines.append(line)
open('models/multitask.py', 'w', encoding='utf-8').writelines(new_lines)
print('Fixed models/multitask.py')
