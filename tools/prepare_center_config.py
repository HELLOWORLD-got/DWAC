"""Single center-supervised configuration; preserves the spatial run's protocol."""
import json
from pathlib import Path


def main():
    base = json.loads(Path('configs/mdmt_resnet50_spatial.json').read_text())
    base['model'].update(mvoq_center_mode='attention_reference')
    base['training']['center_loss_weight'] = .25
    path = Path('configs/mdmt_resnet50_center.json')
    path.write_text(json.dumps(base, indent=2)+'\n')
    print(path)


if __name__ == '__main__':
    main()
