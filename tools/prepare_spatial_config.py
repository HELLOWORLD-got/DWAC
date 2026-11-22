"""Prepare one complete spatial MVOQ training configuration."""
import json
from pathlib import Path


def main():
    base = json.loads(Path('configs/localization_controls/trainable.json').read_text())
    base['model'].update(backbone='resnet50', num_queries=25, dropout=0.,
                         pretrained=True, freeze_backbone=False,
                         mvoq_position_encoding='normalized_keys_values',
                         mvoq_instance_residual=True)
    base['data'].update(height=416, width=720, augment=False)
    base['training'].update(epochs=15, batch_size=4, workers=4, seed=42,
                            background_weight=.1, freeze_bn_stats=True,
                            freeze_backbone_stem=False, limit_train=None,
                            limit_val=None, clip_grad=None)
    path = Path('configs/mdmt_resnet50_spatial.json')
    path.write_text(json.dumps(base, indent=2)+'\n')
    print(path)


if __name__ == '__main__':
    main()
