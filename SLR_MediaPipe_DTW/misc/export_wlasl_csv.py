import csv
import json
import os

from dotenv import load_dotenv

def export_minimal_csv(json_path, n_glosses, csv_path,
                       n_train_samples=-1, n_test_samples=-1, n_val_samples=-1):
    with open(json_path, 'r') as f:
        all_glosses = json.load(f)

    selected_glosses = all_glosses[:n_glosses]

    columns = [
        'video_id', 'glosse', 'glosse_filter', 'gloss_idx', 't_ini_glosse', 't_end_glosse',
        'ini_frame', 'end_frame', 'in_path_frames', 'out_path_frames', 'text_Deutsch',
        'gloss_number', 'gloss_name', 'instance_id', 'split', 'bbox', 'signer_id',
        'variation_id', 'fps', 'frame_start', 'frame_end', 'source', 'url', 'original_idx'
    ]

    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()

        gloss_number = 0
        for gloss_entry in selected_glosses:
            gloss = gloss_entry['gloss']
            split_buckets = {'train': [], 'test': [], 'val': []}

            # Organize instances by split
            for instance in gloss_entry['instances']:
                split = instance.get('split', 'train')
                if split in split_buckets:
                    split_buckets[split].append(instance)

            # Truncate each split based on the provided limits
            selected_instances = []
            if n_train_samples != -1:
                selected_instances.extend(split_buckets['train'][:n_train_samples])
            else:
                selected_instances.extend(split_buckets['train'])

            if n_test_samples != -1:
                selected_instances.extend(split_buckets['test'][:n_test_samples])
            else:
                selected_instances.extend(split_buckets['test'])

            if n_val_samples != -1:
                selected_instances.extend(split_buckets['val'][:n_val_samples])
            else:
                selected_instances.extend(split_buckets['val'])

            for instance in selected_instances:
                split = instance.get('split', 'train')

                writer.writerow({
                    'video_id': -1,
                    'glosse': gloss,
                    'glosse_filter': gloss.lower(),
                    'gloss_idx': -1,
                    't_ini_glosse': -1,
                    't_end_glosse': -1,
                    'ini_frame': -1,
                    'end_frame': -1,
                    'in_path_frames': -1,
                    'out_path_frames': -1,
                    'text_Deutsch': -1,
                    'gloss_number': gloss_number,
                    'gloss_name': gloss,
                    'instance_id': instance['video_id'],
                    'split': split,
                    'bbox': '[-1 -1 -1 -1]',
                    'signer_id': -1,
                    'variation_id': -1,
                    'fps': -1,
                    'frame_start': -1,
                    'frame_end': -1,
                    'source': 'WLASL',
                    'url': '-1',
                    'original_idx': -1
                })

            gloss_number += 1


if __name__ == '__main__':
    n_list=[10,20,30]

    load_dotenv()
    wlasl_metadata_path = os.getenv('WLASL_METADATA_PATH')
    export_paths = [os.getenv('WLASL10_EXPORT_PATH'),
                    os.getenv('WLASL20_EXPORT_PATH'),
                    os.getenv('WLASL30_EXPORT_PATH')
                    ]

    for i,n in enumerate(n_list):
        export_minimal_csv(
            json_path=wlasl_metadata_path,
            n_glosses=n,
            csv_path=export_paths[i],
            n_train_samples=12,
            n_test_samples=2,
            n_val_samples=3
        )
